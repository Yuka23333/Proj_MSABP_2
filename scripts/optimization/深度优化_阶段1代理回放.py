"""Offline rolling replay for the sixth-round surrogate-model decision.

Round 4 is used only to calibrate uncertainty guards.  Round 5 remains a
chronologically later hold-out set.  The controller relays compact arrays to
``coconutg2``; worker mode performs float64 CUDA GP fitting and never starts
CST or mutates an optimization campaign.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _repository_root() -> Path:
    candidates = (Path.cwd(), *Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "src" / "msabp_opt").is_dir():
            return candidate
    raise RuntimeError("cannot locate repository root containing src/msabp_opt")


REPOSITORY_ROOT = _repository_root()
SRC_ROOT = REPOSITORY_ROOT / "src"
for _import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from msabp_opt.optimization import krvea_data, krvea_relay  # noqa: E402
from msabp_opt.simulation.distributed.config import (  # noqa: E402
    DEFAULT_DEVICE_CONFIG_PATH,
    load_device_registry,
)
from msabp_opt.simulation.distributed.transport import (  # noqa: E402
    pull_file_atomic,
    push_file_atomic,
    run_remote_powershell,
)


SCHEMA_VERSION = 1
DEVICE_ID = "coconutg2"
REMOTE_PYTHON = r"C:\Users\telecom\miniforge3\envs\bocuda\python.exe"
REMOTE_TIMEOUT_SECONDS = 3600.0
REMOTE_WORK_ROOT = PureWindowsPath("simulations", "runs", "deep_surrogate_replay")
ROUND_DIRECTORIES = {
    "round4_calibration": REPOSITORY_ROOT
    / "results"
    / "raw"
    / "msabp-krvea-11var-deep-64-004",
    "round5_holdout": REPOSITORY_ROOT
    / "results"
    / "raw"
    / "msabp-krvea-11var-deep-64-005",
}
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "deep_surrogate_replay_round4_round5"
)
OBJECTIVE_NAMES = krvea_relay.EXPENSIVE_OBJECTIVE_NAMES
VARIANTS = (
    "fixed_1e-6",
    "fixed_1e-4",
    "fixed_1e-3",
    "learned_floor_1e-4",
)
TARGET_COVERAGE = 0.90
ONE_SIDED_Z = 1.6448536269514722
CATASTROPHIC_THRESHOLDS = np.asarray((0.25, 0.25, 5.0), dtype=np.float64)
NORMALIZATION_SCALES = np.asarray((0.25, 0.25, 5.0), dtype=np.float64)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_sha256(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _ps_literal(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if any(character in text for character in ("\0", "\r", "\n")):
        raise ValueError("PowerShell literal contains a control character")
    return "'" + text.replace("'", "''") + "'"


def variant_settings(name: str) -> krvea_relay.SurrogateFitSettings:
    """Return a deliberately uncalibrated setting for one replay variant."""

    base = krvea_relay.SurrogateFitSettings(
        uncertainty_calibration_factors=(1.0, 1.0, 1.0),
        uncertainty_calibration_source="stage1_raw_replay",
    )
    if name == "fixed_1e-6":
        return replace(base, gp_noise_mode="fixed", gp_fixed_noise_variance=1e-6)
    if name == "fixed_1e-4":
        return replace(base, gp_noise_mode="fixed", gp_fixed_noise_variance=1e-4)
    if name == "fixed_1e-3":
        return replace(base, gp_noise_mode="fixed", gp_fixed_noise_variance=1e-3)
    if name == "learned_floor_1e-4":
        return replace(
            base,
            gp_noise_mode="learned",
            gp_learned_noise_floor=1e-4,
            gp_learned_noise_initial_variance=1e-2,
        )
    raise ValueError(f"unknown replay variant: {name}")


def _load_round_state(directory: Path) -> list[Mapping[str, Any]]:
    state = json.loads(
        (directory / "optimization_state.json").read_text(encoding="utf-8-sig")
    )
    batches = state.get("completed_batches")
    if not isinstance(batches, list) or len(batches) != 16:
        raise ValueError(f"expected 16 completed batches in {directory}")
    return sorted(batches, key=lambda item: int(item["batch_index"]))


def _observation_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row in frame.iterrows():
        key = (Path(str(row["source_root"])).name.lower(), str(row["case_id"]))
        if key in lookup:
            raise ValueError(f"duplicate observation identity: {key}")
        lookup[key] = row
    return lookup


def collect_rolling_archive() -> tuple[krvea_data.InputSpace, pd.DataFrame, list[dict[str, Any]]]:
    """Reconstruct 32 proposal-time archives without rereading CST exports."""

    final_cache = ROUND_DIRECTORIES["round5_holdout"] / "_krvea" / "observations.csv"
    observations = pd.read_csv(final_cache)
    round_names = {path.name.lower() for path in ROUND_DIRECTORIES.values()}
    initial_observations = observations.loc[
        ~observations["source_root"]
        .astype(str)
        .map(lambda value: Path(value).name.lower() in round_names)
    ].copy()
    input_space = krvea_data.authoritative_input_space()
    archive = krvea_data.build_dataset(initial_observations, input_space=input_space)
    lookup = _observation_lookup(observations)
    rows = [initial_observations]
    stages: list[dict[str, Any]] = []

    for split, directory in ROUND_DIRECTORIES.items():
        for batch in _load_round_state(directory):
            case_ids = [str(value) for value in batch["case_ids"]]
            unit_values = np.asarray(batch["unit_values"], dtype=np.float64)
            if unit_values.shape != (len(case_ids), len(input_space.names)):
                raise ValueError(f"bad unit_values shape in {split} batch")
            actual_rows = [lookup[(directory.name.lower(), case_id)] for case_id in case_ids]
            actual = pd.DataFrame(actual_rows).reset_index(drop=True)
            training_success = archive.metadata["has_completed_result"].to_numpy(bool)
            diagnostics = _mapping(batch["proposal_diagnostics"], "proposal_diagnostics")
            expected_count = int(_mapping(diagnostics["training"], "training")["training_observations"])
            if expected_count != len(archive.x_unit):
                raise ValueError(
                    f"rolling archive drift at {split} batch {batch['batch_index']}: "
                    f"expected {expected_count}, reconstructed {len(archive.x_unit)}"
                )
            reserved = int(diagnostics.get("reserved_exploration_count", 0))
            stages.append(
                {
                    "split": split,
                    "batch_index": int(batch["batch_index"]),
                    "training_count": int(len(archive.x_unit)),
                    "training_success_count": int(training_success.sum()),
                    "train_x": archive.x_unit[training_success].tolist(),
                    "train_y": archive.objectives[training_success][
                        :, krvea_relay.EXPENSIVE_OBJECTIVE_INDICES
                    ].tolist(),
                    "query_x": unit_values.tolist(),
                    "case_ids": case_ids,
                    "actual_y": actual.loc[:, list(OBJECTIVE_NAMES)].to_numpy(float).tolist(),
                    "is_penalty": actual["is_penalty"].astype(bool).tolist(),
                    "is_reserved_exploration": (
                        [False] * (len(case_ids) - reserved) + [True] * reserved
                    ),
                }
            )
            rows.append(actual)
            archive = krvea_data.build_dataset(
                pd.concat(rows, ignore_index=True), input_space=input_space
            )

    if len(stages) != 32 or len(archive.x_unit) != len(observations):
        raise ValueError("rolling replay did not reconstruct all 32 batches")
    return input_space, observations, stages


def build_request(
    variant: str,
    prepared: tuple[krvea_data.InputSpace, pd.DataFrame, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    input_space, observations, stages = prepared or collect_rolling_archive()
    request = {
        "schema_version": SCHEMA_VERSION,
        "kind": "msabp_deep_surrogate_rolling_replay",
        "variant": variant,
        "device": "cuda",
        "seed": 20260901,
        "objective_names": list(OBJECTIVE_NAMES),
        "input_space": {
            "parameter_names": list(input_space.names),
            "lower": input_space.lower.tolist(),
            "upper": input_space.upper.tolist(),
            "nominal": input_space.nominal.tolist(),
        },
        "settings": asdict(variant_settings(variant)),
        "source": {
            "round4": str(ROUND_DIRECTORIES["round4_calibration"].resolve()),
            "round5": str(ROUND_DIRECTORIES["round5_holdout"].resolve()),
            "final_observations": int(len(observations)),
            "controller_sha256": _source_sha256(__file__),
            "krvea_relay_sha256": _source_sha256(krvea_relay.__file__),
        },
        "stages": stages,
    }
    canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    request["request_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return request


def validate_request(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported replay request schema")
    if payload.get("kind") != "msabp_deep_surrogate_rolling_replay":
        raise ValueError("unexpected replay request kind")
    source = _mapping(payload.get("source"), "source")
    if source.get("controller_sha256") != _source_sha256(__file__):
        raise ValueError("worker source differs from the request controller source")
    if source.get("krvea_relay_sha256") != _source_sha256(krvea_relay.__file__):
        raise ValueError("remote krvea_relay.py differs from the controller dependency")
    variant_settings(str(payload.get("variant")))
    stages = payload.get("stages")
    if not isinstance(stages, list) or len(stages) != 32:
        raise ValueError("replay request must contain 32 rolling stages")
    for stage in stages:
        stage_map = _mapping(stage, "stage")
        train_x = np.asarray(stage_map["train_x"], dtype=np.float64)
        train_y = np.asarray(stage_map["train_y"], dtype=np.float64)
        query_x = np.asarray(stage_map["query_x"], dtype=np.float64)
        actual_y = np.asarray(stage_map["actual_y"], dtype=np.float64)
        if train_x.ndim != 2 or train_x.shape[1] != 11:
            raise ValueError("train_x must have shape (n, 11)")
        if train_y.shape != (len(train_x), 3):
            raise ValueError("train_y must have shape (n, 3)")
        if query_x.ndim != 2 or query_x.shape[1] != 11:
            raise ValueError("query_x must have shape (q, 11)")
        if actual_y.shape != (len(query_x), 3):
            raise ValueError("actual_y must have shape (q, 3)")
        if not all(np.isfinite(array).all() for array in (train_x, train_y, query_x, actual_y)):
            raise ValueError("replay arrays must be finite")
        if np.any(train_x < 0.0) or np.any(train_x > 1.0):
            raise ValueError("train_x lies outside [0, 1]")
        if np.any(query_x < 0.0) or np.any(query_x > 1.0):
            raise ValueError("query_x lies outside [0, 1]")


def run_worker(request_path: Path, response_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    validate_request(request)
    variant = str(request["variant"])
    settings = variant_settings(variant)
    results: list[dict[str, Any]] = []

    for stage_index, stage_value in enumerate(request["stages"]):
        stage = _mapping(stage_value, f"stages[{stage_index}]")
        train_x = np.asarray(stage["train_x"], dtype=np.float64)
        train_y = np.asarray(stage["train_y"], dtype=np.float64)
        query_x = np.asarray(stage["query_x"], dtype=np.float64)
        scaler = krvea_relay.SurrogateTargetScaler.fit(train_y)
        predictor, diagnostics = krvea_relay._fit_surrogate_predictor(
            train_x,
            scaler.transform(train_y),
            settings=settings,
            device_name=str(request["device"]),
            seed=int(request["seed"]) + stage_index,
        )
        latent = predictor(query_x)
        mean, std = scaler.inverse_prediction(
            latent.mean,
            latent.std,
            quadrature_order=settings.bounded_moment_quadrature_order,
        )
        results.append(
            {
                "split": stage["split"],
                "batch_index": int(stage["batch_index"]),
                "case_ids": list(stage["case_ids"]),
                "actual_y": stage["actual_y"],
                "is_penalty": stage["is_penalty"],
                "is_reserved_exploration": stage["is_reserved_exploration"],
                "predicted_mean": mean.tolist(),
                "predicted_std": std.tolist(),
                "training_min": np.min(train_y, axis=0).tolist(),
                "training_max": np.max(train_y, axis=0).tolist(),
                "fit_diagnostics": diagnostics,
                "target_scaler": scaler.to_dict(),
            }
        )
        print(
            f"[Replay:{variant}] {stage_index + 1}/32 "
            f"{stage['split']} batch={stage['batch_index']}",
            flush=True,
        )
        del predictor, latent
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - worker dependency contract
            pass

    response = {
        "schema_version": SCHEMA_VERSION,
        "kind": "msabp_deep_surrogate_rolling_replay_response",
        "variant": variant,
        "request_sha256": request["request_sha256"],
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "stages": results,
    }
    _atomic_write_json(response_path, response)


def _request_and_response_paths(output: Path, variant: str) -> tuple[Path, Path]:
    return output / f"request.{variant}.json", output / f"response.{variant}.json"


def prepare_requests(output: Path, variants: Sequence[str]) -> None:
    prepared = collect_rolling_archive()
    for variant in variants:
        request_path, _ = _request_and_response_paths(output, variant)
        request = build_request(variant, prepared)
        _atomic_write_json(request_path, request)
        print(
            f"[Replay] prepared {variant}: {request_path} "
            f"({request_path.stat().st_size / 1024 / 1024:.1f} MiB)"
        )


def relay_variant(output: Path, variant: str) -> None:
    request_path, response_path = _request_and_response_paths(output, variant)
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    if response_path.is_file():
        response = json.loads(response_path.read_text(encoding="utf-8-sig"))
        if response.get("request_sha256") == request.get("request_sha256"):
            print(f"[Replay] cached response is current: {variant}")
            return

    registry = load_device_registry(DEFAULT_DEVICE_CONFIG_PATH)
    device = registry.get_device(DEVICE_ID)
    remote_directory = (
        PureWindowsPath(device.repo_root)
        / REMOTE_WORK_ROOT
        / str(request["request_sha256"])
    )
    remote_script = remote_directory / "deep_surrogate_replay_worker.py"
    remote_request = remote_directory / f"request.{variant}.json"
    remote_response = remote_directory / f"response.{variant}.json"
    push_file_atomic(device, __file__, str(remote_script), transfer_timeout=180.0)
    push_file_atomic(
        device, request_path, str(remote_request), transfer_timeout=180.0
    )
    command = (
        f"Set-Location -LiteralPath {_ps_literal(device.repo_root)}; "
        f"& {_ps_literal(REMOTE_PYTHON)} {_ps_literal(remote_script)} --worker "
        f"--request {_ps_literal(remote_request)} --response {_ps_literal(remote_response)}"
    )
    print(f"[Replay] running {variant} on {DEVICE_ID} (32 rolling fits)")
    completed = run_remote_powershell(
        device,
        command,
        timeout=REMOTE_TIMEOUT_SECONDS,
        action=f"run {variant} deep surrogate replay",
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    pull_file_atomic(
        device, str(remote_response), response_path, transfer_timeout=180.0
    )
    response = json.loads(response_path.read_text(encoding="utf-8-sig"))
    if response.get("request_sha256") != request.get("request_sha256"):
        raise RuntimeError(f"response/request hash mismatch for {variant}")


def responses_to_frame(output: Path, variants: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for variant in variants:
        _, response_path = _request_and_response_paths(output, variant)
        response = json.loads(response_path.read_text(encoding="utf-8-sig"))
        for stage_value in response["stages"]:
            stage = _mapping(stage_value, "response stage")
            actual = np.asarray(stage["actual_y"], dtype=np.float64)
            mean = np.asarray(stage["predicted_mean"], dtype=np.float64)
            std = np.asarray(stage["predicted_std"], dtype=np.float64)
            training_min = np.asarray(stage["training_min"], dtype=np.float64)
            training_max = np.asarray(stage["training_max"], dtype=np.float64)
            for row_index, case_id in enumerate(stage["case_ids"]):
                for objective_index, objective_name in enumerate(OBJECTIVE_NAMES):
                    records.append(
                        {
                            "method": variant,
                            "split": stage["split"],
                            "batch_index": int(stage["batch_index"]),
                            "case_id": str(case_id),
                            "is_penalty": bool(stage["is_penalty"][row_index]),
                            "is_reserved_exploration": bool(
                                stage["is_reserved_exploration"][row_index]
                            ),
                            "objective_index": objective_index,
                            "objective": objective_name,
                            "actual": actual[row_index, objective_index],
                            "mean": mean[row_index, objective_index],
                            "std": std[row_index, objective_index],
                            "training_min": training_min[objective_index],
                            "training_max": training_max[objective_index],
                        }
                    )
    return pd.DataFrame.from_records(records)


def add_ensemble_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add robust median ensembles, with and without archive-range clipping."""

    key = [
        "split",
        "batch_index",
        "case_id",
        "is_penalty",
        "is_reserved_exploration",
        "objective_index",
        "objective",
    ]
    ensemble_rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(key, sort=False, dropna=False):
        means = group["mean"].to_numpy(dtype=np.float64)
        stds = group["std"].to_numpy(dtype=np.float64)
        center = float(np.median(means))
        between = float(1.482602218505602 * np.median(np.abs(means - center)))
        combined_std = float(np.sqrt(np.median(stds**2) + between**2))
        base = dict(zip(key, values, strict=True))
        training_min = float(group["training_min"].iloc[0])
        training_max = float(group["training_max"].iloc[0])
        common = {
            **base,
            "actual": float(group["actual"].iloc[0]),
            "std": combined_std,
            "training_min": training_min,
            "training_max": training_max,
        }
        ensemble_rows.append({**common, "method": "ensemble_median", "mean": center})
        ensemble_rows.append(
            {
                **common,
                "method": "ensemble_median_clipped",
                "mean": float(np.clip(center, training_min, training_max)),
            }
        )
    return pd.concat([frame, pd.DataFrame.from_records(ensemble_rows)], ignore_index=True)


def calibrate_residual_floors(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Freeze a physical-unit one-sided residual floor from successful round 4."""

    calibration = frame.loc[
        (frame["split"] == "round4_calibration") & ~frame["is_penalty"]
    ]
    result: dict[str, dict[str, float]] = {}
    for (method, objective), group in calibration.groupby(["method", "objective"]):
        underprediction = np.maximum(
            group["actual"].to_numpy(float) - group["mean"].to_numpy(float), 0.0
        )
        raw_std = np.maximum(group["std"].to_numpy(float), 1e-12)
        result.setdefault(str(method), {})[str(objective)] = {
            "physical_std_floor": float(np.quantile(underprediction, 0.95) / ONE_SIDED_Z),
            "q95_scalar_factor": float(
                np.quantile(underprediction / raw_std, 0.95) / ONE_SIDED_Z
            ),
            "calibration_rows": int(len(group)),
        }
    return result


def apply_calibration(
    frame: pd.DataFrame, calibration: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> pd.DataFrame:
    calibrated = frame.copy()
    floors = []
    for row in calibrated.itertuples(index=False):
        floors.append(
            float(calibration[str(row.method)][str(row.objective)]["physical_std_floor"])
        )
    calibrated["physical_std_floor"] = floors
    calibrated["guarded_std"] = np.maximum(calibrated["std"], calibrated["physical_std_floor"])
    calibrated["upper_guard"] = calibrated["mean"] + ONE_SIDED_Z * calibrated["guarded_std"]
    calibrated["covered"] = calibrated["actual"] <= calibrated["upper_guard"]
    calibrated["absolute_error"] = np.abs(calibrated["actual"] - calibrated["mean"])
    calibrated["outside_training_range"] = (
        (calibrated["mean"] < calibrated["training_min"])
        | (calibrated["mean"] > calibrated["training_max"])
    )
    thresholds = CATASTROPHIC_THRESHOLDS[
        calibrated["objective_index"].to_numpy(dtype=int)
    ]
    calibrated["catastrophic_optimism"] = (
        calibrated["actual"] - calibrated["mean"]
    ) > thresholds
    return calibrated


def evaluate_holdout(frame: pd.DataFrame) -> pd.DataFrame:
    holdout = frame.loc[
        (frame["split"] == "round5_holdout") & ~frame["is_penalty"]
    ].copy()
    records: list[dict[str, Any]] = []
    for (method, objective_index, objective), group in holdout.groupby(
        ["method", "objective_index", "objective"], sort=False
    ):
        error = group["actual"].to_numpy(float) - group["mean"].to_numpy(float)
        records.append(
            {
                "method": method,
                "objective_index": int(objective_index),
                "objective": objective,
                "rows": int(len(group)),
                "coverage": float(group["covered"].mean()),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "p95_absolute_error": float(np.quantile(np.abs(error), 0.95)),
                "outside_training_range": int(group["outside_training_range"].sum()),
                "catastrophic_optimism": int(group["catastrophic_optimism"].sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def rank_methods(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in summary.groupby("method", sort=False):
        coverage_shortfall = np.maximum(TARGET_COVERAGE - group["coverage"], 0.0)
        normalized_mae = group["mae"].to_numpy(float) / NORMALIZATION_SCALES[
            group["objective_index"].to_numpy(int)
        ]
        rows.append(
            {
                "method": method,
                "catastrophic_optimism": int(group["catastrophic_optimism"].sum()),
                "coverage_shortfall": float(coverage_shortfall.sum()),
                "outside_training_range": int(group["outside_training_range"].sum()),
                "normalized_mae": float(normalized_mae.mean()),
                "mean_coverage": float(group["coverage"].mean()),
            }
        )
    ranking = pd.DataFrame.from_records(rows)
    ranking.sort_values(
        [
            "catastrophic_optimism",
            "coverage_shortfall",
            "outside_training_range",
            "normalized_mae",
        ],
        kind="stable",
        inplace=True,
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking.reset_index(drop=True)


def write_report(
    output: Path,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    calibration: Mapping[str, Any],
    variants: Sequence[str],
) -> Path:
    successful = predictions.loc[~predictions["is_penalty"]]
    fit_seconds: dict[str, float] = {}
    learned_noise: dict[str, list[float]] = {}
    for variant in variants:
        _, response_path = _request_and_response_paths(output, variant)
        response = json.loads(response_path.read_text(encoding="utf-8-sig"))
        diagnostics = [stage["fit_diagnostics"] for stage in response["stages"]]
        fit_seconds[variant] = float(
            sum(float(item["gp_fit_seconds"]) for item in diagnostics)
        )
        noise = [
            value
            for item in diagnostics
            for value in (item.get("gp_learned_noise") or [])
        ]
        learned_noise[variant] = [float(value) for value in noise]

    lines = [
        "# Deep optimization Stage 1: rolling surrogate replay",
        "",
        "This is an offline decision experiment. It did not launch CST or modify any campaign.",
        "Round 4 supplied uncertainty calibration only; Round 5 was held out chronologically.",
        "",
        "## Data contract",
        "",
        "- Initial archive: 768 distinct observations (760 successful, 8 penalties).",
        f"- Calibration: 64 Round-4 proposals, "
        f"{int((predictions.loc[predictions['split'] == 'round4_calibration', ['case_id', 'is_penalty']].drop_duplicates()['is_penalty'] == False).sum())} successful.",  # noqa: E712
        f"- Hold-out: 64 Round-5 proposals, "
        f"{int((predictions.loc[predictions['split'] == 'round5_holdout', ['case_id', 'is_penalty']].drop_duplicates()['is_penalty'] == False).sum())} successful.",  # noqa: E712
        "- Each batch was predicted by a model fitted only to observations available before that batch.",
        "- Geometry/CST penalties were appended chronologically but excluded from GP fitting and scoring.",
        "",
        "## Held-out ranking",
        "",
        "Ranking is lexicographic: catastrophic optimistic errors, coverage shortfall below 90%, "
        "predictions outside the observed target range, then normalized MAE.",
        "",
        "| rank | method | catastrophic | coverage shortfall | outside range | normalized MAE | mean coverage |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranking.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.method} | {row.catastrophic_optimism} | "
            f"{row.coverage_shortfall:.3f} | {row.outside_training_range} | "
            f"{row.normalized_mae:.4f} | {row.mean_coverage:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Per-objective held-out metrics",
            "",
            "| method | objective | coverage | MAE | RMSE | p95 abs error | catastrophic | outside |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered = summary.merge(ranking[["method", "rank"]], on="method").sort_values(
        ["rank", "objective_index"]
    )
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.objective} | {row.coverage:.1%} | "
            f"{row.mae:.6g} | {row.rmse:.6g} | {row.p95_absolute_error:.6g} | "
            f"{row.catastrophic_optimism} | {row.outside_training_range} |"
        )

    lines.extend(["", "## Runtime", ""])
    for variant, seconds in fit_seconds.items():
        noise = learned_noise[variant]
        suffix = ""
        if noise:
            suffix = f"; learned-noise median={np.median(noise):.6g}"
        lines.append(f"- `{variant}`: {seconds:.1f} s total GPU fit time{suffix}.")
    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "The physical residual floors are frozen from Round 4. Round-5 coverage is therefore "
            "a genuine temporal check, not an in-sample calibration score. The chosen method is a "
            "candidate for Stage 2 only; the 128-point CST budget has not been spent.",
            "",
            f"Calibration details: `{(output / 'calibration.json').name}`. "
            f"Successful prediction rows: {len(successful)}.",
        ]
    )
    report_path = output / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def summarize(output: Path, variants: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = responses_to_frame(output, variants)
    predictions = add_ensemble_predictions(base)
    calibration = calibrate_residual_floors(predictions)
    predictions = apply_calibration(predictions, calibration)
    summary = evaluate_holdout(predictions)
    ranking = rank_methods(summary)
    _atomic_write_csv(output / "predictions.csv", predictions)
    _atomic_write_csv(output / "summary.csv", summary)
    _atomic_write_csv(output / "ranking.csv", ranking)
    _atomic_write_json(output / "calibration.json", calibration)
    report_path = write_report(
        output, predictions, summary, ranking, calibration, variants
    )
    print(f"[Replay] best Stage-2 candidate: {ranking.iloc[0]['method']}")
    print(f"[Replay] report: {report_path}")
    return summary, ranking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument(
        "--variant",
        action="append",
        choices=VARIANTS,
        help="repeat to run a subset; default runs all four variants",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="build and validate requests without contacting CoconutG2",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="summarize already downloaded responses",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.request is None or args.response is None:
            raise SystemExit("--worker requires --request and --response")
        run_worker(args.request, args.response)
        return 0

    variants = tuple(args.variant or VARIANTS)
    output = args.output.resolve()
    if args.summarize_only:
        summarize(output, variants)
        return 0
    prepare_requests(output, variants)
    if args.prepare_only:
        return 0
    for variant in variants:
        relay_variant(output, variant)
    summarize(output, variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
