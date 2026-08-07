"""Compare RF-GP IMSE after DoE and two cumulative BO rounds.

The controller reads result manifests locally, freezes one 1024-point Sobol
integration set, and sends only compact training arrays to coconutg2.  Worker
mode performs six independent float64 GP fits on CUDA: two RF objectives for
each of the three cumulative training stages.  No CST process or BO campaign
state is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import warnings
from pathlib import Path, PureWindowsPath
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import proposal_relay, qlogehvi  # noqa: E402
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
INTEGRATION_SIZE = 1024
INTEGRATION_SEED = 20260808
DEVICE_ID = "coconutg2"
REMOTE_PYTHON = r"C:\Users\telecom\miniforge3\envs\bocuda\python.exe"
REMOTE_TIMEOUT_SECONDS = 3600.0
PREDICTION_BATCH_SIZE = 256
OBJECTIVE_NAMES = (
    "negative_worst_s11_linear_amplitude",
    qlogehvi.MEAN_TOT_EFF_COLUMN,
)
STAGE_DIRECTORIES = (
    (
        "doe",
        REPOSITORY_ROOT / "results" / "raw" / "doe-round1-lhs-512",
    ),
    (
        "doe_plus_bo1",
        REPOSITORY_ROOT / "results" / "raw" / "msabp-qlogehvi-gpu-001",
    ),
    (
        "doe_plus_bo1_plus_bo2",
        REPOSITORY_ROOT
        / "results"
        / "raw"
        / "msabp-qlogehvi-area-scaled-001",
    ),
)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "processed" / "bo_imse_1024"
REQUEST_PATH = OUTPUT_DIRECTORY / "request.json"
RESPONSE_PATH = OUTPUT_DIRECTORY / "results.json"
SUMMARY_PATH = OUTPUT_DIRECTORY / "summary.csv"


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


def _validate_unit_array(values: Any, *, columns: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{label} must have shape (n, {columns})")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{label} must lie inside [0, 1]")
    return array


def collect_cumulative_training_stages(
    *,
    band_ghz: tuple[float, float] = (3.1, 4.8),
) -> tuple[qlogehvi.InputSpace, list[dict[str, Any]]]:
    """Collect the exact cumulative datasets used by the three comparisons."""

    from scripts.automation import antenna_sampler

    input_space = qlogehvi.input_space_from_sampling_config(
        antenna_sampler.DEFAULT_CONFIG_PATH
    )
    cumulative: list[pd.DataFrame] = []
    stages: list[dict[str, Any]] = []
    for label, source in STAGE_DIRECTORIES:
        skipped: list[str] = []
        incremental = qlogehvi.collect_observations(
            [source],
            band_ghz=band_ghz,
            skipped_incomplete=skipped,
        )
        cumulative.append(incremental)
        combined = pd.concat(cumulative, ignore_index=True)
        train_x, train_y_rf, _, aggregate = qlogehvi.training_arrays(
            combined,
            input_space,
        )
        stages.append(
            {
                "label": label,
                "added_source": str(source.resolve()),
                "incremental_observations": int(len(incremental)),
                "cumulative_observations_raw": int(len(combined)),
                "cumulative_observations_distinct": int(len(train_x)),
                "penalty_observations": int(combined["is_penalty"].sum()),
                "replicate_groups": int((aggregate["replicate_count"] > 1).sum()),
                "skipped_incomplete": skipped,
                "x_unit": train_x.tolist(),
                "y_rf_maximize": train_y_rf.tolist(),
            }
        )
    return input_space, stages


def collect_doe_parity_training_stages(
    *,
    band_ghz: tuple[float, float] = (3.1, 4.8),
) -> tuple[qlogehvi.InputSpace, list[dict[str, Any]]]:
    """Split the stably ordered 498-row DoE into 1-based odd/even rows."""

    from scripts.automation import antenna_sampler

    input_space = qlogehvi.input_space_from_sampling_config(
        antenna_sampler.DEFAULT_CONFIG_PATH
    )
    doe_source = STAGE_DIRECTORIES[0][1]
    observations = qlogehvi.collect_observations(
        [doe_source],
        band_ghz=band_ghz,
    )
    if len(observations) % 2:
        raise ValueError("DoE parity split requires an even observation count")
    stages: list[dict[str, Any]] = []
    for label, begin in (("doe_rows_odd_1based", 0), ("doe_rows_even_1based", 1)):
        subset = observations.iloc[begin::2].reset_index(drop=True)
        train_x, train_y_rf, _, aggregate = qlogehvi.training_arrays(
            subset,
            input_space,
        )
        stages.append(
            {
                "label": label,
                "added_source": str(doe_source.resolve()),
                "partition": "stable observation order, 1-based row parity",
                "observations_raw": int(len(subset)),
                "observations_distinct": int(len(train_x)),
                "penalty_observations": int(subset["is_penalty"].sum()),
                "replicate_groups": int((aggregate["replicate_count"] > 1).sum()),
                "first_case_id": str(subset.iloc[0]["case_id"]),
                "last_case_id": str(subset.iloc[-1]["case_id"]),
                "x_unit": train_x.tolist(),
                "y_rf_maximize": train_y_rf.tolist(),
            }
        )
    return input_space, stages


def build_request_payload(
    input_space: qlogehvi.InputSpace,
    stages: Sequence[Mapping[str, Any]],
    *,
    integration_size: int = INTEGRATION_SIZE,
    integration_seed: int = INTEGRATION_SEED,
    gp_training_steps: int = qlogehvi.ProposalSettings().gp_training_steps,
    gp_fixed_noise_variance: float = (
        qlogehvi.ProposalSettings().gp_fixed_noise_variance
    ),
) -> dict[str, Any]:
    from scipy.stats import qmc

    if integration_size < 2 or not integration_size.bit_count() == 1:
        raise ValueError("integration_size must be a power of two")
    if not stages:
        raise ValueError("at least one training stage is required")
    exponent = integration_size.bit_length() - 1
    integration_x = qmc.Sobol(
        d=len(input_space.names),
        scramble=True,
        seed=integration_seed,
    ).random_base2(exponent)
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "rf_gp_imse",
        "implementation": {
            "analysis_source_sha256": _source_sha256(__file__),
            "qlogehvi_source_sha256": _source_sha256(qlogehvi.__file__),
        },
        "compute": {"device": "cuda", "dtype": "float64"},
        "input_space": input_space.to_dict(),
        "integration": {
            "definition": (
                "mean latent posterior variance over one fixed scrambled "
                "Sobol set in the normalized 23D unit cube"
            ),
            "size": integration_size,
            "seed": integration_seed,
            "x_unit": integration_x.tolist(),
        },
        "gp": {
            "training_steps": int(gp_training_steps),
            "fixed_noise_variance": float(gp_fixed_noise_variance),
            "outcome_transform": None,
            "prediction_batch_size": PREDICTION_BATCH_SIZE,
        },
        "objectives": list(OBJECTIVE_NAMES),
        "stages": [dict(stage) for stage in stages],
    }


def prepare_request(
    path: Path = REQUEST_PATH,
    *,
    doe_parity_split: bool = False,
) -> Path:
    if doe_parity_split:
        input_space, stages = collect_doe_parity_training_stages()
    else:
        input_space, stages = collect_cumulative_training_stages()
    payload = build_request_payload(input_space, stages)
    _atomic_write_json(path, payload)
    for stage in stages:
        raw_count = stage.get(
            "cumulative_observations_raw",
            stage.get("observations_raw"),
        )
        distinct_count = stage.get(
            "cumulative_observations_distinct",
            stage.get("observations_distinct"),
        )
        print(
            f"[IMSE] {stage['label']}: raw={raw_count} "
            f"distinct={distinct_count} "
            f"penalty={stage['penalty_observations']}",
            flush=True,
        )
    print(
        f"[IMSE] fixed integration set: n={INTEGRATION_SIZE}, "
        f"seed={INTEGRATION_SEED}",
        flush=True,
    )
    return path


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def execute_worker_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Perform all six GP fits and fixed-set IMSE evaluations."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported IMSE request schema")
    if payload.get("task") != "rf_gp_imse":
        raise ValueError("unsupported IMSE request task")
    expected_implementation = {
        "analysis_source_sha256": _source_sha256(__file__),
        "qlogehvi_source_sha256": _source_sha256(qlogehvi.__file__),
    }
    if dict(_mapping(payload.get("implementation"), "implementation")) != (
        expected_implementation
    ):
        raise RuntimeError(
            "IMSE worker implementation differs from the controller; "
            "pull the same Git commit on coconutg2"
        )
    compute = _mapping(payload.get("compute"), "compute")
    if compute.get("dtype") != "float64":
        raise ValueError("IMSE worker requires float64")
    device_name = str(compute.get("device", ""))
    input_space = proposal_relay.input_space_from_payload(
        _mapping(payload.get("input_space"), "input_space")
    )
    dimension = len(input_space.names)
    integration = _mapping(payload.get("integration"), "integration")
    integration_x = _validate_unit_array(
        integration.get("x_unit"),
        columns=dimension,
        label="integration.x_unit",
    )
    gp = _mapping(payload.get("gp"), "gp")
    training_steps = int(gp.get("training_steps", 0))
    fixed_noise = float(gp.get("fixed_noise_variance", 0.0))
    prediction_batch_size = int(gp.get("prediction_batch_size", 0))
    if training_steps < 1 or fixed_noise <= 0.0 or prediction_batch_size < 1:
        raise ValueError("invalid GP settings in IMSE request")
    objectives = payload.get("objectives")
    stages = payload.get("stages")
    if list(objectives or ()) != list(OBJECTIVE_NAMES):
        raise ValueError("unexpected IMSE objective list")
    if not isinstance(stages, list) or not stages:
        raise ValueError("IMSE request must contain at least one stage")

    runtime = qlogehvi._botorch_imports()
    torch = runtime["torch"]
    torch.set_default_dtype(torch.float64)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("IMSE worker requires an available CUDA device")
    dtype = torch.float64
    integration_tensor = torch.as_tensor(
        integration_x,
        dtype=dtype,
        device=device,
    )
    results: list[dict[str, Any]] = []
    for stage_index, raw_stage in enumerate(stages):
        stage = _mapping(raw_stage, f"stages[{stage_index}]")
        train_x = _validate_unit_array(
            stage.get("x_unit"),
            columns=dimension,
            label=f"stages[{stage_index}].x_unit",
        )
        train_y = np.asarray(stage.get("y_rf_maximize"), dtype=float)
        if train_y.shape != (len(train_x), 2) or not np.isfinite(train_y).all():
            raise ValueError(f"stages[{stage_index}].y_rf_maximize is invalid")
        x_tensor = torch.as_tensor(train_x, dtype=dtype, device=device)
        for objective_index, objective_name in enumerate(OBJECTIVE_NAMES):
            torch.manual_seed(INTEGRATION_SEED + 10 * stage_index + objective_index)
            y_tensor = torch.as_tensor(
                train_y[:, objective_index : objective_index + 1],
                dtype=dtype,
                device=device,
            )
            y_variance = torch.full_like(y_tensor, fixed_noise)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Data .* is not standardized.*",
                    category=runtime["InputDataWarning"],
                )
                model = runtime["SingleTaskGP"](
                    x_tensor,
                    y_tensor,
                    train_Yvar=y_variance,
                    outcome_transform=None,
                )
            mll = runtime["ExactMarginalLogLikelihood"](
                model.likelihood,
                model,
            )
            started = perf_counter()
            fit_result = runtime["fit_gpytorch_mll_torch"](
                mll,
                step_limit=training_steps,
                timeout_sec=600,
            )
            fit_seconds = perf_counter() - started
            model.eval()
            variances: list[np.ndarray] = []
            with torch.no_grad():
                for begin in range(0, len(integration_tensor), prediction_batch_size):
                    posterior = model.posterior(
                        integration_tensor[begin : begin + prediction_batch_size]
                    )
                    variances.append(
                        posterior.variance.squeeze(-1).detach().cpu().numpy()
                    )
            variance = np.concatenate(variances)
            if variance.shape != (len(integration_x),):
                raise RuntimeError(f"unexpected posterior variance shape {variance.shape}")
            if not np.isfinite(variance).all() or np.any(variance < 0.0):
                raise RuntimeError("posterior variance contains invalid values")
            record = {
                "stage": str(stage.get("label", stage_index)),
                "objective": objective_name,
                "training_observations": int(len(train_x)),
                "imse": float(np.mean(variance)),
                "variance_median": float(np.median(variance)),
                "variance_p95": float(np.quantile(variance, 0.95)),
                "variance_max": float(np.max(variance)),
                "fit_seconds": fit_seconds,
                "fit_status": str(getattr(fit_result, "status", "unknown")),
            }
            results.append(record)
            print(
                f"[IMSE GPU] {record['stage']} / {objective_name}: "
                f"IMSE={record['imse']:.9g}, fit={fit_seconds:.1f}s, "
                f"status={record['fit_status']}",
                flush=True,
            )
            del model, mll, y_tensor, y_variance
            torch.cuda.empty_cache()

    baseline_stage = str(_mapping(stages[0], "stages[0]").get("label", 0))
    baseline = {
        record["objective"]: record["imse"]
        for record in results
        if record["stage"] == baseline_stage
    }
    for record in results:
        ratio = record["imse"] / baseline[record["objective"]]
        record["fraction_of_baseline_imse"] = ratio
        record["reduction_from_baseline_fraction"] = 1.0 - ratio
        if baseline_stage == "doe":
            record["fraction_of_doe_imse"] = ratio
            record["reduction_from_doe_fraction"] = 1.0 - ratio
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "task": "rf_gp_imse",
        "integration": {
            key: integration[key]
            for key in ("definition", "size", "seed")
        },
        "gp": dict(gp),
        "stage_count": len(stages),
        "baseline_stage": baseline_stage,
        "results": results,
        "runtime": {
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_runtime_version": torch.version.cuda,
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "botorch": _package_version("botorch"),
            "gpytorch": _package_version("gpytorch"),
        },
    }


def execute_worker_file(request_path: Path, response_path: Path) -> None:
    payload = json.loads(request_path.read_text(encoding="utf-8-sig"))
    response = execute_worker_payload(_mapping(payload, "request"))
    response["request_sha256"] = proposal_relay.sha256_file(request_path)
    _atomic_write_json(response_path, response)


def relay_to_coconutg2(
    request_path: Path = REQUEST_PATH,
    response_path: Path = RESPONSE_PATH,
) -> Path:
    registry = load_device_registry(DEFAULT_DEVICE_CONFIG_PATH)
    device = registry.get_device(DEVICE_ID)
    if not device.is_remote:
        raise ValueError("coconutg2 must be SSH-addressable")
    request_sha = proposal_relay.sha256_file(request_path)
    remote_root = PureWindowsPath(device.repo_root) / "simulations" / "runs" / "imse"
    remote_request = remote_root / f"bo_imse_1024_{request_sha[:16]}.request.json"
    remote_response = remote_root / f"bo_imse_1024_{request_sha[:16]}.response.json"
    worker = (
        PureWindowsPath(device.repo_root)
        / "scripts"
        / "optimization"
        / "analyze_bo_imse.py"
    )
    push_file_atomic(device, request_path, str(remote_request), overwrite=True)
    script = "\n".join(
        (
            "$ErrorActionPreference = 'Stop'",
            f"$python = {_ps_literal(REMOTE_PYTHON)}",
            f"$worker = {_ps_literal(str(worker))}",
            f"& $python $worker --worker --request {_ps_literal(str(remote_request))} "
            f"--response {_ps_literal(str(remote_response))}",
            "if ($LASTEXITCODE -ne 0) "
            "{ throw ('IMSE GPU worker exited with code ' + $LASTEXITCODE) }",
        )
    )
    completed = run_remote_powershell(
        device,
        script,
        timeout=REMOTE_TIMEOUT_SECONDS,
        action="run six IMSE GP fits",
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    pull_file_atomic(device, str(remote_response), response_path, overwrite=True)
    response = json.loads(response_path.read_text(encoding="utf-8-sig"))
    if response.get("request_sha256") != request_sha:
        raise RuntimeError("IMSE response belongs to a different request")
    return response_path


def write_summary(
    response_path: Path = RESPONSE_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> pd.DataFrame:
    response = json.loads(response_path.read_text(encoding="utf-8-sig"))
    results = response.get("results")
    expected_results = 2 * int(response.get("stage_count", len(results or ()) // 2))
    if (
        not isinstance(results, list)
        or expected_results < 2
        or len(results) != expected_results
    ):
        raise ValueError(
            "IMSE response result count does not match its training stages"
        )
    frame = pd.DataFrame.from_records(results)
    if "fraction_of_baseline_imse" not in frame:
        frame["fraction_of_baseline_imse"] = frame["fraction_of_doe_imse"]
        frame["reduction_from_baseline_fraction"] = frame[
            "reduction_from_doe_fraction"
        ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("[IMSE] results", flush=True)
    print(
        frame[
            [
                "stage",
                "objective",
                "training_observations",
                "imse",
                "fraction_of_baseline_imse",
                "reduction_from_baseline_fraction",
                "fit_seconds",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"[IMSE] JSON: {response_path}", flush=True)
    print(f"[IMSE] CSV: {summary_path}", flush=True)
    return frame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--request", type=Path, default=REQUEST_PATH)
    parser.add_argument("--response", type=Path, default=RESPONSE_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--doe-parity-split",
        action="store_true",
        help="compare stable 1-based odd/even rows of the initial DoE",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        execute_worker_file(args.request.resolve(), args.response.resolve())
        return 0
    if args.summarize_only:
        write_summary(args.response.resolve(), args.summary.resolve())
        return 0
    request_path = prepare_request(
        args.request.resolve(),
        doe_parity_split=args.doe_parity_split,
    )
    if args.prepare_only:
        print(f"[IMSE] request: {request_path}", flush=True)
        return 0
    response_path = relay_to_coconutg2(request_path, args.response.resolve())
    write_summary(response_path, args.summary.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
