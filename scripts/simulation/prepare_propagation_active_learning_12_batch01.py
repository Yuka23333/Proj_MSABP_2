"""Prepare active-learning propagation batch 01: 8 calibration + 4 exploitation.

The candidate pool contains completed archive designs whose worst in-band
return loss is at least 7 dB and whose FFS is already available. Previously
propagated antenna geometries are excluded.

The current spherical ROI is extracted once from each eligible FFS. A small
bootstrap ridge ensemble predicts September S21 power and BER-v2 Eb/N0 from
rank-transformed far-field features. Calibration points combine predictive
uncertainty with distance from existing labels; exploitation points maximize
a joint optimistic score while remaining non-dominated in S11, September
radiation efficiency, and area. Selection is deterministic and fully audited.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import krvea_data  # noqa: E402
from scripts.automation import antenna_sampler  # noqa: E402
from scripts.postprocessing import ieee802156_three_channel  # noqa: E402
from scripts.postprocessing.cap_gain import parse_ffs  # noqa: E402
from scripts.postprocessing.prepare_link_ffs_tensor import _pattern_power  # noqa: E402
from scripts.postprocessing.search_link_spherical_roi import (  # noqa: E402
    region_feature,
    spherical_theta_cell_weights,
    symmetric_phi_mask,
)


DEFAULT_FREQUENCY_TABLE = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "onbody_proxies_per_frequency_3p1-4p8GHz.csv"
)
DEFAULT_ROI_RESULT = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "spherical_roi_13"
    / "result.json"
)
DEFAULT_ROI_CASE_METRICS = DEFAULT_ROI_RESULT.parent / "optimal_roi_case_metrics.csv"
DEFAULT_LABELED_TABLE = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "link_correlation_13"
    / "selected_13_proxy_frequency_table.csv"
)
DEFAULT_PREVIOUS_WORKLIST = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_selected_13.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_active_learning_12_batch01.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "active_learning_batch01"
)
DEFAULT_FEATURE_CACHE = DEFAULT_OUTPUT_DIRECTORY / "eligible_roi_features.csv"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIRECTORY / "candidate_predictions_frozen.csv"
DEFAULT_AUDIT = DEFAULT_OUTPUT_DIRECTORY / "selection_audit.csv"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIRECTORY / "selection_manifest.json"
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
DEFAULT_RANDOM_SEED = 20260908
DEFAULT_BOOTSTRAPS = 256
RETURN_LOSS_MIN_DB = 7.0
SEPTEMBER_ROBUSTNESS_LAMBDA = 0.5
CALIBRATION_COUNT = 8
EXPLOITATION_COUNT = 4
SIMULATION_MODE = "propagation_s21"
REFERENCE_CASE_NAME = "roblin_wei_2012_reference"
GRID_TOLERANCE = 1.0e-10
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)

PROXY_FEATURES: tuple[tuple[str, str, float], ...] = (
    ("p_cap_f4p0", "P_cap", 4.0),
    ("p_hor_f3p7", "P_hor", 3.7),
    ("chi_tm_f3p7", "chi_TM", 3.7),
    ("chi_tm_f3p8", "chi_TM", 3.8),
    ("g_theta_hor_min_f3p6", "G_theta_hor_min", 3.6),
    (
        "g_endfire_phi90_vertical_f4p0",
        "G_endfire_phi90_vertical",
        4.0,
    ),
    ("g_endfire_phi90_total_f4p0", "G_endfire_phi90_total", 4.0),
)


@dataclass(frozen=True)
class RoiContract:
    component: str
    frequency_min_ghz: float
    frequency_max_ghz: float
    theta_min_deg: float
    theta_max_deg: float
    phi_center_deg: float
    phi_half_width_deg: float


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(source: Any, case_id: Any) -> str:
    return f"{str(source).strip()}::{str(case_id).strip()}"


def _return_loss_db(amplitude: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(amplitude, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    result[valid] = -20.0 * np.log10(values[valid])
    return result


def _september_rad_efficiency(group: pd.DataFrame) -> float:
    frequency_hz = group["freq_ghz"].to_numpy(dtype=np.float64) * 1.0e9
    efficiency = group["Rad_Eff_from_ffs"].to_numpy(dtype=np.float64)
    valid = np.isfinite(efficiency) & (efficiency >= 0.0) & (efficiency <= 1.0)
    frequency_hz = frequency_hz[valid]
    efficiency = efficiency[valid]
    if len(efficiency) < 2:
        raise ValueError("too few valid FFS radiation-efficiency samples")
    channel_values = np.asarray(
        [
            ieee802156_three_channel.quadratic_weighted_mean(
                frequency_hz,
                efficiency,
                channel,
            )
            for channel in ieee802156_three_channel.LOW_BAND_CHANNELS
        ],
        dtype=np.float64,
    )
    return float(
        ieee802156_three_channel.aggregate_mandatory_worst_utility(
            channel_values,
            SEPTEMBER_ROBUSTNESS_LAMBDA,
        )
    )


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _load_roi_contract(path: Path) -> RoiContract:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    optimum = payload["optimal_roi"]
    seed = payload["seed"]
    if payload["field_family"] != "directivity":
        raise ValueError("batch-01 selector requires the directivity ROI")
    return RoiContract(
        component=str(payload["component"]),
        frequency_min_ghz=float(optimum["frequency_min_ghz"]),
        frequency_max_ghz=float(optimum["frequency_max_ghz"]),
        theta_min_deg=float(optimum["theta_min_deg"]),
        theta_max_deg=float(optimum["theta_max_deg"]),
        phi_center_deg=float(seed["phi_deg"]),
        phi_half_width_deg=float(optimum["phi_half_width_deg"]),
    )


def _roi_contract_sha256(contract: RoiContract) -> str:
    encoded = json.dumps(
        contract.__dict__,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _grid_interval(
    grid: np.ndarray, lower: float, upper: float, label: str
) -> tuple[int, int]:
    selected = np.flatnonzero(
        (grid >= lower - GRID_TOLERANCE) & (grid <= upper + GRID_TOLERANCE)
    )
    if len(selected) == 0:
        raise ValueError(f"{label} interval contains no samples")
    if not np.isclose(grid[selected[0]], lower, atol=GRID_TOLERANCE, rtol=0.0):
        raise ValueError(f"{label} lower bound is absent from the FFS grid")
    if not np.isclose(grid[selected[-1]], upper, atol=GRID_TOLERANCE, rtol=0.0):
        raise ValueError(f"{label} upper bound is absent from the FFS grid")
    return int(selected[0]), int(selected[-1])


def _extract_roi_worker(
    record: Mapping[str, Any],
    contract: RoiContract,
) -> dict[str, Any]:
    source = Path(str(record["ffs_path"])).expanduser().resolve()
    ffs = parse_ffs(source)
    frequency = np.asarray(ffs["freq"], dtype=np.float64) / 1.0e9
    theta = np.asarray(ffs["theta_deg"], dtype=np.float64)
    phi_full = np.asarray(ffs["phi_deg"], dtype=np.float64)
    if not (np.isclose(phi_full[0], 0.0) and np.isclose(phi_full[-1], 360.0)):
        raise ValueError(f"FFS does not contain the 0/360 phi endpoints: {source}")
    e_theta_full = np.asarray(ffs["E_theta"], dtype=np.complex128)
    e_phi_full = np.asarray(ffs["E_phi"], dtype=np.complex128)
    pattern_power = _pattern_power(
        e_theta_full,
        e_phi_full,
        theta,
        phi_full,
    )
    phi = phi_full[:-1]
    intensities = {
        "theta": np.abs(e_theta_full[:, :-1]) ** 2,
        "phi": np.abs(e_phi_full[:, :-1]) ** 2,
    }
    intensities["total"] = intensities["theta"] + intensities["phi"]
    component = intensities[contract.component]
    directivity = (4.0 * math.pi * component / pattern_power[:, None, None])[None, ...]
    frequency_interval = _grid_interval(
        frequency,
        contract.frequency_min_ghz,
        contract.frequency_max_ghz,
        "frequency",
    )
    theta_interval = _grid_interval(
        theta,
        contract.theta_min_deg,
        contract.theta_max_deg,
        "theta",
    )
    phi_mask = symmetric_phi_mask(
        phi,
        contract.phi_center_deg,
        contract.phi_half_width_deg,
    )
    roi_metric = float(
        region_feature(
            directivity,
            theta_low=theta_interval[0],
            theta_high=theta_interval[1],
            phi_mask=phi_mask,
            frequency_interval=frequency_interval,
            theta_weights=spherical_theta_cell_weights(theta),
        )[0]
    )
    return {
        "sample_index": int(record["sample_index"]),
        "identity": str(record["identity"]),
        "roi_metric_linear": roi_metric,
        "ffs_path": str(source),
        "ffs_sha256": str(record["ffs_sha256"]),
    }


def _load_candidate_metadata(
    frequency_table_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_frequency = pd.read_csv(frequency_table_path)
    required = {
        "sample_index",
        "sample_kind",
        "source",
        "case_id",
        "case_directory",
        "ffs_path",
        "ffs_sha256",
        "worst_s11_linear_amplitude",
        "mean_total_efficiency_linear",
        "normalized_substrate_area",
        "Rad_Eff_from_ffs",
        "freq_ghz",
        *(column for _, column, _ in PROXY_FEATURES),
    }
    missing = sorted(required.difference(per_frequency.columns))
    if missing:
        raise ValueError(f"frequency table lacks columns: {missing}")
    radiation_efficiency = per_frequency.groupby(
        "sample_index",
        sort=False,
    ).apply(_september_rad_efficiency, include_groups=False)
    per_frequency["rad_eff_september_linear"] = per_frequency["sample_index"].map(
        radiation_efficiency
    )
    metadata = per_frequency.drop_duplicates("sample_index").copy()
    metadata["identity"] = [
        _identity(source, case_id)
        for source, case_id in zip(
            metadata["source"],
            metadata["case_id"],
            strict=True,
        )
    ]
    metadata["return_loss_db"] = _return_loss_db(metadata["worst_s11_linear_amplitude"])
    eligible = metadata.loc[
        metadata["sample_kind"].eq("archive")
        & (metadata["return_loss_db"] >= RETURN_LOSS_MIN_DB)
    ].copy()
    if eligible["identity"].duplicated().any():
        raise ValueError("eligible archive identities are not unique")
    return per_frequency, eligible.reset_index(drop=True)


def _extract_roi_features(
    eligible: pd.DataFrame,
    contract: RoiContract,
    *,
    workers: int,
) -> pd.DataFrame:
    records = eligible[["sample_index", "identity", "ffs_path", "ffs_sha256"]].to_dict(
        "records"
    )
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_extract_roi_worker, record, contract) for record in records
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            rows.append(future.result())
            if completed == 1 or completed % 10 == 0 or completed == len(futures):
                print(f"[ActiveLearning-01] ROI parsed {completed}/{len(futures)}")
    return pd.DataFrame(rows).sort_values("sample_index", kind="stable")


def _load_or_create_roi_cache(
    eligible: pd.DataFrame,
    contract: RoiContract,
    cache_path: Path,
    *,
    workers: int,
    overwrite: bool,
) -> pd.DataFrame:
    contract_hash = _roi_contract_sha256(contract)
    if cache_path.exists() and not overwrite:
        cached = pd.read_csv(cache_path)
        required = {
            "sample_index",
            "identity",
            "roi_metric_linear",
            "ffs_sha256",
            "roi_contract_sha256",
        }
        missing = sorted(required.difference(cached.columns))
        if missing:
            raise ValueError(f"ROI cache lacks columns: {missing}")
        expected = eligible[["sample_index", "identity", "ffs_sha256"]].copy()
        check = expected.merge(
            cached[list(required)],
            on=["sample_index", "identity"],
            how="left",
            suffixes=("_expected", "_cached"),
            validate="one_to_one",
        )
        hashes_match = (
            check["ffs_sha256_expected"]
            .astype(str)
            .str.casefold()
            .eq(check["ffs_sha256_cached"].astype(str).str.casefold())
        )
        contract_matches = cached["roi_contract_sha256"].astype(str).eq(contract_hash)
        if (
            len(cached) != len(expected)
            or not hashes_match.all()
            or not contract_matches.all()
        ):
            raise ValueError(
                "ROI cache does not match the current eligible FFS population; "
                "rerun with --overwrite"
            )
        return cached.sort_values("sample_index", kind="stable")

    extracted = _extract_roi_features(eligible, contract, workers=workers)
    extracted["roi_contract_sha256"] = contract_hash
    _write_csv_atomic(extracted, cache_path)
    return extracted


def _feature_at_frequency(
    per_frequency: pd.DataFrame,
    sample_indices: np.ndarray,
    column: str,
    frequency_ghz: float,
) -> pd.Series:
    rows = per_frequency.loc[
        per_frequency["sample_index"].isin(sample_indices)
        & np.isclose(
            per_frequency["freq_ghz"].to_numpy(dtype=np.float64),
            frequency_ghz,
            atol=GRID_TOLERANCE,
            rtol=0.0,
        ),
        ["sample_index", column],
    ]
    if rows["sample_index"].duplicated().any() or len(rows) != len(sample_indices):
        raise ValueError(f"incomplete proxy feature {column}@{frequency_ghz:g} GHz")
    return rows.set_index("sample_index")[column].reindex(sample_indices)


def _reference_roi_metric(path: Path) -> float:
    frame = pd.read_csv(path)
    selected = frame.loc[frame["case_name"].eq(REFERENCE_CASE_NAME)]
    if len(selected) != 1:
        raise ValueError("optimal ROI case table must contain one reference row")
    return float(selected.iloc[0]["roi_metric_linear"])


def _build_model_feature_table(
    per_frequency: pd.DataFrame,
    eligible: pd.DataFrame,
    roi_features: pd.DataFrame,
    roi_case_metrics_path: Path,
) -> pd.DataFrame:
    reference = per_frequency.loc[
        per_frequency["source"].eq(REFERENCE_CASE_NAME)
        & per_frequency["case_id"].astype(str).eq(REFERENCE_CASE_NAME)
    ].drop_duplicates("sample_index")
    if len(reference) != 1:
        raise ValueError("frequency table must contain one Roblin-Wei reference")
    population = pd.concat(
        [
            eligible[
                [
                    "sample_index",
                    "identity",
                    "source",
                    "case_id",
                    "case_directory",
                    "worst_s11_linear_amplitude",
                    "mean_total_efficiency_linear",
                    "rad_eff_september_linear",
                    "normalized_substrate_area",
                    "return_loss_db",
                ]
            ],
            reference.assign(
                identity=[
                    _identity(reference.iloc[0]["source"], reference.iloc[0]["case_id"])
                ],
                return_loss_db=_return_loss_db(reference["worst_s11_linear_amplitude"]),
            )[
                [
                    "sample_index",
                    "identity",
                    "source",
                    "case_id",
                    "case_directory",
                    "worst_s11_linear_amplitude",
                    "mean_total_efficiency_linear",
                    "rad_eff_september_linear",
                    "normalized_substrate_area",
                    "return_loss_db",
                ]
            ],
        ],
        ignore_index=True,
    )
    population = population.merge(
        roi_features[["sample_index", "roi_metric_linear"]],
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    reference_index = int(reference.iloc[0]["sample_index"])
    population.loc[
        population["sample_index"].eq(reference_index),
        "roi_metric_linear",
    ] = _reference_roi_metric(roi_case_metrics_path)

    sample_indices = population["sample_index"].to_numpy(dtype=np.int64)
    for output_name, source_column, frequency_ghz in PROXY_FEATURES:
        population[output_name] = _feature_at_frequency(
            per_frequency,
            sample_indices,
            source_column,
            frequency_ghz,
        ).to_numpy(dtype=np.float64)
    feature_names = ["roi_metric_linear", *(name for name, _, _ in PROXY_FEATURES)]
    values = population[feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("model feature population contains non-finite values")
    for name in feature_names:
        population[f"{name}_rank"] = population[name].rank(
            method="average",
            pct=True,
        )
    return population


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _ridge_predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coefficients


def _select_ridge_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
    losses: dict[str, float] = {}
    for alpha in RIDGE_ALPHAS:
        predictions = np.empty(len(y), dtype=np.float64)
        for omitted in range(len(y)):
            keep = np.arange(len(y)) != omitted
            coefficients = _ridge_fit(x[keep], y[keep], alpha)
            predictions[omitted] = _ridge_predict(
                x[omitted : omitted + 1],
                coefficients,
            )[0]
        losses[format(alpha, "g")] = float(np.mean((predictions - y) ** 2))
    selected = min(RIDGE_ALPHAS, key=lambda alpha: (losses[format(alpha, "g")], alpha))
    return float(selected), losses


def _bootstrap_ridge_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_population: np.ndarray,
    *,
    alpha: float,
    bootstraps: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    predictions = np.empty((bootstraps, len(x_population)), dtype=np.float64)
    for bootstrap in range(bootstraps):
        selected = rng.integers(0, len(y_train), size=len(y_train))
        coefficients = _ridge_fit(
            x_train[selected],
            y_train[selected],
            alpha,
        )
        predictions[bootstrap] = _ridge_predict(x_population, coefficients)
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=1)


def _attach_predictions(
    population: pd.DataFrame,
    labeled_table_path: Path,
    *,
    bootstraps: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labeled = pd.read_csv(labeled_table_path).drop_duplicates("link_case_name")
    required = {
        "sample_index",
        "s21_power_september",
        "ebn0_ber_v2_at_target_db",
    }
    missing = sorted(required.difference(labeled.columns))
    if missing:
        raise ValueError(f"labeled proxy table lacks columns: {missing}")
    if len(labeled) != 13:
        raise ValueError(f"expected 13 proxy/link labels, got {len(labeled)}")

    feature_names = ["roi_metric_linear", *(name for name, _, _ in PROXY_FEATURES)]
    rank_names = [f"{name}_rank" for name in feature_names]
    model = population.merge(
        labeled[
            [
                "sample_index",
                "s21_power_september",
                "ebn0_ber_v2_at_target_db",
            ]
        ],
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    is_labeled = model["s21_power_september"].notna()
    if int(is_labeled.sum()) != 13:
        raise ValueError("not every link label maps to the model population")
    x_all = model[rank_names].to_numpy(dtype=np.float64)
    x_train_raw = x_all[is_labeled]
    x_mean = x_train_raw.mean(axis=0)
    x_scale = x_train_raw.std(axis=0, ddof=1)
    x_scale[x_scale == 0.0] = 1.0
    x_all = (x_all - x_mean) / x_scale
    x_train = x_all[is_labeled]

    targets = {
        "log10_s21_power": np.log10(
            model.loc[is_labeled, "s21_power_september"].to_numpy(dtype=np.float64)
        ),
        "negative_ebn0_db": -model.loc[
            is_labeled,
            "ebn0_ber_v2_at_target_db",
        ].to_numpy(dtype=np.float64),
    }
    diagnostics: dict[str, Any] = {
        "training_case_count": int(is_labeled.sum()),
        "feature_names": feature_names,
        "bootstraps": int(bootstraps),
        "random_seed": int(random_seed),
        "targets": {},
    }
    for offset, (target_name, y_train) in enumerate(targets.items()):
        alpha, loo_losses = _select_ridge_alpha(x_train, y_train)
        mean, std = _bootstrap_ridge_predictions(
            x_train,
            y_train,
            x_all,
            alpha=alpha,
            bootstraps=bootstraps,
            random_seed=random_seed + offset,
        )
        model[f"predicted_{target_name}"] = mean
        model[f"uncertainty_{target_name}"] = std
        diagnostics["targets"][target_name] = {
            "selected_alpha": alpha,
            "loo_mse_by_alpha": loo_losses,
            "training_mean": float(y_train.mean()),
            "training_std": float(y_train.std(ddof=1)),
        }

    model["predicted_s21_power"] = np.power(
        10.0,
        model["predicted_log10_s21_power"],
    )
    model["predicted_ebn0_db"] = -model["predicted_negative_ebn0_db"]
    s21_training = diagnostics["targets"]["log10_s21_power"]
    ebn0_training = diagnostics["targets"]["negative_ebn0_db"]
    model["normalized_uncertainty"] = 0.5 * (
        model["uncertainty_log10_s21_power"] / s21_training["training_std"]
        + model["uncertainty_negative_ebn0_db"] / ebn0_training["training_std"]
    )
    model["optimistic_s21"] = (
        model["predicted_log10_s21_power"] + 0.25 * model["uncertainty_log10_s21_power"]
    )
    model["optimistic_negative_ebn0"] = (
        model["predicted_negative_ebn0_db"]
        + 0.25 * model["uncertainty_negative_ebn0_db"]
    )
    model["optimistic_s21_rank"] = model["optimistic_s21"].rank(
        method="average",
        pct=True,
    )
    model["optimistic_negative_ebn0_rank"] = model["optimistic_negative_ebn0"].rank(
        method="average", pct=True
    )
    model["joint_optimistic_score"] = np.minimum(
        model["optimistic_s21_rank"],
        model["optimistic_negative_ebn0_rank"],
    )
    return model, diagnostics


def _load_parameters(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        manifest_path = Path(str(row.case_directory)) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        parameters = manifest.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"manifest lacks parameters: {manifest_path}")
        values = {
            name: float(parameters[name]) for name in antenna_sampler.PARAMETER_REGISTRY
        }
        rows.append({"sample_index": int(row.sample_index), **values})
    return pd.DataFrame(rows)


def _nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    values = np.asarray(objectives, dtype=np.float64)
    result = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(values <= point, axis=1) & np.any(
            values < point,
            axis=1,
        )
        if np.any(dominates):
            result[index] = False
    return result


def _selection_space(frame: pd.DataFrame) -> np.ndarray:
    proxy_names = [
        f"{name}_rank"
        for name in ["roi_metric_linear", *(name for name, _, _ in PROXY_FEATURES)]
    ]
    space = krvea_data.authoritative_input_space()
    geometry = space.normalize(frame.loc[:, space.names].to_numpy(dtype=np.float64))
    objective = np.column_stack(
        [
            frame["worst_s11_linear_amplitude"].rank(pct=True),
            (1.0 - frame["rad_eff_september_linear"]).rank(pct=True),
            frame["normalized_substrate_area"].rank(pct=True),
        ]
    )
    proxy = frame[proxy_names].to_numpy(dtype=np.float64)
    blocks = (
        0.60 * proxy / math.sqrt(proxy.shape[1]),
        0.25 * geometry / math.sqrt(geometry.shape[1]),
        0.15 * objective / math.sqrt(objective.shape[1]),
    )
    return np.concatenate(blocks, axis=1)


def _minimum_distances(
    points: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    if len(anchors) == 0:
        return np.ones(len(points), dtype=np.float64)
    differences = points[:, None, :] - anchors[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=2)).min(axis=1)


def _percentile(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(np.asarray(values, dtype=np.float64), method="average") / len(
        values
    )


def _greedy_select(
    candidate_indices: Sequence[int],
    count: int,
    *,
    base_values: np.ndarray,
    selection_space: np.ndarray,
    anchor_indices: Sequence[int],
    diversity_weight: float,
) -> list[int]:
    remaining = sorted(int(index) for index in candidate_indices)
    selected: list[int] = []
    anchors = [int(index) for index in anchor_indices]
    if len(remaining) < count:
        raise ValueError(f"only {len(remaining)} candidates for requested {count}")
    while len(selected) < count:
        indices = np.asarray(remaining, dtype=np.int64)
        anchor_rows = np.asarray(anchors + selected, dtype=np.int64)
        distances = _minimum_distances(
            selection_space[indices],
            selection_space[anchor_rows],
        )
        diversity = _percentile(distances)
        base = _percentile(base_values[indices])
        acquisition = (1.0 - diversity_weight) * base + diversity_weight * diversity
        best_position = max(
            range(len(indices)),
            key=lambda position: (
                float(acquisition[position]),
                float(base[position]),
                float(diversity[position]),
                -int(indices[position]),
            ),
        )
        chosen = int(indices[best_position])
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _known_identities(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    required = {"source", "source_case_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"previous propagation worklist lacks columns: {missing}")
    identities = {
        _identity(source, case_id)
        for source, case_id in zip(
            frame["source"],
            frame["source_case_id"],
            strict=True,
        )
    }
    if len(identities) != 13:
        raise ValueError(
            f"expected 13 prior propagated geometries, got {len(identities)}"
        )
    return identities


def _select_batch(
    population: pd.DataFrame,
    eligible: pd.DataFrame,
    parameters: pd.DataFrame,
    known_identities: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    archive = eligible.merge(
        population.drop(
            columns=[
                "source",
                "case_id",
                "case_directory",
                "worst_s11_linear_amplitude",
                "mean_total_efficiency_linear",
                "rad_eff_september_linear",
                "normalized_substrate_area",
                "return_loss_db",
            ],
            errors="ignore",
        ),
        on=["sample_index", "identity"],
        how="left",
        validate="one_to_one",
    ).merge(
        parameters,
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    archive.reset_index(drop=True, inplace=True)
    if archive["predicted_s21_power"].isna().any():
        raise ValueError("one or more eligible candidates lacks a model prediction")
    archive["previously_propagated"] = archive["identity"].isin(known_identities)

    objectives = np.column_stack(
        [
            archive["worst_s11_linear_amplitude"],
            1.0 - archive["rad_eff_september_linear"],
            archive["normalized_substrate_area"],
        ]
    )
    archive["three_objective_nondominated"] = _nondominated_mask(objectives)
    points = _selection_space(archive)
    known_indices = archive.index[archive["previously_propagated"]].tolist()
    candidate_indices = archive.index[~archive["previously_propagated"]].tolist()
    exploitation_pool = archive.index[
        ~archive["previously_propagated"] & archive["three_objective_nondominated"]
    ].tolist()
    if len(exploitation_pool) < EXPLOITATION_COUNT:
        raise ValueError(
            "fewer than four untested three-objective non-dominated candidates"
        )

    exploitation = _greedy_select(
        exploitation_pool,
        EXPLOITATION_COUNT,
        base_values=archive["joint_optimistic_score"].to_numpy(dtype=np.float64),
        selection_space=points,
        anchor_indices=known_indices,
        diversity_weight=0.20,
    )
    calibration_pool = [
        index for index in candidate_indices if index not in exploitation
    ]
    calibration = _greedy_select(
        calibration_pool,
        CALIBRATION_COUNT,
        base_values=archive["normalized_uncertainty"].to_numpy(dtype=np.float64),
        selection_space=points,
        anchor_indices=[*known_indices, *exploitation],
        diversity_weight=0.45,
    )

    previous_points = points[np.asarray(known_indices, dtype=np.int64)]
    archive["distance_to_previous_labels"] = _minimum_distances(
        points,
        previous_points,
    )
    selected_rows: list[pd.Series] = []
    for role, indices in (("calibration", calibration), ("exploitation", exploitation)):
        for role_rank, index in enumerate(indices, start=1):
            row = archive.loc[index].copy()
            row["acquisition_type"] = role
            row["acquisition_rank_within_type"] = role_rank
            row["selection_reason"] = (
                "bootstrap uncertainty plus distance from prior propagation labels"
                if role == "calibration"
                else (
                    "joint optimistic S21/EbN0 prediction on the retained "
                    "three-objective Pareto set"
                )
            )
            selected_rows.append(row)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    diagnostics = {
        "eligible_return_loss_ge_7db": int(len(archive)),
        "previously_propagated_excluded": int(archive["previously_propagated"].sum()),
        "untested_candidate_count": int(len(candidate_indices)),
        "three_objective_nondominated_count": int(
            archive["three_objective_nondominated"].sum()
        ),
        "untested_three_objective_nondominated_count": int(len(exploitation_pool)),
        "calibration_count": len(calibration),
        "exploitation_count": len(exploitation),
    }
    return selected, diagnostics


def _build_worklist_rows(selected: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counters = {"calibration": 0, "exploitation": 0}
    abbreviations = {"calibration": "cal", "exploitation": "exp"}
    for source in selected.itertuples(index=False):
        acquisition_type = str(source.acquisition_type)
        counters[acquisition_type] += 1
        role_rank = counters[acquisition_type]
        row: dict[str, str] = {
            "sample_id": (f"al01_{abbreviations[acquisition_type]}_{role_rank:02d}"),
            "simulation_mode": SIMULATION_MODE,
            "selection_rank": str(int(source.selection_rank)),
            "acquisition_type": acquisition_type,
            "acquisition_rank_within_type": str(role_rank),
            "selection_reason": str(source.selection_reason),
            "source": str(source.source),
            "source_case_id": str(source.case_id),
            "source_ffs_sha256": str(source.ffs_sha256),
            "geometry_valid": "True",
            "geometry_error": "",
            "final_conductor_components": "1",
            "return_loss_db": format(float(source.return_loss_db), ".17g"),
            "predicted_s21_power": format(
                float(source.predicted_s21_power),
                ".17g",
            ),
            "predicted_ebn0_db": format(float(source.predicted_ebn0_db), ".17g"),
            "uncertainty_log10_s21_power": format(
                float(source.uncertainty_log10_s21_power),
                ".17g",
            ),
            "uncertainty_negative_ebn0_db": format(
                float(source.uncertainty_negative_ebn0_db),
                ".17g",
            ),
            "roi_metric_linear": format(float(source.roi_metric_linear), ".17g"),
            "three_objective_nondominated": str(
                bool(source.three_objective_nondominated)
            ),
        }
        for name in antenna_sampler.PARAMETER_REGISTRY:
            value = float(getattr(source, name))
            if not math.isfinite(value):
                raise ValueError(f"selected parameter {name} is not finite")
            row[name] = format(value, ".17g")
        antenna_sampler.parameters_from_csv_row(row)
        rows.append(row)
    if len(rows) != CALIBRATION_COUNT + EXPLOITATION_COUNT:
        raise AssertionError("active-learning worklist must contain twelve rows")
    return rows


def _write_worklist(rows: Sequence[Mapping[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare_batch(
    frequency_table_path: str | Path = DEFAULT_FREQUENCY_TABLE,
    roi_result_path: str | Path = DEFAULT_ROI_RESULT,
    roi_case_metrics_path: str | Path = DEFAULT_ROI_CASE_METRICS,
    labeled_table_path: str | Path = DEFAULT_LABELED_TABLE,
    previous_worklist_path: str | Path = DEFAULT_PREVIOUS_WORKLIST,
    output_csv_path: str | Path = DEFAULT_OUTPUT_CSV,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    workers: int = DEFAULT_WORKERS,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = {
        "frequency_table": Path(frequency_table_path).expanduser().resolve(),
        "roi_result": Path(roi_result_path).expanduser().resolve(),
        "roi_case_metrics": Path(roi_case_metrics_path).expanduser().resolve(),
        "labeled_table": Path(labeled_table_path).expanduser().resolve(),
        "previous_worklist": Path(previous_worklist_path).expanduser().resolve(),
        "worklist": Path(output_csv_path).expanduser().resolve(),
        "output_directory": Path(output_directory).expanduser().resolve(),
    }
    feature_cache = paths["output_directory"] / DEFAULT_FEATURE_CACHE.name
    predictions_path = paths["output_directory"] / DEFAULT_PREDICTIONS.name
    audit_path = paths["output_directory"] / DEFAULT_AUDIT.name
    manifest_path = paths["output_directory"] / DEFAULT_MANIFEST.name
    existing_outputs = [
        path
        for path in (paths["worklist"], predictions_path, audit_path, manifest_path)
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            f"batch output exists; pass --overwrite to replace: {existing_outputs[0]}"
        )

    contract = _load_roi_contract(paths["roi_result"])
    per_frequency, eligible = _load_candidate_metadata(paths["frequency_table"])
    roi_features = _load_or_create_roi_cache(
        eligible,
        contract,
        feature_cache,
        workers=workers,
        overwrite=overwrite,
    )
    population = _build_model_feature_table(
        per_frequency,
        eligible,
        roi_features,
        paths["roi_case_metrics"],
    )
    population, model_diagnostics = _attach_predictions(
        population,
        paths["labeled_table"],
        bootstraps=bootstraps,
        random_seed=random_seed,
    )
    parameters = _load_parameters(eligible)
    selected, selection_diagnostics = _select_batch(
        population,
        eligible,
        parameters,
        _known_identities(paths["previous_worklist"]),
    )

    for source in selected.itertuples(index=False):
        actual_hash = _sha256(source.ffs_path)
        if actual_hash.casefold() != str(source.ffs_sha256).casefold():
            raise ValueError(f"selected FFS hash mismatch: {source.ffs_path}")

    rows = _build_worklist_rows(selected)
    _write_worklist(rows, paths["worklist"])
    _write_csv_atomic(population, predictions_path)
    _write_csv_atomic(selected, audit_path)
    payload: dict[str, Any] = {
        "schema": "msabp.propagation_active_learning_selection.v1",
        "batch": 1,
        "simulation_mode": SIMULATION_MODE,
        "selection_policy": {
            "return_loss_min_db": RETURN_LOSS_MIN_DB,
            "calibration": (
                "8 points: bootstrap uncertainty plus feature-space coverage"
            ),
            "exploitation": (
                "4 points: joint optimistic prediction, restricted to the "
                "S11/RadEff/area non-dominated set"
            ),
            "cap_gain_used_for_eligibility_or_pareto": False,
            "diversity_space_weights": {
                "farfield_proxy": 0.60,
                "geometry_11d": 0.25,
                "retained_three_objectives": 0.15,
            },
            "optimism_beta": 0.25,
        },
        "roi_contract": contract.__dict__,
        "model": model_diagnostics,
        "population": selection_diagnostics,
        "selected": [
            {
                "sample_id": row["sample_id"],
                "acquisition_type": row["acquisition_type"],
                "source": row["source"],
                "source_case_id": row["source_case_id"],
                "predicted_s21_power": float(row["predicted_s21_power"]),
                "predicted_ebn0_db": float(row["predicted_ebn0_db"]),
            }
            for row in rows
        ],
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if name
            in {
                "frequency_table",
                "roi_result",
                "roi_case_metrics",
                "labeled_table",
                "previous_worklist",
            }
        },
        "outputs": {
            "worklist": str(paths["worklist"]),
            "worklist_sha256": _sha256(paths["worklist"]),
            "feature_cache": str(feature_cache),
            "feature_cache_sha256": _sha256(feature_cache),
            "candidate_predictions_frozen": str(predictions_path),
            "candidate_predictions_frozen_sha256": _sha256(predictions_path),
            "selection_audit": str(audit_path),
        },
    }
    _write_json_atomic(payload, manifest_path)
    print(
        "[ActiveLearning-01] selected "
        f"{CALIBRATION_COUNT} calibration + {EXPLOITATION_COUNT} exploitation"
    )
    display_columns = [
        "selection_rank",
        "acquisition_type",
        "source",
        "case_id",
        "return_loss_db",
        "roi_metric_linear",
        "rad_eff_september_linear",
        "predicted_s21_power",
        "predicted_ebn0_db",
        "normalized_uncertainty",
        "three_objective_nondominated",
    ]
    print(selected[display_columns].to_string(index=False))
    print(f"[ActiveLearning-01] worklist -> {paths['worklist']}")
    print(f"[ActiveLearning-01] audit -> {audit_path}")
    print(f"[ActiveLearning-01] manifest -> {manifest_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-table", type=Path, default=DEFAULT_FREQUENCY_TABLE)
    parser.add_argument("--roi-result", type=Path, default=DEFAULT_ROI_RESULT)
    parser.add_argument(
        "--roi-case-metrics",
        type=Path,
        default=DEFAULT_ROI_CASE_METRICS,
    )
    parser.add_argument("--labeled-table", type=Path, default=DEFAULT_LABELED_TABLE)
    parser.add_argument(
        "--previous-worklist",
        type=Path,
        default=DEFAULT_PREVIOUS_WORKLIST,
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_batch(
        args.frequency_table,
        args.roi_result,
        args.roi_case_metrics,
        args.labeled_table,
        args.previous_worklist,
        args.output_csv,
        args.output_directory,
        workers=args.workers,
        bootstraps=args.bootstraps,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
