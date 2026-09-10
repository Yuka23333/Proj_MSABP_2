"""Phase-2 data contract for the 11-D MSABP K-RVEA campaign.

All objectives use minimization semantics in this order::

    [worst |S11|, -ROI theta-polarized radiation gain dBi,
     normalized substrate area]

The frozen propagation proxy is evaluated at 3.6 GHz over theta=55..85 deg
and phi=90+/-50 deg.  The spatial average is performed in linear power with
solid-angle theta weights, then converted to dBi.  Radiation gain includes
radiation efficiency but excludes mismatch loss; S11 therefore remains an
independent objective and Rad_Eff is not fitted a second time.

This module deliberately reuses only the historical 11-D input and exact-area
contract.  It does not reinterpret or mutate the first-phase four-objective
observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import krvea_data as phase1_data
from . import qlogehvi


SCHEMA_VERSION = 1
ROI_METRIC_VERSION = 1
DEFAULT_BAND_GHZ = phase1_data.DEFAULT_BAND_GHZ
FARFIELD_FILENAME = phase1_data.FARFIELD_FILENAME
ROI_FREQUENCY_GHZ = 3.6
ROI_THETA_BOUNDS_DEG = (55.0, 85.0)
ROI_PHI_CENTER_DEG = 90.0
ROI_PHI_HALF_WIDTH_DEG = 50.0
ROI_CACHE_FILENAME = ".phase2_roi_radiation_gain_cache.json"
GRID_TOLERANCE = 1.0e-10

ACTIVE_PARAMETER_NAMES = phase1_data.ACTIVE_PARAMETER_NAMES
ACTIVE_PARAMETER_LOWER = phase1_data.ACTIVE_PARAMETER_LOWER
ACTIVE_PARAMETER_UPPER = phase1_data.ACTIVE_PARAMETER_UPPER
ACTIVE_PARAMETER_NOMINAL = phase1_data.ACTIVE_PARAMETER_NOMINAL
FIXED_PARAMETER_VALUES = phase1_data.FIXED_PARAMETER_VALUES
InputSpace = phase1_data.InputSpace

WORST_S11_COLUMN = qlogehvi.WORST_S11_COLUMN
AREA_COLUMN = qlogehvi.AREA_COLUMN
NORMALIZED_AREA_COLUMN = phase1_data.NORMALIZED_AREA_COLUMN
ROI_GAIN_LINEAR_COLUMN = "roi_theta_radiation_gain_linear"
ROI_GAIN_DBI_COLUMN = "roi_theta_radiation_gain_dbi"
ROI_GAIN_LOSS_DBI_COLUMN = "negative_roi_theta_radiation_gain_dbi"

OBJECTIVE_NAMES: tuple[str, ...] = (
    WORST_S11_COLUMN,
    ROI_GAIN_LOSS_DBI_COLUMN,
    NORMALIZED_AREA_COLUMN,
)

PENALTY_ROI_GAIN_LOSS_DBI = 100.0
PENALTY_NORMALIZED_AREA = 2.0


class IncompleteObservationError(ValueError):
    """A manifest exists, but its case has not produced a terminal datum."""


@dataclass(frozen=True)
class Dataset:
    """Deduplicated arrays ready for the three-objective Phase-2 engine."""

    x_raw: np.ndarray
    x_unit: np.ndarray
    objectives: np.ndarray
    metadata: pd.DataFrame
    input_space: InputSpace
    objective_names: tuple[str, ...] = OBJECTIVE_NAMES

    @property
    def exact_area(self) -> Callable[[Sequence[float]], float]:
        return self.input_space.exact_normalized_area


authoritative_input_space = phase1_data.authoritative_input_space
substrate_dimensions = phase1_data.substrate_dimensions
reference_substrate_area_mm2 = phase1_data.reference_substrate_area_mm2
normalized_substrate_area = phase1_data.normalized_substrate_area


def _artifact_path(
    manifest: Mapping[str, Any],
    case_directory: Path,
    artifact_name: str,
    default_filename: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        record = artifacts.get(artifact_name)
        if isinstance(record, Mapping) and record.get("path"):
            return case_directory / str(record["path"])
    return case_directory / default_filename


def _farfield_identity(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    record = artifacts.get("farfield_source") if isinstance(artifacts, Mapping) else None
    declared_hash = record.get("sha256") if isinstance(record, Mapping) else None
    stat = path.stat()
    return {
        "sha256": str(declared_hash) if declared_hash else None,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns) if not declared_hash else None,
    }


def _cache_path(ffs_path: Path, cache_directory: str | Path | None) -> Path:
    if cache_directory is None:
        return ffs_path.parent / ROI_CACHE_FILENAME
    cache_root = Path(cache_directory)
    identity = hashlib.sha256(str(ffs_path.resolve()).casefold().encode("utf-8")).hexdigest()
    return cache_root / f"{identity}.json"


def _read_roi_cache(
    cache_path: Path,
    contract: Mapping[str, Any],
) -> tuple[float, float] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("contract") != dict(contract):
        return None
    try:
        linear = float(payload[ROI_GAIN_LINEAR_COLUMN])
        dbi = float(payload[ROI_GAIN_DBI_COLUMN])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(linear) and linear > 0.0 and math.isfinite(dbi)):
        return None
    return linear, dbi


def _write_roi_cache(
    cache_path: Path,
    contract: Mapping[str, Any],
    linear: float,
    dbi: float,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": dict(contract),
        ROI_GAIN_LINEAR_COLUMN: linear,
        ROI_GAIN_DBI_COLUMN: dbi,
        "storage_note": "spatial average in linear power; objective is negative dBi",
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, cache_path)
    except OSError:
        return
    finally:
        try:
            if "temporary" in locals() and temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def _single_grid_index(grid: np.ndarray, value: float, label: str) -> int:
    matches = np.flatnonzero(np.isclose(grid, value, atol=GRID_TOLERANCE, rtol=0.0))
    if len(matches) != 1:
        raise ValueError(f"{label}={value:g} must occur exactly once on the FFS grid")
    return int(matches[0])


def _uncached_roi_radiation_gain_linear(
    ffs_path: Path,
    *,
    frequency_ghz: float,
    theta_bounds_deg: tuple[float, float],
    phi_center_deg: float,
    phi_half_width_deg: float,
) -> float:
    from scripts.postprocessing.cap_gain import parse_ffs
    from scripts.postprocessing.prepare_link_ffs_tensor import _pattern_power
    from scripts.postprocessing.search_link_spherical_roi import (
        spherical_theta_cell_weights,
        symmetric_phi_mask,
    )

    ffs = parse_ffs(ffs_path)
    frequency = np.asarray(ffs["freq"], dtype=np.float64) / 1.0e9
    theta = np.asarray(ffs["theta_deg"], dtype=np.float64)
    phi_full = np.asarray(ffs["phi_deg"], dtype=np.float64)
    e_theta_full = np.asarray(ffs["E_theta"], dtype=np.complex128)
    e_phi_full = np.asarray(ffs["E_phi"], dtype=np.complex128)
    frequency_index = _single_grid_index(frequency, frequency_ghz, "frequency_ghz")

    pattern_power = _pattern_power(e_theta_full, e_phi_full, theta, phi_full)
    p_rad = float(np.asarray(ffs["p_rad"], dtype=np.float64)[frequency_index])
    p_acc = float(np.asarray(ffs["p_acc"], dtype=np.float64)[frequency_index])
    if not (math.isfinite(p_rad) and math.isfinite(p_acc) and p_rad > 0.0 and p_acc > 0.0):
        raise ValueError("FFS radiated and accepted powers must be finite and positive")
    radiation_efficiency = p_rad / p_acc
    if radiation_efficiency > 1.0 + 1.0e-6:
        raise ValueError("FFS radiation efficiency exceeds the passive bound")

    phi = phi_full
    e_theta = e_theta_full
    if (
        len(phi_full) >= 2
        and np.isclose(phi_full[0], 0.0, atol=GRID_TOLERANCE, rtol=0.0)
        and np.isclose(phi_full[-1], 360.0, atol=GRID_TOLERANCE, rtol=0.0)
    ):
        phi = phi_full[:-1]
        e_theta = e_theta_full[:, :-1, :]

    theta_low, theta_high = theta_bounds_deg
    theta_mask = (theta >= theta_low - GRID_TOLERANCE) & (
        theta <= theta_high + GRID_TOLERANCE
    )
    if not np.any(theta_mask):
        raise ValueError("the frozen theta ROI contains no FFS grid points")
    if not (
        np.isclose(theta[theta_mask][0], theta_low, atol=GRID_TOLERANCE, rtol=0.0)
        and np.isclose(theta[theta_mask][-1], theta_high, atol=GRID_TOLERANCE, rtol=0.0)
    ):
        raise ValueError("the frozen theta ROI boundaries are absent from the FFS grid")
    phi_mask = symmetric_phi_mask(phi, phi_center_deg, phi_half_width_deg)
    if not np.any(phi_mask):
        raise ValueError("the frozen phi ROI contains no FFS grid points")

    theta_weights = spherical_theta_cell_weights(theta)[theta_mask]
    directivity_theta = (
        np.abs(e_theta[frequency_index, phi_mask][:, theta_mask]) ** 2
        * (4.0 * math.pi / float(pattern_power[frequency_index]))
    )
    radiation_gain_theta = directivity_theta * radiation_efficiency
    theta_average = np.average(radiation_gain_theta, axis=1, weights=theta_weights)
    scalar = float(np.mean(theta_average))
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("frozen-ROI theta radiation gain must be finite and positive")
    return scalar


def roi_radiation_gain_scalar(
    ffs_path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    frequency_ghz: float = ROI_FREQUENCY_GHZ,
    theta_bounds_deg: tuple[float, float] = ROI_THETA_BOUNDS_DEG,
    phi_center_deg: float = ROI_PHI_CENTER_DEG,
    phi_half_width_deg: float = ROI_PHI_HALF_WIDTH_DEG,
    cache_directory: str | Path | None = None,
) -> tuple[float, float, bool]:
    """Return frozen-ROI theta radiation gain as (linear, dBi, cache_hit)."""

    path = Path(ffs_path).resolve()
    manifest = {} if manifest is None else manifest
    contract = {
        "metric_version": ROI_METRIC_VERSION,
        "ffs": _farfield_identity(path, manifest),
        "frequency_ghz": float(frequency_ghz),
        "theta_bounds_deg": [float(value) for value in theta_bounds_deg],
        "phi_center_deg": float(phi_center_deg),
        "phi_half_width_deg": float(phi_half_width_deg),
        "component": "E_theta",
        "gain_type": "radiation_gain",
        "spatial_average": "linear_power_solid_angle_weighted",
    }
    cache_path = _cache_path(path, cache_directory)
    cached = _read_roi_cache(cache_path, contract)
    if cached is not None:
        return cached[0], cached[1], True
    linear = _uncached_roi_radiation_gain_linear(
        path,
        frequency_ghz=frequency_ghz,
        theta_bounds_deg=theta_bounds_deg,
        phi_center_deg=phi_center_deg,
        phi_half_width_deg=phi_half_width_deg,
    )
    dbi = float(10.0 * math.log10(linear))
    _write_roi_cache(cache_path, contract, linear, dbi)
    return linear, dbi, False


def _worst_s11_linear(
    path: Path,
    band_ghz: tuple[float, float],
) -> float:
    frequency, values_dbi = qlogehvi.read_cst_curve(path)
    values_linear = np.power(10.0, values_dbi / 20.0)
    selected = qlogehvi._band_values_with_endpoints(
        frequency,
        values_linear,
        band_ghz,
    )
    if np.any(selected < 0.0) or np.any(selected > 1.0 + 1.0e-9):
        raise ValueError("S11 linear amplitude falls outside passive range [0, 1]")
    return float(np.max(np.clip(selected, 0.0, 1.0)))


def _validate_fixed_parameters(parameters: Mapping[str, Any], manifest_path: Path) -> None:
    missing = sorted(set(FIXED_PARAMETER_VALUES) - set(parameters))
    if missing:
        raise ValueError(f"manifest is missing fixed parameters {missing}: {manifest_path}")
    mismatches = []
    for name, expected in FIXED_PARAMETER_VALUES.items():
        actual = float(parameters[name])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            mismatches.append(f"{name}={actual:g} (expected {expected:g})")
    if mismatches:
        raise ValueError(
            f"manifest is not in the current fixed 11-D design space: {manifest_path}: "
            + ", ".join(mismatches)
        )


def parse_manifest(
    manifest_path: str | Path,
    *,
    source_root: str | Path,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
    cache_directory: str | Path | None = None,
    strict_fixed_parameters: bool = True,
) -> dict[str, Any]:
    """Reduce one completed or explicitly penalized case to Phase-2 objectives."""

    path = Path(manifest_path).resolve()
    case_directory = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    status = str(manifest.get("status", ""))
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"manifest has no parameter mapping: {path}")
    missing = set(ACTIVE_PARAMETER_NAMES) - set(parameters)
    if missing:
        raise ValueError(f"manifest is missing active parameters {sorted(missing)}")
    if strict_fixed_parameters:
        _validate_fixed_parameters(parameters, path)
    raw = {name: float(parameters[name]) for name in ACTIVE_PARAMETER_NAMES}
    width, height, area = substrate_dimensions(raw)

    penalty_path = case_directory / qlogehvi.OPTIMIZATION_PENALTY_FILENAME
    sidecar: Mapping[str, Any] | None = None
    if penalty_path.is_file():
        loaded = json.loads(penalty_path.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"invalid optimization penalty sidecar: {penalty_path}")
        sidecar = loaded
    is_penalty = status == "penalized" or sidecar is not None

    if is_penalty:
        objective_payload = (
            sidecar.get("optimization_objectives")
            if sidecar is not None
            else manifest.get("optimization_objectives")
        )
        if not isinstance(objective_payload, Mapping):
            raise ValueError(f"penalized manifest has no objective mapping: {path}")
        worst_s11 = float(objective_payload.get(WORST_S11_COLUMN, 1.0))
        normalized_area_objective = float(
            objective_payload.get(NORMALIZED_AREA_COLUMN, PENALTY_NORMALIZED_AREA)
        )
        roi_loss_dbi = float(
            objective_payload.get(
                ROI_GAIN_LOSS_DBI_COLUMN,
                PENALTY_ROI_GAIN_LOSS_DBI,
            )
        )
        roi_dbi = -roi_loss_dbi
        roi_linear = float(np.power(10.0, roi_dbi / 10.0))
        cache_hit = False
    elif status == "completed":
        s11_path = _artifact_path(manifest, case_directory, "s11", qlogehvi.S11_FILENAME)
        ffs_path = _artifact_path(manifest, case_directory, "farfield_source", FARFIELD_FILENAME)
        missing_artifacts = [
            artifact.name for artifact in (s11_path, ffs_path) if not artifact.is_file()
        ]
        if missing_artifacts:
            raise IncompleteObservationError(
                f"completed manifest is missing Phase-2 artifact(s) {missing_artifacts}: {path}"
            )
        worst_s11 = _worst_s11_linear(s11_path, band_ghz)
        roi_linear, roi_dbi, cache_hit = roi_radiation_gain_scalar(
            ffs_path,
            manifest=manifest,
            cache_directory=cache_directory,
        )
        roi_loss_dbi = -roi_dbi
        normalized_area_objective = area / reference_substrate_area_mm2()
    else:
        raise IncompleteObservationError(
            f"manifest status {status!r} is not a terminal observation: {path}"
        )

    if not 0.0 <= worst_s11 <= 1.0:
        raise ValueError(f"worst S11 must lie in [0,1]: {path}")
    if not math.isfinite(normalized_area_objective) or normalized_area_objective <= 0.0:
        raise ValueError(f"normalized area objective must be positive: {path}")
    if not (math.isfinite(roi_linear) and roi_linear > 0.0 and math.isfinite(roi_dbi)):
        raise ValueError(f"ROI radiation gain must be finite and positive: {path}")

    return {
        "source": Path(source_root).resolve().name,
        "source_root": str(Path(source_root).resolve()),
        "case_id": str(manifest.get("case_id", case_directory.name)),
        "case_directory": str(case_directory),
        "status": "penalized" if is_penalty else status,
        "is_penalty": is_penalty,
        **raw,
        "substrate_width_mm": width,
        "substrate_height_mm": height,
        AREA_COLUMN: area,
        NORMALIZED_AREA_COLUMN: normalized_area_objective,
        WORST_S11_COLUMN: worst_s11,
        ROI_GAIN_LINEAR_COLUMN: roi_linear,
        ROI_GAIN_DBI_COLUMN: roi_dbi,
        ROI_GAIN_LOSS_DBI_COLUMN: roi_loss_dbi,
        "roi_gain_cache_hit": cache_hit,
    }


def _manifest_paths(source_root: Path) -> list[Path]:
    root = source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"result source does not exist: {root}")
    direct = root / qlogehvi.MANIFEST_FILENAME
    if direct.is_file():
        return [direct]
    return sorted(root.rglob(qlogehvi.MANIFEST_FILENAME))


def collect_observations(
    source_roots: Sequence[str | Path],
    *,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
    cache_directory: str | Path | None = None,
    skipped_incomplete: list[str] | None = None,
    strict_fixed_parameters: bool = True,
) -> pd.DataFrame:
    """Collect terminal Phase-2 observations and skip only incomplete cases."""

    records: list[dict[str, Any]] = []
    seen_manifests: set[Path] = set()
    for source_value in source_roots:
        source = Path(source_value).resolve()
        for manifest_path in _manifest_paths(source):
            resolved = manifest_path.resolve()
            if resolved in seen_manifests:
                continue
            seen_manifests.add(resolved)
            try:
                records.append(
                    parse_manifest(
                        resolved,
                        source_root=source,
                        band_ghz=band_ghz,
                        cache_directory=cache_directory,
                        strict_fixed_parameters=strict_fixed_parameters,
                    )
                )
            except IncompleteObservationError as exc:
                if skipped_incomplete is not None:
                    skipped_incomplete.append(str(exc))
    if not records:
        raise ValueError("no completed or penalized Phase-2 K-RVEA observations were found")
    frame = pd.DataFrame.from_records(records)
    frame.sort_values(["source_root", "case_id"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def build_dataset(
    observations: pd.DataFrame,
    *,
    input_space: InputSpace | None = None,
) -> Dataset:
    """Deduplicate designs and expose raw/unit inputs plus three min objectives."""

    space = authoritative_input_space() if input_space is None else input_space
    missing = set(space.names + OBJECTIVE_NAMES + ("is_penalty",)) - set(observations.columns)
    if missing:
        raise ValueError(f"observation table is missing columns {sorted(missing)}")
    raw = observations.loc[:, space.names].to_numpy(dtype=np.float64)
    unit = space.normalize(raw)
    tolerance = 1.0e-9
    if np.any(unit < -tolerance) or np.any(unit > 1.0 + tolerance):
        raise ValueError("one or more observations fall outside the authoritative 11-D bounds")
    working = observations.copy()
    working["_unit_key"] = [
        tuple(np.round(row, 12)) for row in np.clip(unit, 0.0, 1.0)
    ]

    x_raw_rows: list[np.ndarray] = []
    x_unit_rows: list[np.ndarray] = []
    objective_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    for key, group in working.groupby("_unit_key", sort=False):
        completed = group.loc[~group["is_penalty"].astype(bool)]
        selected = completed if not completed.empty else group
        representative = selected.iloc[0]
        raw_row = representative.loc[list(space.names)].to_numpy(dtype=np.float64)
        objective = np.asarray(
            [float(selected[name].mean()) for name in OBJECTIVE_NAMES],
            dtype=np.float64,
        )
        x_raw_rows.append(raw_row)
        x_unit_rows.append(np.asarray(key, dtype=np.float64))
        objective_rows.append(objective)
        metadata_rows.append(
            {
                "design_key": json.dumps(list(key), separators=(",", ":")),
                "case_directories": tuple(selected["case_directory"].astype(str)),
                "case_ids": tuple(selected["case_id"].astype(str)),
                "replicate_count": int(len(selected)),
                "discarded_penalty_count": int(len(group) - len(selected)),
                "has_completed_result": bool(not completed.empty),
                AREA_COLUMN: float(representative[AREA_COLUMN]),
                NORMALIZED_AREA_COLUMN: float(representative[NORMALIZED_AREA_COLUMN]),
                ROI_GAIN_DBI_COLUMN: float(-objective[1]),
                ROI_GAIN_LINEAR_COLUMN: float(np.power(10.0, -objective[1] / 10.0)),
            }
        )
    return Dataset(
        x_raw=np.asarray(x_raw_rows, dtype=np.float64),
        x_unit=np.asarray(x_unit_rows, dtype=np.float64),
        objectives=np.asarray(objective_rows, dtype=np.float64),
        metadata=pd.DataFrame.from_records(metadata_rows),
        input_space=space,
    )


def load_dataset(
    source_roots: Sequence[str | Path],
    *,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
    cache_directory: str | Path | None = None,
    strict_fixed_parameters: bool = True,
) -> Dataset:
    observations = collect_observations(
        source_roots,
        band_ghz=band_ghz,
        cache_directory=cache_directory,
        strict_fixed_parameters=strict_fixed_parameters,
    )
    return build_dataset(observations)


__all__ = [
    "ACTIVE_PARAMETER_LOWER",
    "ACTIVE_PARAMETER_NAMES",
    "ACTIVE_PARAMETER_NOMINAL",
    "ACTIVE_PARAMETER_UPPER",
    "AREA_COLUMN",
    "Dataset",
    "FIXED_PARAMETER_VALUES",
    "InputSpace",
    "NORMALIZED_AREA_COLUMN",
    "OBJECTIVE_NAMES",
    "PENALTY_NORMALIZED_AREA",
    "PENALTY_ROI_GAIN_LOSS_DBI",
    "ROI_FREQUENCY_GHZ",
    "ROI_GAIN_DBI_COLUMN",
    "ROI_GAIN_LINEAR_COLUMN",
    "ROI_GAIN_LOSS_DBI_COLUMN",
    "ROI_METRIC_VERSION",
    "ROI_PHI_CENTER_DEG",
    "ROI_PHI_HALF_WIDTH_DEG",
    "ROI_THETA_BOUNDS_DEG",
    "SCHEMA_VERSION",
    "WORST_S11_COLUMN",
    "authoritative_input_space",
    "build_dataset",
    "collect_observations",
    "load_dataset",
    "normalized_substrate_area",
    "parse_manifest",
    "reference_substrate_area_mm2",
    "roi_radiation_gain_scalar",
    "substrate_dimensions",
]
