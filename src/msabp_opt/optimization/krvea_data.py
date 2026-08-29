"""Data contracts and objective extraction for the 11-D MSABP K-RVEA run.

All four optimization objectives in this module use minimization semantics::

    [worst |S11|, 1 - mean(Tot_Eff), normalized substrate area,
     cap-averaged realized gain]

S11 and efficiency are stored in linear units.  Cap gain is averaged over
both angle and frequency in linear power, then stored as dBi in the objective
matrix to match the project's existing cap-gain tables.  Both representations
remain available in the observation table.

The active input order and bounds are deliberately tied to the authoritative
11-variable DoE builder.  The other twelve antenna parameters are checked
against their fixed values so data from older 23-variable campaigns cannot be
silently collapsed onto the same 11-dimensional design.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import qlogehvi


SCHEMA_VERSION = 1
DEFAULT_BAND_GHZ = (3.1, 4.8)
CAP_THETA_MAX_DEG = 15
CAP_CACHE_FILENAME = ".krvea_cap_gain_cache.json"
CAP_METRIC_VERSION = 1
FARFIELD_FILENAME = "Farfield Source [1].ffs"

# Registry order after filtering the authoritative 11-variable sampling plan.
ACTIVE_PARAMETER_NAMES: tuple[str, ...] = (
    "SLOT_MAIN_LENGTH",
    "PATCH_BRICK_1_SIDE_MARGIN",
    "PATCH_BRICK_1_TOP_MARGIN",
    "PATCH_BRICK_2_HEIGHT_MARGIN",
    "UPPER_CORNER_NOTCH_1_K1",
    "UPPER_CORNER_NOTCH_1_K2",
    "UPPER_CORNER_EAR_1_K1",
    "UPPER_CORNER_EAR_1_K2",
    "BRANCH_UP_1_K",
    "BRANCH_UP_1_K2",
    "BRANCH_UP_1_K3",
)

# Resolved from prepare_doe_11var_branch_up.build_sampling_config() at the
# campaign checkpoint. Keeping this contract local allows objective auditing
# in lightweight environments that do not install Shapely.
ACTIVE_PARAMETER_LOWER = np.asarray(
    [47.7, 5.4, 2.34, 13.5, 0.46818181818181814, 0.7, 0.2617647058823529, 0.0, 0.05, 0.05, 0.05],
    dtype=np.float64,
)
ACTIVE_PARAMETER_UPPER = np.asarray(
    [58.3, 6.6, 2.86, 16.5, 0.7681818181818181, 1.0, 0.5617647058823529, 0.3, 1.0, 1.0, 1.0], dtype=np.float64,
)
ACTIVE_PARAMETER_NOMINAL = np.asarray(
    [53.0, 6.0, 2.6, 15.0, 0.6181818181818182, 0.9333333333333333, 0.4117647058823529, 0.07142857142857142, 0.5, 0.5, 0.5], dtype=np.float64,
)
FIXED_PARAMETER_VALUES: dict[str, float] = {
    "SLOT_MAIN_HEIGHT": 2.0,
    "PATCH_BRICK_3_BOTTOM_MARGIN": 2.0,
    "PATCH_BRICK_4_MARGIN": 4.0,
    "LOWER_CORNER_NOTCH_1_K1": 0.7745454545454545,
    "LOWER_CORNER_NOTCH_1_K2": 0.7058823529411765,
    "LOWER_CORNER_EAR_1_K1": 0.2347417840375587,
    "LOWER_CORNER_EAR_1_K2": 0.6666666666666666,
    "LOWER_CORNER_EAR_2_K1": 0.24539877300613497,
    "LOWER_CORNER_EAR_2_K2": 0.25,
    "BRANCH_DOWN_1_K": 0.5,
    "BRANCH_DOWN_1_K2": 0.5,
    "BRANCH_DOWN_1_K3": 0.0,
}

WORST_S11_COLUMN = qlogehvi.WORST_S11_COLUMN
MEAN_TOT_EFF_COLUMN = qlogehvi.MEAN_TOT_EFF_COLUMN
TOT_EFF_LOSS_COLUMN = "one_minus_mean_total_efficiency_linear"
AREA_COLUMN = qlogehvi.AREA_COLUMN
NORMALIZED_AREA_COLUMN = "normalized_substrate_area"
CAP_GAIN_LINEAR_COLUMN = "cap_realized_gain_linear"
CAP_GAIN_DBI_COLUMN = "cap_realized_gain_dbi"

OBJECTIVE_NAMES: tuple[str, ...] = (
    WORST_S11_COLUMN,
    TOT_EFF_LOSS_COLUMN,
    NORMALIZED_AREA_COLUMN,
    CAP_GAIN_DBI_COLUMN,
)

# A terminal invalid design must be dominated in every expensive objective.
# 10 linear = 10 dBi, deliberately above any useful broadside-cap result.
PENALTY_CAP_GAIN_LINEAR = 10.0
PENALTY_NORMALIZED_AREA = 2.0


class IncompleteObservationError(ValueError):
    """A manifest exists, but its case has not produced a terminal datum."""


@dataclass(frozen=True)
class InputSpace:
    """The authoritative 11-dimensional raw and normalized design space."""

    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    nominal: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        nominal = np.asarray(self.nominal, dtype=np.float64)
        expected = (len(self.names),)
        if lower.shape != expected or upper.shape != expected or nominal.shape != expected:
            raise ValueError("input-space arrays do not match parameter names")
        if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise ValueError("input-space bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("each upper bound must exceed its lower bound")
        if np.any(nominal < lower) or np.any(nominal > upper):
            raise ValueError("nominal point falls outside the input space")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "nominal", nominal)

    def normalize(self, raw_values: np.ndarray | Sequence[float]) -> np.ndarray:
        raw = np.asarray(raw_values, dtype=np.float64)
        return (raw - self.lower) / (self.upper - self.lower)

    def denormalize(self, unit_values: np.ndarray | Sequence[float]) -> np.ndarray:
        unit = np.asarray(unit_values, dtype=np.float64)
        return self.lower + unit * (self.upper - self.lower)

    def values(self, raw_values: Sequence[float]) -> dict[str, float]:
        raw = np.asarray(raw_values, dtype=np.float64)
        if raw.shape != (len(self.names),):
            raise ValueError("raw parameter vector has the wrong shape")
        return dict(zip(self.names, raw.tolist()))

    def exact_normalized_area(self, raw_values: Sequence[float]) -> float:
        return normalized_substrate_area(self.values(raw_values))


@dataclass(frozen=True)
class Dataset:
    """Deduplicated arrays ready for a minimization-based K-RVEA engine."""

    x_raw: np.ndarray
    x_unit: np.ndarray
    objectives: np.ndarray
    metadata: pd.DataFrame
    input_space: InputSpace
    objective_names: tuple[str, ...] = OBJECTIVE_NAMES

    @property
    def exact_area(self) -> Callable[[Sequence[float]], float]:
        """Return the deterministic normalized-area callable for raw vectors."""

        return self.input_space.exact_normalized_area


@lru_cache(maxsize=1)
def authoritative_input_space() -> InputSpace:
    """Return the resolved bounds used to create the authoritative 11-var DoE."""
    return InputSpace(
        names=ACTIVE_PARAMETER_NAMES,
        lower=ACTIVE_PARAMETER_LOWER.copy(),
        upper=ACTIVE_PARAMETER_UPPER.copy(),
        nominal=ACTIVE_PARAMETER_NOMINAL.copy(),
    )


def substrate_dimensions(values: Mapping[str, Any]) -> tuple[float, float, float]:
    """Calculate the board dimensions exactly, without constructing polygons."""

    slot_length = float(values["SLOT_MAIN_LENGTH"])
    side_margin = float(values["PATCH_BRICK_1_SIDE_MARGIN"])
    top_margin = float(values["PATCH_BRICK_1_TOP_MARGIN"])
    brick_2_height = float(values["PATCH_BRICK_2_HEIGHT_MARGIN"])
    dimensions = (slot_length, side_margin, top_margin, brick_2_height)
    if not all(math.isfinite(item) for item in dimensions):
        raise ValueError("substrate dimensions require finite values")

    # Fixed values contribute: SLOT_MAIN_HEIGHT 2, lower margin 2,
    # PATCH_BRICK_4_MARGIN 4, two 1-mm offsets and a 13-mm segment.
    width_mm = slot_length + 2.0 * (1.0 + side_margin)
    height_mm = top_margin + brick_2_height + 23.0
    if width_mm <= 0.0 or height_mm <= 0.0:
        raise ValueError("computed substrate dimensions must be positive")
    return width_mm, height_mm, width_mm * height_mm


@lru_cache(maxsize=1)
def reference_substrate_area_mm2() -> float:
    """Area of the standard/code-default antenna; this design is exactly 1."""

    space = authoritative_input_space()
    return substrate_dimensions(space.values(space.nominal))[2]


def normalized_substrate_area(values: Mapping[str, Any]) -> float:
    return substrate_dimensions(values)[2] / reference_substrate_area_mm2()


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
        return ffs_path.parent / CAP_CACHE_FILENAME
    cache_root = Path(cache_directory)
    identity = hashlib.sha256(str(ffs_path.resolve()).casefold().encode("utf-8")).hexdigest()
    return cache_root / f"{identity}.json"


def _read_cap_cache(cache_path: Path, contract: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("contract") != dict(contract):
        return None
    try:
        linear = float(payload[CAP_GAIN_LINEAR_COLUMN])
        dbi = float(payload[CAP_GAIN_DBI_COLUMN])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(linear) and linear > 0.0 and math.isfinite(dbi)):
        return None
    return linear, dbi


def _write_cap_cache(
    cache_path: Path,
    contract: Mapping[str, Any],
    linear: float,
    dbi: float,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": dict(contract),
        CAP_GAIN_LINEAR_COLUMN: linear,
        CAP_GAIN_DBI_COLUMN: dbi,
        "storage_note": "average in linear power; K-RVEA objective is stored as dBi",
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, cache_path)
    except OSError:
        # Read-only result archives remain usable; they merely lose persistence.
        return
    finally:
        try:
            if "temporary" in locals() and temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def _uncached_cap_gain_linear(
    ffs_path: Path,
    *,
    band_ghz: tuple[float, float],
    theta_max_deg: int,
) -> float:
    from scripts.postprocessing.cap_gain import cap_gain, parse_ffs

    frame = cap_gain(parse_ffs(ffs_path), [theta_max_deg])
    selected = frame.loc[
        frame["theta_max_deg"] == theta_max_deg,
        ["freq_ghz", "G_realized_dBi"],
    ].sort_values("freq_ghz", kind="stable")
    frequency = selected["freq_ghz"].to_numpy(dtype=np.float64)
    gain_linear = np.power(
        10.0,
        selected["G_realized_dBi"].to_numpy(dtype=np.float64) / 10.0,
    )
    low, high = map(float, band_ghz)
    if frequency.size == 0 or not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("invalid or empty cap-gain frequency data")
    mask = (frequency >= low) & (frequency <= high)
    if not np.any(mask):
        raise ValueError(f"farfield source has no samples in [{low:g}, {high:g}] GHz")
    # This intentionally matches build_cap_gain_tables.py: arithmetic frequency
    # mean of already angle-averaged gain, always in linear power.
    scalar = float(np.mean(gain_linear[mask]))
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("cap-averaged realized gain must be finite and positive")
    return scalar


def cap_gain_scalar(
    ffs_path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
    theta_max_deg: int = CAP_THETA_MAX_DEG,
    cache_directory: str | Path | None = None,
) -> tuple[float, float, bool]:
    """Return band/cap-averaged realized gain as (linear, dBi, cache_hit)."""

    ffs_path = Path(ffs_path).resolve()
    manifest = {} if manifest is None else manifest
    contract = {
        "metric_version": CAP_METRIC_VERSION,
        "ffs": _farfield_identity(ffs_path, manifest),
        "band_ghz": [float(band_ghz[0]), float(band_ghz[1])],
        "theta_max_deg": int(theta_max_deg),
        "angle_average": "IEEE-145 solid-angle average in linear power",
        "frequency_average": "arithmetic mean in linear power",
        "gain_type": "realized_gain",
    }
    cache_path = _cache_path(ffs_path, cache_directory)
    cached = _read_cap_cache(cache_path, contract)
    if cached is not None:
        return cached[0], cached[1], True
    linear = _uncached_cap_gain_linear(
        ffs_path,
        band_ghz=band_ghz,
        theta_max_deg=theta_max_deg,
    )
    dbi = float(10.0 * math.log10(linear))
    _write_cap_cache(cache_path, contract, linear, dbi)
    return linear, dbi, False


def _validate_fixed_parameters(parameters: Mapping[str, Any], manifest_path: Path) -> None:
    missing = sorted(set(FIXED_PARAMETER_VALUES) - set(parameters))
    if missing:
        raise ValueError(
            f"manifest is missing fixed parameters {missing}: {manifest_path}"
        )
    mismatches = []
    for name, expected in FIXED_PARAMETER_VALUES.items():
        actual = float(parameters[name])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
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
    """Reduce one completed or explicitly penalized case to four objectives."""

    manifest_path = Path(manifest_path).resolve()
    case_directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    status = str(manifest.get("status", ""))
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"manifest has no parameter mapping: {manifest_path}")
    missing = set(ACTIVE_PARAMETER_NAMES) - set(parameters)
    if missing:
        raise ValueError(f"manifest is missing active parameters {sorted(missing)}")
    if strict_fixed_parameters:
        _validate_fixed_parameters(parameters, manifest_path)
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
            raise ValueError(f"penalized manifest has no objective mapping: {manifest_path}")
        worst_s11 = float(objective_payload.get(WORST_S11_COLUMN, 1.0))
        mean_eff = float(objective_payload.get(MEAN_TOT_EFF_COLUMN, 0.0))
        normalized_area_objective = float(
            objective_payload.get(NORMALIZED_AREA_COLUMN, PENALTY_NORMALIZED_AREA)
        )
        if CAP_GAIN_LINEAR_COLUMN in objective_payload:
            cap_linear = float(objective_payload[CAP_GAIN_LINEAR_COLUMN])
            cap_dbi = 10.0 * math.log10(cap_linear)
        elif CAP_GAIN_DBI_COLUMN in objective_payload:
            cap_dbi = float(objective_payload[CAP_GAIN_DBI_COLUMN])
            cap_linear = float(np.power(10.0, cap_dbi / 10.0))
        else:
            cap_linear = PENALTY_CAP_GAIN_LINEAR
            cap_dbi = 10.0 * math.log10(cap_linear)
        kept = int(objective_payload.get("tot_eff_samples_kept", 0))
        removed = int(objective_payload.get("tot_eff_samples_removed_above_one", 0))
        cache_hit = False
    elif status == "completed":
        s11_path = _artifact_path(manifest, case_directory, "s11", qlogehvi.S11_FILENAME)
        eff_path = _artifact_path(manifest, case_directory, "tot_eff", qlogehvi.TOT_EFF_FILENAME)
        ffs_path = _artifact_path(manifest, case_directory, "farfield_source", FARFIELD_FILENAME)
        missing_artifacts = [path.name for path in (s11_path, eff_path, ffs_path) if not path.is_file()]
        if missing_artifacts:
            raise IncompleteObservationError(
                f"completed manifest is missing objective artifact(s) {missing_artifacts}: {manifest_path}"
            )
        worst_s11, mean_eff, kept, removed = qlogehvi.rf_objectives_from_curves(
            s11_path,
            eff_path,
            band_ghz=band_ghz,
        )
        cap_linear, cap_dbi, cache_hit = cap_gain_scalar(
            ffs_path,
            manifest=manifest,
            band_ghz=band_ghz,
            cache_directory=cache_directory,
        )
        normalized_area_objective = area / reference_substrate_area_mm2()
    else:
        raise IncompleteObservationError(
            f"manifest status {status!r} is not a terminal observation: {manifest_path}"
        )

    if not 0.0 <= worst_s11 <= 1.0:
        raise ValueError(f"worst S11 must lie in [0,1]: {manifest_path}")
    if not 0.0 <= mean_eff <= 1.0:
        raise ValueError(f"mean Tot_Eff must lie in [0,1]: {manifest_path}")
    if not math.isfinite(normalized_area_objective) or normalized_area_objective <= 0.0:
        raise ValueError(f"normalized area objective must be positive: {manifest_path}")
    if not math.isfinite(cap_linear) or cap_linear <= 0.0:
        raise ValueError(f"cap realized gain must be positive: {manifest_path}")

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
        MEAN_TOT_EFF_COLUMN: mean_eff,
        TOT_EFF_LOSS_COLUMN: 1.0 - mean_eff,
        CAP_GAIN_LINEAR_COLUMN: cap_linear,
        CAP_GAIN_DBI_COLUMN: cap_dbi,
        "cap_cache_hit": cache_hit,
        "tot_eff_samples_kept": kept,
        "tot_eff_samples_removed_above_one": removed,
    }


def _manifest_paths(source_root: Path) -> list[Path]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"result source does not exist: {source_root}")
    direct = source_root / qlogehvi.MANIFEST_FILENAME
    if direct.is_file():
        return [direct]
    return sorted(source_root.rglob(qlogehvi.MANIFEST_FILENAME))


def collect_observations(
    source_roots: Sequence[str | Path],
    *,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
    cache_directory: str | Path | None = None,
    skipped_incomplete: list[str] | None = None,
    strict_fixed_parameters: bool = True,
) -> pd.DataFrame:
    """Collect terminal observations, skipping only genuinely incomplete cases."""

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
        raise ValueError("no completed or penalized K-RVEA observations were found")
    frame = pd.DataFrame.from_records(records)
    frame.sort_values(["source_root", "case_id"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def build_dataset(
    observations: pd.DataFrame,
    *,
    input_space: InputSpace | None = None,
) -> Dataset:
    """Deduplicate designs and expose raw/unit inputs plus four min objectives."""

    space = authoritative_input_space() if input_space is None else input_space
    missing = set(space.names + OBJECTIVE_NAMES + ("is_penalty",)) - set(observations.columns)
    if missing:
        raise ValueError(f"observation table is missing columns {sorted(missing)}")
    raw = observations.loc[:, space.names].to_numpy(dtype=np.float64)
    unit = space.normalize(raw)
    tolerance = 1e-9
    if np.any(unit < -tolerance) or np.any(unit > 1.0 + tolerance):
        raise ValueError("one or more observations fall outside the authoritative 11-D bounds")
    working = observations.copy()
    working["_unit_key"] = [tuple(np.round(row, 12)) for row in np.clip(unit, 0.0, 1.0)]

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
                CAP_GAIN_LINEAR_COLUMN: float(np.power(10.0, objective[3] / 10.0)),
                CAP_GAIN_DBI_COLUMN: float(objective[3]),
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
    skipped_incomplete: list[str] | None = None,
) -> Dataset:
    """Collect and deduplicate one or more result roots in one call."""

    observations = collect_observations(
        source_roots,
        band_ghz=band_ghz,
        cache_directory=cache_directory,
        skipped_incomplete=skipped_incomplete,
    )
    return build_dataset(observations)
