"""Reusable data and acquisition logic for the MSABP qLogEHVI campaign.

BoTorch imports are deliberately lazy.  Geometry/result auditing therefore
remains testable with the repository's ordinary Python, while the actual GP
fit and qLogEHVI proposal require the dedicated optimization environment.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
OPTIMIZATION_PENALTY_FILENAME = "optimization_penalty.json"
S11_FILENAME = "S11.csv"
TOT_EFF_FILENAME = "Tot_Eff.csv"
PARAMETER_COLUMNS: tuple[str, ...] = ()

WORST_S11_COLUMN = "worst_s11_linear_amplitude"
MEAN_TOT_EFF_COLUMN = "mean_total_efficiency_linear"
AREA_COLUMN = "substrate_area_mm2"
PENALTY_WORST_S11 = 1.0
PENALTY_MEAN_TOT_EFF = 0.0

FLOAT_PAIR_RE = re.compile(
    r"^\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)


@dataclass(frozen=True)
class InputSpace:
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.shape != (len(self.names),) or upper.shape != lower.shape:
            raise ValueError("input-space bounds do not match parameter names")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("input-space bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("every input-space upper bound must exceed its lower bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def normalize(self, raw_values: np.ndarray) -> np.ndarray:
        return (np.asarray(raw_values, dtype=float) - self.lower) / (
            self.upper - self.lower
        )

    def denormalize(self, unit_values: np.ndarray) -> np.ndarray:
        return self.lower + np.asarray(unit_values, dtype=float) * (
            self.upper - self.lower
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_names": list(self.names),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "normalization": "x_unit=(x_raw-lower)/(upper-lower)",
        }


@dataclass(frozen=True)
class ProposalSettings:
    q: int = 4
    seed: int = 20260807
    raw_samples: int = 256
    num_restarts: int = 8
    mc_samples: int = 64
    optimization_batch_limit: int = 2
    optimization_maxiter: int = 100
    gp_training_steps: int = 50
    gp_fixed_noise_variance: float = 1e-6


@dataclass(frozen=True)
class ProposalResult:
    unit_values: np.ndarray
    raw_values: np.ndarray
    acquisition_values: tuple[float, ...]
    diagnostics: dict[str, Any]


def parameter_columns() -> tuple[str, ...]:
    """Import the authoritative parameter order only when needed."""

    global PARAMETER_COLUMNS
    if not PARAMETER_COLUMNS:
        from scripts.automation import antenna_sampler

        PARAMETER_COLUMNS = tuple(antenna_sampler.PARAMETER_REGISTRY)
    return PARAMETER_COLUMNS


def input_space_from_sampling_config(path: str | Path) -> InputSpace:
    from scripts.automation import antenna_sampler

    config = antenna_sampler.load_sampling_config(path)
    plan = antenna_sampler.resolve_sampling_plan(config, n_samples=1)
    sampled = [item for item in plan.resolved_parameters if item.effective_sample]
    names = tuple(item.spec.name for item in sampled)
    expected = parameter_columns()
    if names != expected:
        raise ValueError(
            "qLogEHVI requires all 23 parameters sampled in authoritative order"
        )
    return InputSpace(
        names=names,
        lower=np.asarray([item.lower for item in sampled], dtype=float),
        upper=np.asarray([item.upper for item in sampled], dtype=float),
    )


def substrate_dimensions_from_values(
    values: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Return exact board width, height, and area without building polygons."""

    slot_length = float(values["SLOT_MAIN_LENGTH"])
    slot_height = float(values["SLOT_MAIN_HEIGHT"])
    side_margin = float(values["PATCH_BRICK_1_SIDE_MARGIN"])
    top_margin = float(values["PATCH_BRICK_1_TOP_MARGIN"])
    bottom_margin = float(values["PATCH_BRICK_3_BOTTOM_MARGIN"])
    brick_2_height = float(values["PATCH_BRICK_2_HEIGHT_MARGIN"])
    brick_4_margin = float(values["PATCH_BRICK_4_MARGIN"])
    dimensions = (
        slot_length,
        slot_height,
        side_margin,
        top_margin,
        bottom_margin,
        brick_2_height,
        brick_4_margin,
    )
    if not all(math.isfinite(value) for value in dimensions):
        raise ValueError("substrate dimensions require finite parameter values")

    # See shapely_antenna_model.build_antenna_geometry.  The two fixed 1 mm
    # offsets and fixed lower brick segment (13 mm) contribute 15 mm total.
    width_mm = slot_length + 2.0 * (1.0 + side_margin)
    height_mm = (
        slot_height
        + top_margin
        + bottom_margin
        + brick_2_height
        + brick_4_margin
        + 15.0
    )
    if width_mm <= 0.0 or height_mm <= 0.0:
        raise ValueError("computed substrate dimensions must be positive")
    return width_mm, height_mm, width_mm * height_mm


def substrate_dimensions_from_array(
    raw_values: Sequence[float],
    input_space: InputSpace,
) -> tuple[float, float, float]:
    values = np.asarray(raw_values, dtype=float)
    if values.shape != (len(input_space.names),):
        raise ValueError("raw parameter vector has the wrong shape")
    return substrate_dimensions_from_values(dict(zip(input_space.names, values)))


def read_cst_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    frequencies: list[float] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            match = FLOAT_PAIR_RE.match(line)
            if match is None:
                continue
            frequency = float(match.group(1))
            value = float(match.group(2))
            if math.isfinite(frequency) and math.isfinite(value):
                frequencies.append(frequency)
                values.append(value)
    if not frequencies:
        raise ValueError(f"no numeric CST curve rows in {path}")
    frequency_array = np.asarray(frequencies, dtype=float)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(frequency_array, kind="stable")
    frequency_array = frequency_array[order]
    value_array = value_array[order]
    if np.any(np.diff(frequency_array) <= 0.0):
        raise ValueError(f"CST curve frequencies must be strictly increasing: {path}")
    return frequency_array, value_array


def _band_values_with_endpoints(
    frequencies_ghz: np.ndarray,
    values: np.ndarray,
    band_ghz: tuple[float, float],
) -> np.ndarray:
    low, high = map(float, band_ghz)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("analysis band must satisfy finite low < high")
    if frequencies_ghz[0] > low or frequencies_ghz[-1] < high:
        raise ValueError(
            f"curve [{frequencies_ghz[0]:g}, {frequencies_ghz[-1]:g}] GHz "
            f"does not cover [{low:g}, {high:g}] GHz"
        )
    interior = (frequencies_ghz > low) & (frequencies_ghz < high)
    return np.concatenate(
        (
            [np.interp(low, frequencies_ghz, values)],
            values[interior],
            [np.interp(high, frequencies_ghz, values)],
        )
    )


def rf_objectives_from_curves(
    s11_path: str | Path,
    tot_eff_path: str | Path,
    *,
    band_ghz: tuple[float, float],
) -> tuple[float, float, int, int]:
    """Return linear worst S11 and mean total efficiency.

    Tot_Eff samples above unity are discarded individually as solver defects.
    The mean is the arithmetic mean of the remaining linear samples.
    """

    s11_frequency, s11_db = read_cst_curve(s11_path)
    s11_linear = np.power(10.0, s11_db / 20.0)
    s11_band = _band_values_with_endpoints(s11_frequency, s11_linear, band_ghz)
    if np.any(s11_band < 0.0) or np.any(s11_band > 1.0 + 1e-9):
        raise ValueError("S11 linear amplitude falls outside passive range [0, 1]")
    worst_s11 = float(np.max(np.clip(s11_band, 0.0, 1.0)))

    efficiency_frequency, efficiency_db = read_cst_curve(tot_eff_path)
    efficiency_linear = np.power(10.0, efficiency_db / 10.0)
    efficiency_band = _band_values_with_endpoints(
        efficiency_frequency,
        efficiency_linear,
        band_ghz,
    )
    valid_efficiency = efficiency_band <= 1.0
    removed = int((~valid_efficiency).sum())
    kept = efficiency_band[valid_efficiency]
    if kept.size == 0:
        raise ValueError("all Tot_Eff samples in the band exceed unity")
    if np.any(kept < 0.0):
        raise ValueError("Tot_Eff linear values must be non-negative")
    mean_efficiency = float(np.mean(kept))
    return worst_s11, mean_efficiency, int(kept.size), removed


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


def parse_case_manifest(
    manifest_path: str | Path,
    *,
    source_root: str | Path,
    band_ghz: tuple[float, float],
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    case_directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    status = str(manifest.get("status", ""))
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"manifest has no parameter mapping: {manifest_path}")
    names = parameter_columns()
    missing = set(names) - set(parameters)
    if missing:
        raise ValueError(f"manifest is missing parameters {sorted(missing)}")
    raw_parameters = {name: float(parameters[name]) for name in names}
    width_mm, height_mm, area_mm2 = substrate_dimensions_from_values(raw_parameters)

    penalty_sidecar = case_directory / OPTIMIZATION_PENALTY_FILENAME
    sidecar_payload: Mapping[str, Any] | None = None
    if penalty_sidecar.is_file():
        loaded_sidecar = json.loads(penalty_sidecar.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded_sidecar, Mapping):
            raise ValueError(f"invalid optimization penalty sidecar: {penalty_sidecar}")
        sidecar_payload = loaded_sidecar
    is_penalty = status == "penalized" or sidecar_payload is not None
    if is_penalty:
        optimization = (
            sidecar_payload.get("optimization_objectives")
            if sidecar_payload is not None
            else manifest.get("optimization_objectives")
        )
        if not isinstance(optimization, Mapping):
            raise ValueError(f"penalized manifest has no objectives: {manifest_path}")
        worst_s11 = float(optimization[WORST_S11_COLUMN])
        mean_efficiency = float(optimization[MEAN_TOT_EFF_COLUMN])
        efficiency_kept = int(optimization.get("tot_eff_samples_kept", 0))
        efficiency_removed = int(
            optimization.get("tot_eff_samples_removed_above_one", 0)
        )
    elif status == "completed":
        worst_s11, mean_efficiency, efficiency_kept, efficiency_removed = (
            rf_objectives_from_curves(
                _artifact_path(manifest, case_directory, "s11", S11_FILENAME),
                _artifact_path(
                    manifest,
                    case_directory,
                    "tot_eff",
                    TOT_EFF_FILENAME,
                ),
                band_ghz=band_ghz,
            )
        )
    else:
        raise ValueError(f"unsupported manifest status {status!r}: {manifest_path}")

    if not 0.0 <= worst_s11 <= 1.0:
        raise ValueError(f"worst S11 must be in [0,1]: {manifest_path}")
    if not 0.0 <= mean_efficiency <= 1.0:
        raise ValueError(f"mean Tot_Eff must be in [0,1]: {manifest_path}")
    return {
        "source": Path(source_root).resolve().name,
        "source_root": str(Path(source_root).resolve()),
        "case_id": str(manifest.get("case_id", case_directory.name)),
        "case_directory": str(case_directory),
        "status": "penalized" if is_penalty else status,
        "is_penalty": is_penalty,
        **raw_parameters,
        "substrate_width_mm": width_mm,
        "substrate_height_mm": height_mm,
        AREA_COLUMN: area_mm2,
        WORST_S11_COLUMN: worst_s11,
        MEAN_TOT_EFF_COLUMN: mean_efficiency,
        "tot_eff_samples_kept": efficiency_kept,
        "tot_eff_samples_removed_above_one": efficiency_removed,
    }


def _manifest_paths(source_root: Path) -> list[Path]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"result source does not exist: {source_root}")
    direct = source_root / MANIFEST_FILENAME
    if direct.is_file():
        return [direct]
    return sorted(source_root.rglob(MANIFEST_FILENAME))


def collect_observations(
    source_roots: Sequence[str | Path],
    *,
    band_ghz: tuple[float, float],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source_root_value in source_roots:
        source_root = Path(source_root_value).resolve()
        manifests = _manifest_paths(source_root)
        if not manifests:
            # An empty target directory is valid at the start of a campaign.
            continue
        for manifest_path in manifests:
            resolved = manifest_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append(
                parse_case_manifest(
                    resolved,
                    source_root=source_root,
                    band_ghz=band_ghz,
                )
            )
    if not records:
        raise ValueError("no completed or penalized observations were found")
    frame = pd.DataFrame.from_records(records)
    frame.sort_values(["source_root", "case_id"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def training_arrays(
    observations: pd.DataFrame,
    input_space: InputSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Aggregate exact duplicate inputs and produce unstandardized objectives."""

    raw_x = observations.loc[:, input_space.names].to_numpy(dtype=float)
    unit_x = input_space.normalize(raw_x)
    tolerance = 1e-9
    if np.any(unit_x < -tolerance) or np.any(unit_x > 1.0 + tolerance):
        raise ValueError("one or more source observations fall outside the sampling space")
    unit_x = np.clip(unit_x, 0.0, 1.0)
    working = observations.copy()
    working["_unit_key"] = [
        tuple(np.round(row, 12)) for row in unit_x
    ]
    aggregated_records: list[dict[str, Any]] = []
    aggregated_x: list[np.ndarray] = []
    for key, group in working.groupby("_unit_key", sort=False):
        # Completed data supersede penalty placeholders at the same design.
        completed = group.loc[~group["is_penalty"].astype(bool)]
        selected = completed if not completed.empty else group
        aggregated_x.append(np.asarray(key, dtype=float))
        aggregated_records.append(
            {
                WORST_S11_COLUMN: float(selected[WORST_S11_COLUMN].mean()),
                MEAN_TOT_EFF_COLUMN: float(selected[MEAN_TOT_EFF_COLUMN].mean()),
                AREA_COLUMN: float(selected[AREA_COLUMN].iloc[0]),
                "replicate_count": int(len(selected)),
                "has_completed_result": bool(not completed.empty),
            }
        )
    aggregate = pd.DataFrame.from_records(aggregated_records)
    train_x = np.asarray(aggregated_x, dtype=float)
    train_y_rf = np.column_stack(
        (
            -aggregate[WORST_S11_COLUMN].to_numpy(dtype=float),
            aggregate[MEAN_TOT_EFF_COLUMN].to_numpy(dtype=float),
        )
    )
    train_y_full = np.column_stack(
        (
            train_y_rf,
            -aggregate[AREA_COLUMN].to_numpy(dtype=float),
        )
    )
    return train_x, train_y_rf, train_y_full, aggregate


def maximum_substrate_area(input_space: InputSpace) -> float:
    return substrate_dimensions_from_array(input_space.upper, input_space)[2]


def reference_point(input_space: InputSpace) -> np.ndarray:
    """A fixed maximization reference; penalty RF values lie on its boundary."""

    return np.asarray(
        [-PENALTY_WORST_S11, PENALTY_MEAN_TOT_EFF, -1.01 * maximum_substrate_area(input_space)],
        dtype=float,
    )


def preflight_candidate(
    raw_values: Sequence[float],
    input_space: InputSpace,
    *,
    coordinate_quantum_mm: float = 0.01,
) -> tuple[bool, str, dict[str, Any]]:
    """Run the same full geometry checks used before Princess simulation."""

    from scripts.automation import antenna_sampler
    from scripts.automation import check_sampled_curve_intersections
    from scripts.automation import cst_build_msabp_geometry
    from scripts.geometry import shapely_antenna_model

    mapping = dict(zip(input_space.names, np.asarray(raw_values, dtype=float)))
    width_mm, height_mm, area_mm2 = substrate_dimensions_from_values(mapping)
    report = {
        "substrate_width_mm": width_mm,
        "substrate_height_mm": height_mm,
        AREA_COLUMN: area_mm2,
    }
    try:
        parameters = antenna_sampler.parameters_from_csv_row(mapping)
        payload = shapely_antenna_model.polygon_export_payload(
            parameters,
            quantize_step_mm=coordinate_quantum_mm,
        )
        checks = payload["meta"]["self_intersection_check"]
        invalid = [
            name
            for name, item in checks.items()
            if not (item["ring_is_simple"] and item["polygon_is_valid"])
        ]
        if invalid:
            raise ValueError("invalid exported polygon(s): " + ", ".join(invalid))
        relations = check_sampled_curve_intersections.inspect_curve_boundaries(payload)
        intersections = {
            item
            for item in str(relations["non_bottom_intersection_pairs"]).split(";")
            if item
        }
        crossing = {
            item
            for item in str(relations["non_bottom_crossing_pairs"]).split(";")
            if item
        }
        touching = {
            item
            for item in str(relations["non_bottom_touching_pairs"]).split(";")
            if item
        }
        overlapping = {
            item
            for item in str(relations["non_bottom_overlapping_pairs"]).split(";")
            if item
        }
        allowed = {"Slot__CPW_Feed_Pin"}
        unexpected = intersections - allowed
        if unexpected:
            raise ValueError(
                "unexpected non-bottom curve intersection: "
                + ", ".join(sorted(unexpected))
            )
        if (crossing | touching) & allowed:
            raise ValueError("Slot/CPW fixed relation became crossing or touching")
        if allowed - overlapping:
            raise ValueError("required Slot/CPW overlap is missing")
        _, geometry_report = cst_build_msabp_geometry.build_sampled_polygon_specs(
            parameters,
            coordinate_quantum_mm=coordinate_quantum_mm,
        )
        report["geometry"] = asdict(geometry_report)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", report
    return True, "", report


def penalty_manifest_payload(
    *,
    case_id: str,
    parameters: Mapping[str, Any],
    failure_stage: str,
    failure_message: str,
    band_ghz: tuple[float, float],
) -> dict[str, Any]:
    width_mm, height_mm, area_mm2 = substrate_dimensions_from_values(parameters)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(case_id),
        "status": "penalized",
        "dry_run": False,
        "parameters": {
            name: float(parameters[name]) for name in parameter_columns()
        },
        "geometry": {
            "substrate_width_mm": width_mm,
            "substrate_height_mm": height_mm,
            AREA_COLUMN: area_mm2,
        },
        "failure": {
            "stage": str(failure_stage),
            "message": str(failure_message),
        },
        "optimization_objectives": {
            "band_ghz": [float(band_ghz[0]), float(band_ghz[1])],
            WORST_S11_COLUMN: PENALTY_WORST_S11,
            MEAN_TOT_EFF_COLUMN: PENALTY_MEAN_TOT_EFF,
            AREA_COLUMN: area_mm2,
            "tot_eff_samples_kept": 0,
            "tot_eff_samples_removed_above_one": 0,
            "is_penalty": True,
        },
        "artifacts": {},
    }


def _ensure_environment_scripts_on_path() -> None:
    import sys

    scripts_directory = Path(sys.executable).resolve().parent / "Scripts"
    if not scripts_directory.is_dir():
        return
    entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    if str(scripts_directory).casefold() not in {entry.casefold() for entry in entries}:
        os.environ["PATH"] = os.pathsep.join([str(scripts_directory), *entries])


def _ensure_msvc_environment() -> None:
    """Load the VS x64 compiler environment into this process when available."""

    if os.name != "nt" or shutil.which("cl") is not None:
        return
    vswhere = Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if not vswhere.is_file():
        return
    discovery = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    installation = discovery.stdout.strip()
    if discovery.returncode != 0 or not installation:
        return
    developer_command = Path(installation) / "Common7" / "Tools" / "VsDevCmd.bat"
    if not developer_command.is_file():
        return
    environment = subprocess.run(
        (
            f'call "{developer_command}" -arch=x64 -host_arch=x64 '
            ">nul && set"
        ),
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if environment.returncode != 0:
        return
    allowed_exact = {
        "PATH",
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "VCINSTALLDIR",
        "VCTOOLSINSTALLDIR",
        "WINDOWSSDKDIR",
        "WINDOWSSDKVERSION",
        "UNIVERSALCRTSDKDIR",
        "UCRTVERSION",
    }
    for line in environment.stdout.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        normalized = name.upper()
        if normalized in allowed_exact or normalized.startswith("VSCMD_"):
            os.environ[name] = value
    # Keep cl.exe version output ASCII so torch's compiler probe is independent
    # of the Windows display language / active code page.
    os.environ["VSLANG"] = "1033"


def _botorch_imports() -> dict[str, Any]:
    _ensure_environment_scripts_on_path()
    _ensure_msvc_environment()
    try:
        import torch as torch_module
        import torch.utils.cpp_extension as torch_cpp_extension
        from botorch.acquisition.multi_objective.logei import (
            qLogExpectedHypervolumeImprovement as qlog_ehvi_class,
        )
        from botorch.acquisition.multi_objective.objective import (
            MCMultiOutputObjective as mc_multi_output_objective_class,
        )
        from botorch.exceptions.warnings import InputDataWarning as input_data_warning
        from botorch.fit import fit_gpytorch_mll_torch as fit_mll_torch
        from botorch.models import SingleTaskGP as single_task_gp_class
        from botorch.optim.optimize import optimize_acqf as optimize_acquisition
        from botorch.sampling.normal import (
            SobolQMCNormalSampler as sobol_normal_sampler_class,
        )
        from botorch.utils.multi_objective.box_decompositions.non_dominated import (
            NondominatedPartitioning as nondominated_partitioning_class,
        )
        from gpytorch.mlls import (
            ExactMarginalLogLikelihood as exact_marginal_log_likelihood_class,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qLogEHVI requires botorch, torch, and gpytorch in the current "
            "interpreter; run this controller with the prepared optimization environment"
        ) from exc
    if os.name == "nt" and shutil.which("cl") is not None:
        compiler_output = subprocess.check_output(
            ["cl"], stderr=subprocess.STDOUT
        )
        for encoding in ("utf-8", "mbcs", "cp936", "cp1252"):
            try:
                compiler_output.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
            torch_cpp_extension.SUBPROCESS_DECODE_ARGS = (encoding,)
            break
    return {
        "torch": torch_module,
        "qLogExpectedHypervolumeImprovement": qlog_ehvi_class,
        "MCMultiOutputObjective": mc_multi_output_objective_class,
        "InputDataWarning": input_data_warning,
        "fit_gpytorch_mll_torch": fit_mll_torch,
        "SingleTaskGP": single_task_gp_class,
        "optimize_acqf": optimize_acquisition,
        "SobolQMCNormalSampler": sobol_normal_sampler_class,
        "NondominatedPartitioning": nondominated_partitioning_class,
        "ExactMarginalLogLikelihood": exact_marginal_log_likelihood_class,
    }


def propose_qlogehvi_batch(
    observations: pd.DataFrame,
    input_space: InputSpace,
    *,
    settings: ProposalSettings,
    iteration: int,
) -> ProposalResult:
    """Fit two CPU/float64 RF GPs and propose a qLogEHVI batch."""

    if settings.q < 1:
        raise ValueError("q must be at least one")
    if settings.raw_samples < settings.num_restarts:
        raise ValueError("raw_samples must be at least num_restarts")
    if settings.num_restarts < 1:
        raise ValueError("num_restarts must be at least one")
    runtime = _botorch_imports()
    torch = runtime["torch"]
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    dtype = torch.float64

    train_x, train_y_rf, train_y_full, aggregate = training_arrays(
        observations,
        input_space,
    )
    if len(train_x) < 2:
        raise ValueError("at least two distinct observations are required for GP fitting")
    x_tensor = torch.as_tensor(train_x, dtype=dtype, device=device)
    y_tensor = torch.as_tensor(train_y_rf, dtype=dtype, device=device)
    y_variance = torch.full_like(y_tensor, settings.gp_fixed_noise_variance)
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
    mll = runtime["ExactMarginalLogLikelihood"](model.likelihood, model)
    fit_started = perf_counter()
    fit_result = runtime["fit_gpytorch_mll_torch"](
        mll,
        step_limit=settings.gp_training_steps,
        timeout_sec=600,
    )
    model.eval()
    gp_fit_seconds = perf_counter() - fit_started

    lower = torch.as_tensor(input_space.lower, dtype=dtype, device=device)
    span = torch.as_tensor(
        input_space.upper - input_space.lower,
        dtype=dtype,
        device=device,
    )
    index = {name: input_space.names.index(name) for name in input_space.names}
    base_objective = runtime["MCMultiOutputObjective"]

    class ExactAreaObjective(base_objective):
        def forward(self, samples: Any, X: Any | None = None) -> Any:
            if X is None:
                raise ValueError("exact area objective requires candidate X")
            raw = lower + X * span
            width = raw[..., index["SLOT_MAIN_LENGTH"]] + 2.0 * (
                1.0 + raw[..., index["PATCH_BRICK_1_SIDE_MARGIN"]]
            )
            height = (
                raw[..., index["SLOT_MAIN_HEIGHT"]]
                + raw[..., index["PATCH_BRICK_1_TOP_MARGIN"]]
                + raw[..., index["PATCH_BRICK_3_BOTTOM_MARGIN"]]
                + raw[..., index["PATCH_BRICK_2_HEIGHT_MARGIN"]]
                + raw[..., index["PATCH_BRICK_4_MARGIN"]]
                + 15.0
            )
            negative_area = -(width * height).unsqueeze(-1)
            while negative_area.dim() < samples.dim():
                negative_area = negative_area.unsqueeze(0)
            negative_area = negative_area.expand(*samples.shape[:-1], 1)
            return torch.cat((samples, negative_area), dim=-1)

    ref = reference_point(input_space)
    ref_tensor = torch.as_tensor(ref, dtype=dtype, device=device)
    full_y_tensor = torch.as_tensor(train_y_full, dtype=dtype, device=device)
    partitioning = runtime["NondominatedPartitioning"](
        ref_point=ref_tensor,
        Y=full_y_tensor,
    )
    sampler = runtime["SobolQMCNormalSampler"](
        sample_shape=torch.Size([settings.mc_samples]),
        seed=settings.seed + iteration,
    )
    acquisition = runtime["qLogExpectedHypervolumeImprovement"](
        model=model,
        ref_point=ref_tensor,
        partitioning=partitioning,
        sampler=sampler,
        objective=ExactAreaObjective(),
    )

    torch.manual_seed(settings.seed + 1000 + iteration)
    bounds = torch.stack(
        (
            torch.zeros(len(input_space.names), dtype=dtype, device=device),
            torch.ones(len(input_space.names), dtype=dtype, device=device),
        )
    )
    acquisition_started = perf_counter()
    chosen, acquisition_values = runtime["optimize_acqf"](
        acquisition,
        bounds=bounds,
        q=settings.q,
        num_restarts=settings.num_restarts,
        raw_samples=settings.raw_samples,
        options={
            "batch_limit": settings.optimization_batch_limit,
            "maxiter": settings.optimization_maxiter,
        },
        sequential=False,
    )
    acquisition_seconds = perf_counter() - acquisition_started
    unit_values = chosen.detach().cpu().numpy()
    raw_values = input_space.denormalize(unit_values)
    if unit_values.shape != (settings.q, len(input_space.names)):
        raise RuntimeError(f"unexpected qLogEHVI proposal shape: {unit_values.shape}")
    if len(np.unique(np.round(unit_values, 12), axis=0)) != settings.q:
        raise RuntimeError("qLogEHVI proposed duplicate points inside the batch")
    nearest = np.min(
        np.linalg.norm(unit_values[:, None, :] - train_x[None, :, :], axis=2),
        axis=1,
    )
    if np.any(nearest <= 1e-10):
        raise RuntimeError("qLogEHVI proposed a previously observed point")
    values_array = np.atleast_1d(acquisition_values.detach().cpu().numpy())
    fit_status = str(getattr(fit_result, "status", "unknown"))
    return ProposalResult(
        unit_values=unit_values,
        raw_values=raw_values,
        acquisition_values=tuple(float(value) for value in values_array.ravel()),
        diagnostics={
            "algorithm": "qLogExpectedHypervolumeImprovement",
            "device": "cpu",
            "dtype": "float64",
            "q": settings.q,
            "training_observations_raw": int(len(observations)),
            "training_observations_distinct": int(len(train_x)),
            "penalty_observations": int(observations["is_penalty"].sum()),
            "raw_samples": settings.raw_samples,
            "num_restarts": settings.num_restarts,
            "mc_samples": settings.mc_samples,
            "optimization_batch_limit": settings.optimization_batch_limit,
            "optimization_maxiter": settings.optimization_maxiter,
            "gp_fit_seconds": gp_fit_seconds,
            "acquisition_seconds": acquisition_seconds,
            "gp_outputs": ["negative_worst_s11", MEAN_TOT_EFF_COLUMN],
            "deterministic_output": "negative_substrate_area_mm2",
            "output_standardization": False,
            "reference_point_maximize": ref.tolist(),
            "nearest_observation_distances": nearest.tolist(),
            "fit_status": fit_status,
            "replicate_groups": int((aggregate["replicate_count"] > 1).sum()),
        },
    )
