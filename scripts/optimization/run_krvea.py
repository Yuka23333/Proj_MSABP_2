"""Run the resumable 11-D, four-objective K-RVEA smoke campaign.

The controller owns the immutable campaign plan, observation cache, candidate
manifests, and Princess/Maid execution.  CoconutG2 is used only as the GPU
proposal worker; the two configured Maids remain the expensive-evaluation
workers.  All objectives use minimization semantics internally.

F5 uses the constants below.  A normal run still requires typing ``RUN``.
Use ``--prepare-only`` to validate/cache the 512 historical observations, or
``--stop-after-proposal`` to persist one resumable proposal without CST.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import krvea, krvea_data, qlogehvi  # noqa: E402
from msabp_opt.simulation.distributed.config import (  # noqa: E402
    DEFAULT_DEVICE_CONFIG_PATH,
    load_device_registry,
)
from msabp_opt.simulation.distributed.runtime import (  # noqa: E402
    PrincessRunPaths,
    select_devices,
    validate_run_id,
)
from msabp_opt.simulation.distributed.state import PrincessState  # noqa: E402
from scripts.automation import antenna_sampler  # noqa: E402


F5_PLAN_ID = "msabp-krvea-11var-smoke-128-001"
F5_SOURCE_DIRECTORIES = (
    REPOSITORY_ROOT / "results" / "raw" / "doe-11var-branch-up-lhs-512-001",
)
F5_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "raw" / F5_PLAN_ID
F5_TOTAL_BUDGET = 128
F5_Q = 4
F5_BAND_GHZ = (3.1, 4.8)
F5_DEVICE_IDS = ("convallariag5", "coconutg2")
F5_REQUIRE_CONFIRMATION = True

SAMPLING_CONFIG = (
    REPOSITORY_ROOT / "configs" / "optimization" / "krvea_11var_branch_up.json"
)
DEVICE_CONFIG = DEFAULT_DEVICE_CONFIG_PATH
PROJECT_TEMPLATE = REPOSITORY_ROOT / "simulations" / "models" / "msa-bp.cst"
PRINCESS_SCRIPT = REPOSITORY_ROOT / "scripts" / "simulation" / "princess.py"

EXPECTED_INITIAL_TRAINING_COUNT = 512
NOMINAL_AREA_REFERENCE_MM2 = 2720.2
PENALTY_WORST_S11 = 1.0
PENALTY_MEAN_TOT_EFF = 0.0
PENALTY_TOT_EFF_LOSS = 1.0
PENALTY_NORMALIZED_AREA = 2.0
PENALTY_CAP_GAIN_LINEAR = 10.0
PENALTY_CAP_GAIN_DBI = 10.0
GP_TRAINING_STEPS = 50
GP_FIXED_NOISE_VARIANCE = 1e-6
GP_TIMEOUT_SECONDS = 600.0
UPSTREAM_KRVEA_COMMIT = "1f32028fb974e2b5739eb795a4a008b6edc1a703"

PLAN_FILENAME = "optimization_plan.json"
STATE_FILENAME = "optimization_state.json"
CONTROL_DIRECTORY_NAME = "_krvea"
OBSERVATIONS_FILENAME = "observations.csv"
HISTORY_OBSERVATIONS_FILENAME = "history_observations.csv"
HISTORY_CACHE_META_FILENAME = "history_observations.meta.json"
CAP_CACHE_DIRECTORY_NAME = "cap_gain_cache"
PLAN_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RemoteProposalConfig:
    """Serializable local view of the CoconutG2 proposal endpoint."""

    device_id: str = "coconutg2"
    python_path: str = r"C:\Users\telecom\miniforge3\envs\bocuda\python.exe"
    compute_device: str = "cuda"
    timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("proposal device id must be non-empty")
        if not Path(self.python_path).drive:
            raise ValueError("proposal Python must be an absolute Windows path")
        if not self.compute_device.strip():
            raise ValueError("proposal compute device must be non-empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("proposal timeout must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


F5_PROPOSAL_REMOTE = RemoteProposalConfig()
F5_PROPOSAL = krvea.KRVEAConfig(
    n_variables=len(krvea_data.ACTIVE_PARAMETER_NAMES),
    n_objectives=4,
    reference_partitions=7,
    q=F5_Q,
    inner_evaluations=10_000,
    seed=20260829,
)


@dataclass(frozen=True)
class CampaignConfig:
    plan_id: str
    source_directories: tuple[Path, ...]
    output_directory: Path
    total_budget: int = F5_TOTAL_BUDGET
    band_ghz: tuple[float, float] = F5_BAND_GHZ
    device_ids: tuple[str, ...] = F5_DEVICE_IDS
    sampling_config: Path = SAMPLING_CONFIG
    device_config: Path = DEVICE_CONFIG
    project_template: Path = PROJECT_TEMPLATE
    proposal: krvea.KRVEAConfig = F5_PROPOSAL
    proposal_remote: RemoteProposalConfig = F5_PROPOSAL_REMOTE
    coordinate_quantum_mm: float = 0.01
    allow_disconnected_conductor: bool = False
    max_attempts: int = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format=antenna_sampler.CSV_FLOAT_FORMAT,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _unique_resolved_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        identity = str(path).casefold()
        if identity not in seen:
            seen.add(identity)
            result.append(path)
    return tuple(result)


def _validate_sampling_contract(path: Path) -> None:
    actual = antenna_sampler.load_sampling_config(path)
    expected = antenna_sampler.load_sampling_config(SAMPLING_CONFIG)
    if actual != expected:
        raise ValueError(
            "K-RVEA sampling config must exactly match "
            f"{SAMPLING_CONFIG.relative_to(REPOSITORY_ROOT)}"
        )
    plan = antenna_sampler.resolve_sampling_plan(actual, n_samples=1)
    active = tuple(
        item.spec.name for item in plan.resolved_parameters if item.effective_sample
    )
    if active != krvea_data.ACTIVE_PARAMETER_NAMES:
        raise RuntimeError(f"unexpected K-RVEA active parameter order: {active!r}")
    sampled = [item for item in plan.resolved_parameters if item.effective_sample]
    if not np.allclose(
        [item.nominal for item in sampled],
        krvea_data.ACTIVE_PARAMETER_NOMINAL,
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        [item.lower for item in sampled],
        krvea_data.ACTIVE_PARAMETER_LOWER,
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        [item.upper for item in sampled],
        krvea_data.ACTIVE_PARAMETER_UPPER,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("tracked K-RVEA bounds or nominal values drifted")
    fixed = {
        item.spec.name: float(item.nominal)
        for item in plan.resolved_parameters
        if not item.effective_sample
    }
    if fixed != krvea_data.FIXED_PARAMETER_VALUES:
        raise RuntimeError("tracked K-RVEA fixed parameter values drifted")


def _validate_config(config: CampaignConfig) -> CampaignConfig:
    validate_run_id(config.plan_id)
    if config.total_budget != F5_TOTAL_BUDGET:
        raise ValueError("this smoke plan requires exactly 128 new evaluations")
    if config.proposal.q != F5_Q:
        raise ValueError("this smoke plan requires q=4")
    if config.proposal.n_variables != len(krvea_data.ACTIVE_PARAMETER_NAMES):
        raise ValueError("K-RVEA proposal must use the authoritative 11 variables")
    if config.proposal.n_objectives != 4:
        raise ValueError("K-RVEA proposal must use four objectives")
    if len(config.source_directories) != 1:
        raise ValueError("this smoke plan requires exactly one historical source")
    if tuple(config.device_ids) != F5_DEVICE_IDS:
        raise ValueError(
            "this smoke plan requires Princess devices convallariag5,coconutg2"
        )
    low, high = config.band_ghz
    if not low < high:
        raise ValueError("band must satisfy low < high")
    sources = _unique_resolved_paths(config.source_directories)
    output = config.output_directory.expanduser().resolve()
    if str(output).casefold() == str(sources[0]).casefold():
        raise ValueError("the output is an automatic source and cannot be historical")
    if not sources[0].is_dir():
        raise FileNotFoundError(f"historical source does not exist: {sources[0]}")
    for path, label in (
        (config.sampling_config, "sampling config"),
        (config.device_config, "device config"),
        (config.project_template, "CST project template"),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    _validate_sampling_contract(config.sampling_config)
    if not np.isclose(
        krvea_data.reference_substrate_area_mm2(),
        NOMINAL_AREA_REFERENCE_MM2,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError("authoritative nominal substrate area is no longer 2720.2 mm2")
    return replace(
        config,
        source_directories=sources,
        output_directory=output,
        sampling_config=config.sampling_config.resolve(),
        device_config=config.device_config.resolve(),
        project_template=config.project_template.resolve(),
    )


def _manifest_snapshot(source: Path) -> dict[str, Any]:
    manifests = sorted(source.rglob(qlogehvi.MANIFEST_FILENAME))
    records = []
    digest = hashlib.sha256()
    for path in manifests:
        relative = path.relative_to(source).as_posix()
        record = f"{relative}\0{_sha256(path)}\n"
        digest.update(record.encode("utf-8"))
        records.append(relative)
    return {
        "manifest_count": len(records),
        "manifest_metadata_sha256": digest.hexdigest(),
    }


def _control_directory(config: CampaignConfig) -> Path:
    return config.output_directory / CONTROL_DIRECTORY_NAME


def _history_cache_contract(config: CampaignConfig) -> dict[str, Any]:
    source = config.source_directories[0]
    cap_gain_source = REPOSITORY_ROOT / "scripts" / "postprocessing" / "cap_gain.py"
    return {
        "schema_version": 1,
        "source_directory": str(source),
        "source_snapshot": _manifest_snapshot(source),
        "band_ghz": list(config.band_ghz),
        "sampling_config_sha256": _sha256(config.sampling_config),
        "expected_trainable_count": EXPECTED_INITIAL_TRAINING_COUNT,
        "objective_extractor": {
            "schema_version": krvea_data.SCHEMA_VERSION,
            "cap_metric_version": krvea_data.CAP_METRIC_VERSION,
            "krvea_data_sha256": _sha256(krvea_data.__file__),
            "cap_gain_sha256": _sha256(cap_gain_source),
        },
    }


def load_or_build_history_cache(
    config: CampaignConfig,
) -> tuple[pd.DataFrame, krvea_data.Dataset, dict[str, Any]]:
    control = _control_directory(config)
    frame_path = control / HISTORY_OBSERVATIONS_FILENAME
    meta_path = control / HISTORY_CACHE_META_FILENAME
    contract = _history_cache_contract(config)
    observations: pd.DataFrame | None = None
    if frame_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if metadata.get("contract") == contract:
            observations = pd.read_csv(frame_path, encoding="utf-8-sig")
    if observations is None:
        skipped: list[str] = []
        observations = krvea_data.collect_observations(
            config.source_directories,
            band_ghz=config.band_ghz,
            cache_directory=control / CAP_CACHE_DIRECTORY_NAME,
            skipped_incomplete=skipped,
        )
        if skipped:
            raise RuntimeError(
                "historical source is not fully trainable; incomplete examples: "
                + "; ".join(skipped[:3])
            )
        _atomic_write_csv(observations, frame_path)
        _atomic_write_json(meta_path, {"contract": contract, "created_at_utc": _utc_now()})
    dataset = krvea_data.build_dataset(observations)
    if len(observations) != EXPECTED_INITIAL_TRAINING_COUNT or len(dataset.x_unit) != EXPECTED_INITIAL_TRAINING_COUNT:
        raise RuntimeError(
            "historical source must contain exactly 512 trainable, distinct designs; "
            f"found rows={len(observations)}, distinct={len(dataset.x_unit)}"
        )
    return observations, dataset, contract["source_snapshot"]


def _plan_payload(
    config: CampaignConfig,
    input_space: krvea_data.InputSpace,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    from msabp_opt.optimization import krvea_relay

    cap_gain_source = REPOSITORY_ROOT / "scripts" / "postprocessing" / "cap_gain.py"
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": config.plan_id,
        "created_at_utc": _utc_now(),
        "algorithm": "K-RVEA",
        "provenance": {
            "implementation": "independent Python reimplementation",
            "upstream_repository": "https://github.com/tichugh/K-RVEA.git",
            "upstream_commit_reviewed": UPSTREAM_KRVEA_COMMIT,
            "citation": (
                "T. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya, "
                "A surrogate-assisted reference vector guided evolutionary "
                "algorithm for computationally expensive many-objective "
                "optimization, IEEE Transactions on Evolutionary Computation, "
                "22(1):129-142, 2018."
            ),
        },
        "campaign": "512 historical + 128 new expensive evaluations",
        "total_budget": config.total_budget,
        "q": config.proposal.q,
        "band_ghz": list(config.band_ghz),
        "source_directories": [str(path) for path in config.source_directories],
        "historical_training_count": EXPECTED_INITIAL_TRAINING_COUNT,
        "historical_source_snapshot": dict(source_snapshot),
        "output_directory": str(config.output_directory),
        "output_is_automatic_training_source": True,
        "sampling_config": str(config.sampling_config),
        "sampling_config_sha256": _sha256(config.sampling_config),
        "input_space": {
            "parameter_names": list(input_space.names),
            "lower": input_space.lower.tolist(),
            "upper": input_space.upper.tolist(),
            "normalization": "x_unit=(x_raw-lower)/(upper-lower)",
            "fixed_parameter_values": dict(krvea_data.FIXED_PARAMETER_VALUES),
        },
        "objectives": [
            {
                "name": krvea_data.WORST_S11_COLUMN,
                "direction": "minimize",
                "domain": "linear_amplitude",
                "penalty": PENALTY_WORST_S11,
            },
            {
                "name": krvea_data.TOT_EFF_LOSS_COLUMN,
                "reported_metric": krvea_data.MEAN_TOT_EFF_COLUMN,
                "direction": "minimize one_minus_mean_efficiency",
                "domain": "linear_power_ratio",
                "samples_above_one": "discard_individually",
                "penalty": PENALTY_TOT_EFF_LOSS,
            },
            {
                "name": krvea_data.NORMALIZED_AREA_COLUMN,
                "direction": "minimize",
                "model": "exact_deterministic_formula",
                "posterior_variance": 0.0,
                "definition": "substrate_area_mm2 / 2720.2",
                "nominal_area_reference_mm2": NOMINAL_AREA_REFERENCE_MM2,
                "nominal_design_value": 1.0,
                "penalty": PENALTY_NORMALIZED_AREA,
                "actual_area_mm2": "diagnostic_only",
            },
            {
                "name": krvea_data.CAP_GAIN_DBI_COLUMN,
                "direction": "minimize",
                "quantity": "realized_gain",
                "theta_deg": [0.0, 15.0],
                "frequency_band_ghz": list(config.band_ghz),
                "averaging": "angle_and_frequency_in_linear_power_then_convert_to_dBi",
                "optimization_scalar": "dBi",
                "penalty_linear": PENALTY_CAP_GAIN_LINEAR,
                "penalty_dbi": PENALTY_CAP_GAIN_DBI,
            },
        ],
        "proposal": {
            **asdict(config.proposal),
            "expensive_objective_indices": [0, 1, 3],
            "exact_objective_indices": [2],
            "dtype": "float64",
            "target_standardization": "per expensive objective on GPU worker",
            "objective_scaler": {
                "fit_rows": "non_penalty_only",
                "center": "median",
                "scale": "max(normalized_iqr, normalized_mad, std, epsilon)",
                "penalty": "greater_than_each_valid_standardized_max_by_3",
            },
            "surrogate_settings": {
                "gp_training_steps": GP_TRAINING_STEPS,
                "gp_fixed_noise_variance": GP_FIXED_NOISE_VARIANCE,
                "gp_timeout_seconds": GP_TIMEOUT_SECONDS,
            },
            "remote": config.proposal_remote.to_dict(),
        },
        "simulation": {
            "device_ids": list(config.device_ids),
            "device_config": str(config.device_config),
            "project_template": str(config.project_template),
            "coordinate_quantum_mm": config.coordinate_quantum_mm,
            "allow_disconnected_conductor": config.allow_disconnected_conductor,
            "max_attempts": config.max_attempts,
        },
        "software": {
            "controller_python": platform.python_version(),
            "implementation_sha256": {
                "run_krvea": _sha256(__file__),
                "krvea": _sha256(krvea.__file__),
                "krvea_data": _sha256(krvea_data.__file__),
                "krvea_relay": _sha256(krvea_relay.__file__),
                "cap_gain": _sha256(cap_gain_source),
            },
        },
    }


def load_or_create_plan(
    config: CampaignConfig,
    input_space: krvea_data.InputSpace,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    config.output_directory.mkdir(parents=True, exist_ok=True)
    path = config.output_directory / PLAN_FILENAME
    expected = _plan_payload(config, input_space, source_snapshot)
    if not path.exists():
        if list(config.output_directory.glob("case_*")):
            raise RuntimeError("refusing to create a plan in a non-empty result directory")
        _atomic_write_json(path, expected)
        return expected
    actual = json.loads(path.read_text(encoding="utf-8-sig"))
    immutable = tuple(name for name in expected if name != "created_at_utc")
    mismatches = [name for name in immutable if actual.get(name) != expected.get(name)]
    if mismatches:
        raise RuntimeError(
            "existing K-RVEA plan differs from requested settings: "
            + ", ".join(mismatches)
        )
    return actual


def _default_state(config: CampaignConfig) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "plan_id": config.plan_id,
        "next_batch_index": 0,
        "next_candidate_index": 0,
        # The authors' MATLAB driver initializes Empty_ref_old to zero.
        "previous_empty_reference_count": 0,
        "active_batch": None,
        "completed_batches": [],
        "updated_at_utc": _utc_now(),
    }


def load_state(config: CampaignConfig) -> dict[str, Any]:
    path = config.output_directory / STATE_FILENAME
    if not path.exists():
        state = _default_state(config)
        _atomic_write_json(path, state)
        return state
    state = json.loads(path.read_text(encoding="utf-8-sig"))
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RuntimeError("unsupported K-RVEA state schema")
    if state.get("plan_id") != config.plan_id:
        raise RuntimeError("K-RVEA state belongs to a different plan")
    return state


def save_state(config: CampaignConfig, state: Mapping[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at_utc"] = _utc_now()
    _atomic_write_json(config.output_directory / STATE_FILENAME, payload)


def target_case_directories(config: CampaignConfig) -> list[Path]:
    case_dirs = sorted(path for path in config.output_directory.glob("case_*") if path.is_dir())
    missing = [path for path in case_dirs if not (path / qlogehvi.MANIFEST_FILENAME).is_file()]
    active_ids: set[str] = set()
    state_path = config.output_directory / STATE_FILENAME
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        active = state.get("active_batch")
        if isinstance(active, Mapping):
            active_ids = {str(value) for value in active.get("case_ids", [])}
    unexpected = [path for path in missing if path.name.removeprefix("case_") not in active_ids]
    if unexpected:
        raise RuntimeError(
            "target contains case directories without manifests outside the active batch: "
            + ", ".join(path.name for path in unexpected[:5])
        )
    if len(case_dirs) > config.total_budget:
        raise RuntimeError(
            f"target raw count {len(case_dirs)} exceeds planned budget {config.total_budget}; "
            "start a new optimization plan"
        )
    return case_dirs


def actual_sources(config: CampaignConfig) -> tuple[Path, ...]:
    return (*config.source_directories, config.output_directory)


def refresh_observations(
    config: CampaignConfig,
    history_observations: pd.DataFrame,
) -> tuple[pd.DataFrame, krvea_data.Dataset]:
    skipped: list[str] = []
    target = pd.DataFrame()
    if list(config.output_directory.glob("case_*")):
        try:
            target = krvea_data.collect_observations(
                (config.output_directory,),
                band_ghz=config.band_ghz,
                cache_directory=_control_directory(config) / CAP_CACHE_DIRECTORY_NAME,
                skipped_incomplete=skipped,
            )
        except ValueError as exc:
            if "no completed or penalized" not in str(exc):
                raise
    observations = (
        pd.concat((history_observations, target), ignore_index=True)
        if not target.empty
        else history_observations.copy()
    )
    _atomic_write_csv(observations, _control_directory(config) / OBSERVATIONS_FILENAME)
    if skipped:
        print(
            f"[K-RVEA] skipped {len(skipped)} incomplete target manifest(s): "
            + "; ".join(skipped[:3]),
            flush=True,
        )
    return observations, krvea_data.build_dataset(observations)


def full_parameter_mapping(active_parameters: Mapping[str, Any]) -> dict[str, float]:
    values = {**krvea_data.FIXED_PARAMETER_VALUES, **active_parameters}
    missing = set(antenna_sampler.PARAMETER_REGISTRY) - set(values)
    if missing:
        raise ValueError(f"full parameter mapping is missing {sorted(missing)}")
    return {
        name: float(values[name])
        for name in antenna_sampler.PARAMETER_REGISTRY
    }


def preflight_full_parameters(
    parameters: Mapping[str, Any],
    *,
    coordinate_quantum_mm: float,
) -> tuple[bool, str, dict[str, Any]]:
    names = tuple(antenna_sampler.PARAMETER_REGISTRY)
    raw = np.asarray([float(parameters[name]) for name in names], dtype=np.float64)
    helper_space = qlogehvi.InputSpace(
        names=names,
        lower=np.zeros(len(names), dtype=np.float64),
        upper=np.ones(len(names), dtype=np.float64),
    )
    return qlogehvi.preflight_candidate(
        raw,
        helper_space,
        coordinate_quantum_mm=coordinate_quantum_mm,
    )


def _candidate_case_id(index: int) -> str:
    return f"krvea_{index:04d}"


def _case_directory(config: CampaignConfig, case_id: str) -> Path:
    return config.output_directory / f"case_{case_id}"


def _write_new_directory_atomic(destination: Path, files: Mapping[str, str]) -> None:
    if destination.exists():
        raise FileExistsError(f"case directory already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary case directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for name, text in files.items():
            (temporary / name).write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


def _penalty_objectives(config: CampaignConfig, parameters: Mapping[str, Any]) -> dict[str, Any]:
    _, _, area = krvea_data.substrate_dimensions(parameters)
    return {
        "band_ghz": list(config.band_ghz),
        krvea_data.WORST_S11_COLUMN: PENALTY_WORST_S11,
        krvea_data.MEAN_TOT_EFF_COLUMN: PENALTY_MEAN_TOT_EFF,
        krvea_data.TOT_EFF_LOSS_COLUMN: PENALTY_TOT_EFF_LOSS,
        krvea_data.AREA_COLUMN: area,
        krvea_data.NORMALIZED_AREA_COLUMN: PENALTY_NORMALIZED_AREA,
        krvea_data.CAP_GAIN_LINEAR_COLUMN: PENALTY_CAP_GAIN_LINEAR,
        krvea_data.CAP_GAIN_DBI_COLUMN: PENALTY_CAP_GAIN_DBI,
        "tot_eff_samples_kept": 0,
        "tot_eff_samples_removed_above_one": 0,
        "is_penalty": True,
    }


def penalty_manifest_payload(
    config: CampaignConfig,
    *,
    case_id: str,
    parameters: Mapping[str, Any],
    failure_stage: str,
    failure_message: str,
) -> dict[str, Any]:
    width, height, area = krvea_data.substrate_dimensions(parameters)
    return {
        "schema_version": 1,
        "case_id": case_id,
        "status": "penalized",
        "dry_run": False,
        "parameters": full_parameter_mapping(parameters),
        "geometry": {
            "substrate_width_mm": width,
            "substrate_height_mm": height,
            krvea_data.AREA_COLUMN: area,
        },
        "failure": {"stage": failure_stage, "message": failure_message},
        "optimization_objectives": _penalty_objectives(config, parameters),
        "artifacts": {},
    }


def write_penalty_case(
    config: CampaignConfig,
    *,
    case_id: str,
    parameters: Mapping[str, Any],
    failure_stage: str,
    failure_message: str,
) -> Path:
    payload = penalty_manifest_payload(
        config,
        case_id=case_id,
        parameters=parameters,
        failure_stage=failure_stage,
        failure_message=failure_message,
    )
    destination = _case_directory(config, case_id)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        manifest = destination / qlogehvi.MANIFEST_FILENAME
        if manifest.exists():
            raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")
        _atomic_write_text(manifest, text)
    else:
        _write_new_directory_atomic(destination, {qlogehvi.MANIFEST_FILENAME: text})
    return destination


def write_completed_case_penalty_sidecar(
    config: CampaignConfig,
    *,
    case_id: str,
    parameters: Mapping[str, Any],
    failure_stage: str,
    failure_message: str,
) -> Path:
    destination = _case_directory(config, case_id) / qlogehvi.OPTIMIZATION_PENALTY_FILENAME
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "failure": {"stage": failure_stage, "message": failure_message},
        "optimization_objectives": _penalty_objectives(config, parameters),
    }
    _atomic_write_json(destination, payload)
    return destination


def _request_remote_proposal(
    config: CampaignConfig,
    dataset: krvea_data.Dataset,
    *,
    batch_index: int,
    q: int,
    remaining_budget: int,
    previous_empty_reference_count: int | None,
) -> Any:
    from msabp_opt.optimization import krvea_relay

    settings = replace(
        config.proposal,
        q=q,
        seed=config.proposal.seed + batch_index,
    )
    penalty_mask = ~dataset.metadata["has_completed_result"].to_numpy(dtype=bool)
    request = krvea_relay.build_request_payload(
        dataset.x_unit,
        dataset.objectives[:, [0, 1, 3]],
        dataset.objectives,
        penalty_mask,
        dataset.input_space,
        config=settings,
        iteration=batch_index,
        remaining_expensive_budget=remaining_budget,
        previous_empty_reference_count=previous_empty_reference_count,
        compute_device=config.proposal_remote.compute_device,
        surrogate_settings=krvea_relay.SurrogateFitSettings(
            gp_training_steps=GP_TRAINING_STEPS,
            gp_fixed_noise_variance=GP_FIXED_NOISE_VARIANCE,
            gp_timeout_seconds=GP_TIMEOUT_SECONDS,
        ),
    )
    control = _control_directory(config)
    request_path = control / f"batch_{batch_index:04d}_proposal_request.json"
    response_path = control / f"batch_{batch_index:04d}_proposal_response.json"
    krvea_relay.write_request(request_path, request)
    registry = load_device_registry(config.device_config)
    device = select_devices(registry, (config.proposal_remote.device_id,))[0]
    remote = krvea_relay.RemoteProposalConfig(**config.proposal_remote.to_dict())
    return krvea_relay.relay_remote_proposal(
        device=device,
        remote=remote,
        plan_id=config.plan_id,
        batch_index=batch_index,
        local_request_path=request_path,
        local_response_path=response_path,
        expected_q=q,
        expected_dimension=len(dataset.input_space.names),
        observed_x_unit=dataset.x_unit,
        input_space=dataset.input_space,
    )


def create_batch(
    config: CampaignConfig,
    state: dict[str, Any],
    dataset: krvea_data.Dataset,
    *,
    q: int,
) -> dict[str, Any]:
    batch_index = int(state["next_batch_index"])
    candidate_start = int(state["next_candidate_index"])
    remaining = config.total_budget - len(target_case_directories(config))
    proposal = _request_remote_proposal(
        config,
        dataset,
        batch_index=batch_index,
        q=q,
        remaining_budget=remaining,
        previous_empty_reference_count=state.get("previous_empty_reference_count"),
    )
    raw_values = np.asarray(proposal.raw_values, dtype=np.float64)
    unit_values = np.asarray(proposal.unit_values, dtype=np.float64)
    expected_shape = (q, len(dataset.input_space.names))
    if raw_values.shape != expected_shape or unit_values.shape != expected_shape:
        raise RuntimeError(f"proposal shape must be {expected_shape}")
    diagnostics = dict(proposal.diagnostics)
    if "empty_reference_count" not in diagnostics:
        raise RuntimeError("proposal diagnostics omit empty_reference_count")

    rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    invalid_ids: list[str] = []
    case_ids: list[str] = []
    for offset, raw in enumerate(raw_values):
        case_id = _candidate_case_id(candidate_start + offset)
        case_ids.append(case_id)
        active = dataset.input_space.values(raw)
        parameters = full_parameter_mapping(active)
        valid, error, geometry = preflight_full_parameters(
            parameters,
            coordinate_quantum_mm=config.coordinate_quantum_mm,
        )
        row = {
            "sample_id": case_id,
            **parameters,
            "optimization_source": "krvea",
            "optimization_batch_index": batch_index,
            "geometry_valid": valid,
            "geometry_error": error,
            "substrate_width_mm": geometry.get("substrate_width_mm"),
            "substrate_height_mm": geometry.get("substrate_height_mm"),
            krvea_data.AREA_COLUMN: geometry.get(krvea_data.AREA_COLUMN),
        }
        rows.append(row)
        if valid:
            valid_rows.append(row)
        else:
            invalid_ids.append(case_id)

    candidate_frame = pd.DataFrame.from_records(rows)
    ordered = ["sample_id", *antenna_sampler.PARAMETER_REGISTRY]
    extras = [name for name in candidate_frame.columns if name not in ordered]
    candidate_frame = candidate_frame.loc[:, [*ordered, *extras]]
    worklist_frame = candidate_frame.loc[
        candidate_frame["geometry_valid"].astype(bool)
    ].copy()
    control = _control_directory(config)
    candidate_path = control / f"batch_{batch_index:04d}_candidates.csv"
    worklist_path = control / f"batch_{batch_index:04d}_worklist.csv"
    _atomic_write_csv(candidate_frame, candidate_path)
    _atomic_write_csv(worklist_frame, worklist_path)
    batch = {
        "batch_index": batch_index,
        "run_id": f"{config.plan_id}-batch-{batch_index:04d}",
        "status": "proposed",
        "case_ids": case_ids,
        "invalid_preflight_case_ids": invalid_ids,
        "candidate_csv": str(candidate_path),
        "worklist_csv": str(worklist_path),
        "proposal_diagnostics": diagnostics,
        "proposed_empty_reference_count": int(diagnostics["empty_reference_count"]),
        "previous_empty_reference_count_at_proposal": state.get("previous_empty_reference_count"),
        "unit_values": unit_values.tolist(),
        "predicted_mean": np.asarray(proposal.predicted_mean, dtype=np.float64).tolist(),
        "predicted_std": np.asarray(proposal.predicted_std, dtype=np.float64).tolist(),
        "created_at_utc": _utc_now(),
    }
    state["active_batch"] = batch
    state["next_batch_index"] = batch_index + 1
    state["next_candidate_index"] = candidate_start + q
    save_state(config, state)

    row_by_id = {str(row["sample_id"]): row for row in rows}
    for case_id in invalid_ids:
        row = row_by_id[case_id]
        if not _case_directory(config, case_id).exists():
            write_penalty_case(
                config,
                case_id=case_id,
                parameters=row,
                failure_stage="geometry_preflight",
                failure_message=str(row["geometry_error"]),
            )
    return batch


def _read_batch_rows(batch: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    with Path(str(batch["candidate_csv"])).open("r", encoding="utf-8-sig", newline="") as stream:
        return {str(row["sample_id"]): row for row in csv.DictReader(stream)}


def build_princess_command(config: CampaignConfig, batch: Mapping[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(PRINCESS_SCRIPT),
        "start",
        "--csv",
        str(batch["worklist_csv"]),
        "--run-id",
        str(batch["run_id"]),
        "--project",
        str(config.project_template),
        "--results-root",
        str(config.output_directory),
        "--devices-config",
        str(config.device_config),
        "--coordinate-quantum-mm",
        str(config.coordinate_quantum_mm),
        "--max-attempts",
        str(config.max_attempts),
    ]
    for device_id in config.device_ids:
        command.extend(("--device", device_id))
    if config.allow_disconnected_conductor:
        command.append("--allow-disconnected-conductor")
    return command


def run_and_tee(command: Sequence[str], log_path: Path, *, output: TextIO = sys.stdout) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n$ {subprocess.list2cmdline(list(command))}\n")
        process = subprocess.Popen(
            list(command),
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            log.write(line)
        return process.wait()


def _task_records(config: CampaignConfig, batch: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = PrincessRunPaths.for_run(
        str(batch["run_id"]),
        repository_root=REPOSITORY_ROOT,
        results_root=config.output_directory,
    )
    if not paths.database.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    invalid = set(batch["invalid_preflight_case_ids"])
    with PrincessState(paths.database) as princess_state:
        for case_id in batch["case_ids"]:
            if case_id in invalid:
                continue
            try:
                result[str(case_id)] = princess_state.get_task(str(batch["run_id"]), str(case_id))
            except Exception:
                continue
    return result


def _penalize_terminal_failure(
    config: CampaignConfig,
    *,
    case_id: str,
    parameters: Mapping[str, Any],
    stage: str,
    message: str,
) -> None:
    manifest = _case_directory(config, case_id) / qlogehvi.MANIFEST_FILENAME
    if manifest.is_file():
        write_completed_case_penalty_sidecar(
            config,
            case_id=case_id,
            parameters=parameters,
            failure_stage=stage,
            failure_message=message,
        )
    else:
        write_penalty_case(
            config,
            case_id=case_id,
            parameters=parameters,
            failure_stage=stage,
            failure_message=message,
        )


def finalize_active_batch(
    config: CampaignConfig,
    state: dict[str, Any],
    *,
    princess_exit_code: int | None,
) -> bool:
    batch = state.get("active_batch")
    if not isinstance(batch, Mapping):
        return True
    rows = _read_batch_rows(batch)
    invalid = set(str(value) for value in batch["invalid_preflight_case_ids"])
    tasks = _task_records(config, batch)
    unresolved: list[str] = []
    outcomes: dict[str, str] = {}
    for value in batch["case_ids"]:
        case_id = str(value)
        row = rows[case_id]
        case_directory = _case_directory(config, case_id)
        if case_id in invalid:
            if not (case_directory / qlogehvi.MANIFEST_FILENAME).is_file():
                write_penalty_case(
                    config,
                    case_id=case_id,
                    parameters=row,
                    failure_stage="geometry_preflight",
                    failure_message=str(row["geometry_error"]),
                )
            outcomes[case_id] = "penalized_geometry"
            continue
        task = tasks.get(case_id)
        task_status = str(task.get("status", "")) if task else ""
        manifest = case_directory / qlogehvi.MANIFEST_FILENAME
        if task_status == "completed":
            if not manifest.is_file():
                _penalize_terminal_failure(
                    config,
                    case_id=case_id,
                    parameters=row,
                    stage="objective_extraction",
                    message="Princess marked the task completed but manifest.json is missing",
                )
                outcomes[case_id] = "penalized_missing_manifest"
                continue
            try:
                krvea_data.parse_manifest(
                    manifest,
                    source_root=config.output_directory,
                    band_ghz=config.band_ghz,
                    cache_directory=_control_directory(config) / CAP_CACHE_DIRECTORY_NAME,
                )
            except Exception as exc:
                write_completed_case_penalty_sidecar(
                    config,
                    case_id=case_id,
                    parameters=row,
                    failure_stage="objective_extraction",
                    failure_message=f"{type(exc).__name__}: {exc}",
                )
                outcomes[case_id] = "penalized_objective_extraction"
            else:
                outcomes[case_id] = "completed"
            continue
        if task_status == "failed":
            _penalize_terminal_failure(
                config,
                case_id=case_id,
                parameters=row,
                stage=str(task.get("last_error_kind", "cst_attempts_exhausted")),
                message=str(task.get("last_error_message", "")),
            )
            outcomes[case_id] = "penalized_cst_failure"
            continue
        unresolved.append(case_id)

    if unresolved:
        active = dict(batch)
        active["status"] = "awaiting_princess_resume"
        active["last_princess_exit_code"] = princess_exit_code
        active["unresolved_case_ids"] = unresolved
        state["active_batch"] = active
        save_state(config, state)
        return False

    completed = dict(batch)
    completed["status"] = "completed"
    completed["completed_at_utc"] = _utc_now()
    completed["princess_exit_code"] = princess_exit_code
    completed["outcomes"] = outcomes
    state["completed_batches"] = [*state.get("completed_batches", []), completed]
    state["previous_empty_reference_count"] = int(batch["proposed_empty_reference_count"])
    state["active_batch"] = None
    save_state(config, state)
    return True


def execute_active_batch(config: CampaignConfig, state: dict[str, Any]) -> bool:
    batch = state.get("active_batch")
    if not isinstance(batch, Mapping):
        return True
    invalid = set(str(value) for value in batch["invalid_preflight_case_ids"])
    valid = [str(case_id) for case_id in batch["case_ids"] if str(case_id) not in invalid]
    exit_code: int | None = None
    if valid:
        command = build_princess_command(config, batch)
        log_path = REPOSITORY_ROOT / "logs" / f"princess.{batch['run_id']}.krvea.log"
        exit_code = run_and_tee(command, log_path)
    return finalize_active_batch(config, state, princess_exit_code=exit_code)


def validate_devices(config: CampaignConfig) -> None:
    registry = load_device_registry(config.device_config)
    select_devices(registry, config.device_ids)
    proposal = select_devices(registry, (config.proposal_remote.device_id,))[0]
    if not proposal.is_remote:
        raise ValueError("the K-RVEA GPU proposal device must be SSH-addressable")


def run_campaign(
    config: CampaignConfig,
    *,
    prepare_only: bool = False,
    stop_after_proposal: bool = False,
) -> int:
    config = _validate_config(config)
    history, history_dataset, snapshot = load_or_build_history_cache(config)
    load_or_create_plan(config, history_dataset.input_space, snapshot)
    state = load_state(config)
    observations, dataset = refresh_observations(config, history)
    target_count = len(target_case_directories(config))
    print(
        f"[K-RVEA] history={len(history_dataset.x_unit)} "
        f"training={len(dataset.x_unit)} target={target_count}/{config.total_budget} "
        f"q={config.proposal.q} proposal={config.proposal_remote.device_id}:"
        f"{config.proposal_remote.compute_device} dtype=float64",
        flush=True,
    )
    if prepare_only:
        return 0
    validate_devices(config)

    while True:
        target_count = len(target_case_directories(config))
        if state.get("active_batch") is not None:
            if stop_after_proposal:
                print("[K-RVEA] active batch preserved; Princess was not started")
                return 0
            if not execute_active_batch(config, state):
                active = state["active_batch"]
                raise RuntimeError(
                    "Princess batch is not terminal; fix infrastructure and rerun "
                    f"{active['run_id']}: {active.get('unresolved_case_ids', [])}"
                )
            observations, dataset = refresh_observations(config, history)
            continue
        if target_count == config.total_budget:
            refresh_observations(config, history)
            print(f"[K-RVEA] campaign complete: {target_count}/{config.total_budget}")
            return 0
        remaining = config.total_budget - target_count
        q = min(config.proposal.q, remaining)
        observations, dataset = refresh_observations(config, history)
        batch = create_batch(config, state, dataset, q=q)
        diagnostics = batch["proposal_diagnostics"]
        print(
            f"[K-RVEA] proposed batch={batch['batch_index']} "
            f"cases={','.join(batch['case_ids'])} "
            f"preflight_invalid={len(batch['invalid_preflight_case_ids'])} "
            f"mode={diagnostics.get('mode', 'unknown')} "
            f"empty_refs={diagnostics['empty_reference_count']}",
            flush=True,
        )
        if stop_after_proposal:
            print("[K-RVEA] stop-after-proposal requested; active batch is resumable")
            return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--source", action="append", dest="sources", type=Path)
    parser.add_argument("--output", type=Path, default=F5_OUTPUT_DIRECTORY)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--band", nargs=2, type=float, default=None)
    parser.add_argument("--device", action="append", dest="device_ids")
    parser.add_argument("--sampling-config", type=Path, default=None)
    parser.add_argument("--devices-config", type=Path, default=None)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--reference-partitions", type=int, default=None)
    parser.add_argument("--inner-evaluations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--proposal-device", default=None)
    parser.add_argument("--proposal-python", default=None)
    parser.add_argument("--proposal-compute-device", default=None)
    parser.add_argument("--proposal-timeout-seconds", type=float, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-after-proposal", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> CampaignConfig:
    output = args.output.expanduser().resolve()
    plan_path = output / PLAN_FILENAME
    existing: Mapping[str, Any] = {}
    if plan_path.is_file():
        loaded = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"invalid existing K-RVEA plan: {plan_path}")
        existing = loaded
    simulation = existing.get("simulation", {})
    if not isinstance(simulation, Mapping):
        simulation = {}
    proposal_saved = existing.get("proposal", {})
    if not isinstance(proposal_saved, Mapping):
        proposal_saved = {}
    remote_saved = proposal_saved.get("remote", {})
    if not isinstance(remote_saved, Mapping):
        remote_saved = {}

    sources = (
        tuple(args.sources)
        if args.sources
        else tuple(Path(value) for value in existing.get("source_directories", ()))
        or F5_SOURCE_DIRECTORIES
    )
    devices = (
        tuple(args.device_ids)
        if args.device_ids
        else tuple(str(value) for value in simulation.get("device_ids", ()))
        or F5_DEVICE_IDS
    )
    band = args.band if args.band is not None else existing.get("band_ghz", F5_BAND_GHZ)
    q = int(args.q if args.q is not None else existing.get("q", F5_Q))

    def saved_or(argument: Any, name: str, fallback: Any) -> Any:
        return argument if argument is not None else proposal_saved.get(name, fallback)

    return CampaignConfig(
        plan_id=str(args.plan_id or existing.get("plan_id", F5_PLAN_ID)),
        source_directories=tuple(Path(path) for path in sources),
        output_directory=output,
        total_budget=int(args.budget if args.budget is not None else existing.get("total_budget", F5_TOTAL_BUDGET)),
        band_ghz=(float(band[0]), float(band[1])),
        device_ids=devices,
        sampling_config=Path(args.sampling_config or existing.get("sampling_config", SAMPLING_CONFIG)),
        device_config=Path(args.devices_config or simulation.get("device_config", DEVICE_CONFIG)),
        project_template=Path(args.project or simulation.get("project_template", PROJECT_TEMPLATE)),
        proposal=krvea.KRVEAConfig(
            n_variables=len(krvea_data.ACTIVE_PARAMETER_NAMES),
            n_objectives=4,
            reference_partitions=int(saved_or(args.reference_partitions, "reference_partitions", 7)),
            q=q,
            inner_evaluations=int(saved_or(args.inner_evaluations, "inner_evaluations", 10_000)),
            population_size=(
                int(args.population_size)
                if args.population_size is not None
                else (
                    int(proposal_saved["population_size"])
                    if proposal_saved.get("population_size") is not None
                    else None
                )
            ),
            seed=int(saved_or(args.seed, "seed", F5_PROPOSAL.seed)),
        ),
        proposal_remote=RemoteProposalConfig(
            device_id=str(args.proposal_device or remote_saved.get("device_id", F5_PROPOSAL_REMOTE.device_id)),
            python_path=str(args.proposal_python or remote_saved.get("python_path", F5_PROPOSAL_REMOTE.python_path)),
            compute_device=str(args.proposal_compute_device or remote_saved.get("compute_device", F5_PROPOSAL_REMOTE.compute_device)),
            timeout_seconds=float(
                args.proposal_timeout_seconds
                if args.proposal_timeout_seconds is not None
                else remote_saved.get("timeout_seconds", F5_PROPOSAL_REMOTE.timeout_seconds)
            ),
        ),
        coordinate_quantum_mm=float(simulation.get("coordinate_quantum_mm", 0.01)),
        allow_disconnected_conductor=bool(simulation.get("allow_disconnected_conductor", False)),
        max_attempts=int(simulation.get("max_attempts", 3)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _config_from_args(args)
    if not args.prepare_only and not args.stop_after_proposal:
        if F5_REQUIRE_CONFIRMATION and not args.yes:
            answer = input(
                f"Type RUN to start/resume K-RVEA plan {config.plan_id} "
                f"({config.total_budget} new evaluations): "
            )
            if answer.strip() != "RUN":
                print("Cancelled; no proposal worker or solver was started.")
                return 1
    try:
        return run_campaign(
            config,
            prepare_only=args.prepare_only,
            stop_after_proposal=args.stop_after_proposal,
        )
    except KeyboardInterrupt:
        print("[K-RVEA] interrupted; plan and active batch remain resumable")
        return 130
    except Exception as exc:
        print(f"K-RVEA error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
