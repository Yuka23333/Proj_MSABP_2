"""Run a resumable qLogEHVI campaign through Princess/Maid.

F5 uses the constants below.  The output directory owns an immutable
``optimization_plan.json`` whose budget counts only cases created in that
directory; all configured historical sources plus the output directory are
merged into the GP training set on every batch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import qlogehvi  # noqa: E402
from msabp_opt.simulation.distributed.config import (  # noqa: E402
    DEFAULT_DEVICE_CONFIG_PATH,
    load_device_registry,
)
from msabp_opt.simulation.distributed.runtime import (  # noqa: E402
    PrincessRunPaths,
    select_devices,
)
from msabp_opt.simulation.distributed.state import PrincessState  # noqa: E402
from scripts.automation import antenna_sampler  # noqa: E402


# IDE / F5 campaign settings.
F5_PLAN_ID = "msabp-qlogehvi-001"
F5_SOURCE_DIRECTORIES = (
    REPOSITORY_ROOT / "results" / "raw" / "doe-round1-lhs-512",
)
F5_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "raw" / F5_PLAN_ID
F5_TOTAL_BUDGET = 200
F5_Q = 4
F5_BAND_GHZ = (3.1, 4.8)
F5_DEVICE_IDS = ("convallariag5", "coconutg2")
F5_REQUIRE_CONFIRMATION = True

SAMPLING_CONFIG = (
    REPOSITORY_ROOT / "configs" / "optimization" / "antenna_sampling.json"
)
DEVICE_CONFIG = DEFAULT_DEVICE_CONFIG_PATH
PROJECT_TEMPLATE = REPOSITORY_ROOT / "simulations" / "models" / "msa-bp.cst"
PRINCESS_SCRIPT = REPOSITORY_ROOT / "scripts" / "simulation" / "princess.py"

PLAN_FILENAME = "optimization_plan.json"
STATE_FILENAME = "optimization_state.json"
CONTROL_DIRECTORY_NAME = "_qlogehvi"
OBSERVATIONS_FILENAME = "observations.csv"
PLAN_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1


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
    proposal: qlogehvi.ProposalSettings = qlogehvi.ProposalSettings(q=F5_Q)
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
        if identity in seen:
            continue
        seen.add(identity)
        result.append(path)
    return tuple(result)


def _validate_config(config: CampaignConfig) -> CampaignConfig:
    if not config.plan_id or any(character.isspace() for character in config.plan_id):
        raise ValueError("plan_id must be non-empty and contain no whitespace")
    if config.total_budget < 1:
        raise ValueError("total_budget must be at least one")
    if config.proposal.q < 1:
        raise ValueError("q must be at least one")
    if not config.source_directories:
        raise ValueError("at least one historical source directory is required")
    if not config.device_ids:
        raise ValueError("at least one Maid device is required")
    if len(set(config.device_ids)) != len(config.device_ids):
        raise ValueError("Maid device ids must be unique")
    low, high = config.band_ghz
    if not low < high:
        raise ValueError("band must satisfy low < high")
    sources = _unique_resolved_paths(config.source_directories)
    output = config.output_directory.expanduser().resolve()
    if str(output).casefold() in {str(path).casefold() for path in sources}:
        raise ValueError("output directory is added automatically and must not be a source")
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"source directory does not exist: {source}")
    if not config.sampling_config.is_file():
        raise FileNotFoundError(f"sampling config does not exist: {config.sampling_config}")
    return CampaignConfig(
        **{
            **asdict(config),
            "source_directories": sources,
            "output_directory": output,
            "sampling_config": config.sampling_config.resolve(),
            "device_config": config.device_config.resolve(),
            "project_template": config.project_template.resolve(),
            "proposal": config.proposal,
        }
    )


def _plan_payload(config: CampaignConfig, input_space: qlogehvi.InputSpace) -> dict[str, Any]:
    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": config.plan_id,
        "created_at_utc": _utc_now(),
        "total_budget": config.total_budget,
        "q": config.proposal.q,
        "band_ghz": list(config.band_ghz),
        "source_directories": [str(path) for path in config.source_directories],
        "output_directory": str(config.output_directory),
        "sampling_config": str(config.sampling_config),
        "sampling_config_sha256": _sha256(config.sampling_config),
        "input_space": input_space.to_dict(),
        "algorithm": "qLogExpectedHypervolumeImprovement",
        "compute": {"device": "cpu", "dtype": "float64", "cuda": False},
        "software": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "botorch": package_version("botorch"),
            "gpytorch": package_version("gpytorch"),
            "ninja": package_version("ninja"),
        },
        "objectives": [
            {
                "name": qlogehvi.WORST_S11_COLUMN,
                "direction": "minimize",
                "domain": "linear_amplitude",
                "penalty": qlogehvi.PENALTY_WORST_S11,
            },
            {
                "name": qlogehvi.MEAN_TOT_EFF_COLUMN,
                "direction": "maximize",
                "domain": "linear_power_ratio",
                "penalty": qlogehvi.PENALTY_MEAN_TOT_EFF,
                "samples_above_one": "discard_individually",
            },
            {
                "name": qlogehvi.AREA_COLUMN,
                "direction": "minimize",
                "model": "exact_deterministic_formula",
                "posterior_variance": 0.0,
            },
        ],
        "proposal_settings": asdict(config.proposal),
        "simulation": {
            "device_ids": list(config.device_ids),
            "device_config": str(config.device_config),
            "project_template": str(config.project_template),
            "coordinate_quantum_mm": config.coordinate_quantum_mm,
            "allow_disconnected_conductor": config.allow_disconnected_conductor,
            "max_attempts": config.max_attempts,
        },
    }


def load_or_create_plan(
    config: CampaignConfig,
    input_space: qlogehvi.InputSpace,
) -> dict[str, Any]:
    config.output_directory.mkdir(parents=True, exist_ok=True)
    path = config.output_directory / PLAN_FILENAME
    expected = _plan_payload(config, input_space)
    if not path.exists():
        existing_case_dirs = list(config.output_directory.glob("case_*"))
        if existing_case_dirs:
            raise RuntimeError(
                "refusing to create a plan inside a non-empty raw result directory"
            )
        _atomic_write_json(path, expected)
        return expected
    actual = json.loads(path.read_text(encoding="utf-8-sig"))
    immutable_fields = (
        "schema_version",
        "plan_id",
        "total_budget",
        "q",
        "band_ghz",
        "source_directories",
        "output_directory",
        "sampling_config_sha256",
        "input_space",
        "algorithm",
        "compute",
        "software",
        "objectives",
        "proposal_settings",
        "simulation",
    )
    mismatches = [name for name in immutable_fields if actual.get(name) != expected.get(name)]
    if mismatches:
        raise RuntimeError(
            "existing optimization plan differs from requested settings: "
            + ", ".join(mismatches)
        )
    return actual


def _default_state(config: CampaignConfig) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "plan_id": config.plan_id,
        "next_batch_index": 0,
        "next_candidate_index": 0,
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
        raise RuntimeError("unsupported optimization state schema")
    if state.get("plan_id") != config.plan_id:
        raise RuntimeError("optimization state belongs to a different plan")
    return state


def save_state(config: CampaignConfig, state: Mapping[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at_utc"] = _utc_now()
    _atomic_write_json(config.output_directory / STATE_FILENAME, payload)


def target_case_directories(config: CampaignConfig) -> list[Path]:
    case_dirs = sorted(
        path for path in config.output_directory.glob("case_*") if path.is_dir()
    )
    missing = [path for path in case_dirs if not (path / qlogehvi.MANIFEST_FILENAME).is_file()]
    if missing:
        raise RuntimeError(
            "target contains case directories without manifest: "
            + ", ".join(path.name for path in missing[:5])
        )
    if len(case_dirs) > config.total_budget:
        raise RuntimeError(
            f"target raw count {len(case_dirs)} exceeds planned budget "
            f"{config.total_budget}; start a new optimization plan"
        )
    return case_dirs


def actual_sources(config: CampaignConfig) -> tuple[Path, ...]:
    return _unique_resolved_paths((*config.source_directories, config.output_directory))


def refresh_observations(config: CampaignConfig) -> pd.DataFrame:
    observations = qlogehvi.collect_observations(
        actual_sources(config),
        band_ghz=config.band_ghz,
    )
    destination = (
        config.output_directory / CONTROL_DIRECTORY_NAME / OBSERVATIONS_FILENAME
    )
    _atomic_write_csv(observations, destination)
    return observations


def _candidate_case_id(index: int) -> str:
    return f"bo_{index:04d}"


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


def write_penalty_case(
    config: CampaignConfig,
    *,
    case_id: str,
    parameters: Mapping[str, Any],
    failure_stage: str,
    failure_message: str,
) -> Path:
    payload = qlogehvi.penalty_manifest_payload(
        case_id=case_id,
        parameters=parameters,
        failure_stage=failure_stage,
        failure_message=failure_message,
        band_ghz=config.band_ghz,
    )
    destination = _case_directory(config, case_id)
    _write_new_directory_atomic(
        destination,
        {
            qlogehvi.MANIFEST_FILENAME: json.dumps(
                payload, ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n"
        },
    )
    return destination


def write_completed_case_penalty_sidecar(
    config: CampaignConfig,
    *,
    case_id: str,
    parameters: Mapping[str, Any],
    failure_message: str,
) -> Path:
    case_directory = _case_directory(config, case_id)
    if not (case_directory / qlogehvi.MANIFEST_FILENAME).is_file():
        raise FileNotFoundError(f"completed case manifest is missing: {case_directory}")
    full_payload = qlogehvi.penalty_manifest_payload(
        case_id=case_id,
        parameters=parameters,
        failure_stage="objective_extraction",
        failure_message=failure_message,
        band_ghz=config.band_ghz,
    )
    sidecar = {
        "schema_version": 1,
        "case_id": case_id,
        "failure": full_payload["failure"],
        "optimization_objectives": full_payload["optimization_objectives"],
    }
    destination = case_directory / qlogehvi.OPTIMIZATION_PENALTY_FILENAME
    _atomic_write_json(destination, sidecar)
    return destination


def create_batch(
    config: CampaignConfig,
    state: dict[str, Any],
    observations: pd.DataFrame,
    input_space: qlogehvi.InputSpace,
    *,
    q: int,
) -> dict[str, Any]:
    batch_index = int(state["next_batch_index"])
    candidate_start = int(state["next_candidate_index"])
    settings = qlogehvi.ProposalSettings(**{**asdict(config.proposal), "q": q})
    proposal = qlogehvi.propose_qlogehvi_batch(
        observations,
        input_space,
        settings=settings,
        iteration=batch_index,
    )
    control_directory = config.output_directory / CONTROL_DIRECTORY_NAME
    candidate_path = control_directory / f"batch_{batch_index:04d}_candidates.csv"
    worklist_path = control_directory / f"batch_{batch_index:04d}_worklist.csv"
    rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    invalid_ids: list[str] = []
    case_ids: list[str] = []
    for offset, raw_values in enumerate(proposal.raw_values):
        case_id = _candidate_case_id(candidate_start + offset)
        case_ids.append(case_id)
        parameters = {
            name: float(value)
            for name, value in zip(input_space.names, raw_values)
        }
        valid, error, geometry = qlogehvi.preflight_candidate(
            raw_values,
            input_space,
            coordinate_quantum_mm=config.coordinate_quantum_mm,
        )
        row = {
            "sample_id": case_id,
            "bo_source": "qlogehvi",
            "bo_batch_index": batch_index,
            **parameters,
            "geometry_valid": valid,
            "geometry_error": error,
            "substrate_width_mm": geometry["substrate_width_mm"],
            "substrate_height_mm": geometry["substrate_height_mm"],
            qlogehvi.AREA_COLUMN: geometry[qlogehvi.AREA_COLUMN],
        }
        rows.append(row)
        if valid:
            valid_rows.append(row)
        else:
            invalid_ids.append(case_id)
    candidate_frame = pd.DataFrame.from_records(rows)
    worklist_frame = pd.DataFrame.from_records(valid_rows, columns=candidate_frame.columns)
    _atomic_write_csv(candidate_frame, candidate_path)
    _atomic_write_csv(worklist_frame, worklist_path)
    run_id = f"{config.plan_id}-batch-{batch_index:04d}"
    batch = {
        "batch_index": batch_index,
        "run_id": run_id,
        "status": "proposed",
        "case_ids": case_ids,
        "invalid_preflight_case_ids": invalid_ids,
        "candidate_csv": str(candidate_path),
        "worklist_csv": str(worklist_path),
        "proposal_diagnostics": proposal.diagnostics,
        "acquisition_values": list(proposal.acquisition_values),
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
    with Path(str(batch["candidate_csv"])).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        return {
            str(row["sample_id"]): row
            for row in csv.DictReader(stream)
        }


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


def run_and_tee(
    command: Sequence[str],
    log_path: Path,
    *,
    output: TextIO = sys.stdout,
) -> int:
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
    with PrincessState(paths.database) as state:
        for case_id in batch["case_ids"]:
            if case_id in set(batch["invalid_preflight_case_ids"]):
                continue
            try:
                result[str(case_id)] = state.get_task(str(batch["run_id"]), str(case_id))
            except Exception:
                continue
    return result


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
    invalid_ids = set(str(value) for value in batch["invalid_preflight_case_ids"])
    tasks = _task_records(config, batch)
    unresolved: list[str] = []
    outcomes: dict[str, str] = {}
    for case_id_value in batch["case_ids"]:
        case_id = str(case_id_value)
        row = rows[case_id]
        case_directory = _case_directory(config, case_id)
        if case_id in invalid_ids:
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
        manifest_path = case_directory / qlogehvi.MANIFEST_FILENAME
        if task_status == "completed" and manifest_path.is_file():
            try:
                qlogehvi.parse_case_manifest(
                    manifest_path,
                    source_root=config.output_directory,
                    band_ghz=config.band_ghz,
                )
            except Exception as exc:
                write_completed_case_penalty_sidecar(
                    config,
                    case_id=case_id,
                    parameters=row,
                    failure_message=f"{type(exc).__name__}: {exc}",
                )
                outcomes[case_id] = "penalized_objective_extraction"
            else:
                outcomes[case_id] = "completed"
            continue
        if task_status == "failed":
            write_penalty_case(
                config,
                case_id=case_id,
                parameters=row,
                failure_stage=str(task.get("last_error_kind", "cst_attempts_exhausted")),
                failure_message=str(task.get("last_error_message", "")),
            )
            outcomes[case_id] = "penalized_cst_failure"
            continue
        unresolved.append(case_id)

    if unresolved:
        batch = dict(batch)
        batch["status"] = "awaiting_princess_resume"
        batch["last_princess_exit_code"] = princess_exit_code
        batch["unresolved_case_ids"] = unresolved
        state["active_batch"] = batch
        save_state(config, state)
        return False

    completed = dict(batch)
    completed["status"] = "completed"
    completed["completed_at_utc"] = _utc_now()
    completed["princess_exit_code"] = princess_exit_code
    completed["outcomes"] = outcomes
    state["completed_batches"] = [*state.get("completed_batches", []), completed]
    state["active_batch"] = None
    save_state(config, state)
    return True


def execute_active_batch(
    config: CampaignConfig,
    state: dict[str, Any],
) -> bool:
    batch = state.get("active_batch")
    if not isinstance(batch, Mapping):
        return True
    invalid_ids = set(str(value) for value in batch["invalid_preflight_case_ids"])
    valid_ids = [
        str(case_id) for case_id in batch["case_ids"] if str(case_id) not in invalid_ids
    ]
    exit_code: int | None = None
    if valid_ids:
        command = build_princess_command(config, batch)
        log_path = REPOSITORY_ROOT / "logs" / f"princess.{batch['run_id']}.qlogehvi.log"
        exit_code = run_and_tee(command, log_path)
    return finalize_active_batch(
        config,
        state,
        princess_exit_code=exit_code,
    )


def validate_devices(config: CampaignConfig) -> None:
    registry = load_device_registry(config.device_config)
    select_devices(registry, config.device_ids)


def run_campaign(
    config: CampaignConfig,
    *,
    prepare_only: bool = False,
    stop_after_proposal: bool = False,
) -> int:
    config = _validate_config(config)
    input_space = qlogehvi.input_space_from_sampling_config(config.sampling_config)
    load_or_create_plan(config, input_space)
    state = load_state(config)
    target_count = len(target_case_directories(config))
    observations = refresh_observations(config)
    print(
        f"[qLogEHVI] sources={len(actual_sources(config))} "
        f"observations={len(observations)} target={target_count}/{config.total_budget} "
        f"q={config.proposal.q} device=cpu dtype=float64",
        flush=True,
    )
    if prepare_only:
        return 0
    validate_devices(config)

    while True:
        target_count = len(target_case_directories(config))
        if state.get("active_batch") is not None:
            if stop_after_proposal:
                print("[qLogEHVI] active batch preserved; solver not started")
                return 0
            if not execute_active_batch(config, state):
                batch = state["active_batch"]
                raise RuntimeError(
                    "Princess batch is not terminal; fix infrastructure and rerun to "
                    f"resume {batch['run_id']}: {batch.get('unresolved_case_ids', [])}"
                )
            observations = refresh_observations(config)
            continue

        if target_count == config.total_budget:
            refresh_observations(config)
            print(f"[qLogEHVI] campaign complete: {target_count}/{config.total_budget}")
            return 0

        remaining = config.total_budget - target_count
        q = min(config.proposal.q, remaining)
        observations = refresh_observations(config)
        batch = create_batch(
            config,
            state,
            observations,
            input_space,
            q=q,
        )
        print(
            f"[qLogEHVI] proposed batch={batch['batch_index']} "
            f"cases={','.join(batch['case_ids'])} "
            f"preflight_invalid={len(batch['invalid_preflight_case_ids'])} "
            f"gp_fit={batch['proposal_diagnostics']['gp_fit_seconds']:.1f}s "
            f"acquisition={batch['proposal_diagnostics']['acquisition_seconds']:.1f}s",
            flush=True,
        )
        if stop_after_proposal:
            print("[qLogEHVI] stop-after-proposal requested; active batch is resumable")
            return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        type=Path,
        help="historical result root; repeat for multiple sources",
    )
    parser.add_argument("--output", type=Path, default=F5_OUTPUT_DIRECTORY)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--band", nargs=2, type=float, default=None)
    parser.add_argument("--device", action="append", dest="device_ids")
    parser.add_argument("--sampling-config", type=Path, default=None)
    parser.add_argument("--devices-config", type=Path, default=None)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--raw-samples", type=int, default=None)
    parser.add_argument("--num-restarts", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--optimization-batch-limit", type=int, default=None)
    parser.add_argument("--optimization-maxiter", type=int, default=None)
    parser.add_argument("--gp-training-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
            raise ValueError(f"invalid existing optimization plan: {plan_path}")
        existing = loaded
    simulation = existing.get("simulation", {})
    if not isinstance(simulation, Mapping):
        simulation = {}
    proposal_defaults = existing.get("proposal_settings", {})
    if not isinstance(proposal_defaults, Mapping):
        proposal_defaults = {}

    sources = (
        tuple(args.sources)
        if args.sources
        else tuple(Path(value) for value in existing.get("source_directories", ()))
        or F5_SOURCE_DIRECTORIES
    )
    device_ids = (
        tuple(args.device_ids)
        if args.device_ids
        else tuple(str(value) for value in simulation.get("device_ids", ()))
        or F5_DEVICE_IDS
    )
    band_values = args.band if args.band is not None else existing.get("band_ghz", F5_BAND_GHZ)

    def proposal_value(argument: Any, name: str, fallback: Any) -> Any:
        return argument if argument is not None else proposal_defaults.get(name, fallback)

    return CampaignConfig(
        plan_id=str(args.plan_id or existing.get("plan_id", F5_PLAN_ID)),
        source_directories=tuple(Path(path) for path in sources),
        output_directory=output,
        total_budget=int(
            args.budget
            if args.budget is not None
            else existing.get("total_budget", F5_TOTAL_BUDGET)
        ),
        band_ghz=(float(band_values[0]), float(band_values[1])),
        device_ids=device_ids,
        sampling_config=Path(
            args.sampling_config
            or existing.get("sampling_config", SAMPLING_CONFIG)
        ),
        device_config=Path(
            args.devices_config
            or simulation.get("device_config", DEVICE_CONFIG)
        ),
        project_template=Path(
            args.project
            or simulation.get("project_template", PROJECT_TEMPLATE)
        ),
        proposal=qlogehvi.ProposalSettings(
            q=int(
                args.q
                if args.q is not None
                else existing.get("q", proposal_defaults.get("q", F5_Q))
            ),
            seed=int(proposal_value(args.seed, "seed", 20260807)),
            raw_samples=int(proposal_value(args.raw_samples, "raw_samples", 256)),
            num_restarts=int(proposal_value(args.num_restarts, "num_restarts", 8)),
            mc_samples=int(proposal_value(args.mc_samples, "mc_samples", 64)),
            optimization_batch_limit=int(
                proposal_value(
                    args.optimization_batch_limit,
                    "optimization_batch_limit",
                    2,
                )
            ),
            optimization_maxiter=int(
                proposal_value(args.optimization_maxiter, "optimization_maxiter", 100)
            ),
            gp_training_steps=int(
                proposal_value(args.gp_training_steps, "gp_training_steps", 50)
            ),
        ),
        coordinate_quantum_mm=float(
            simulation.get("coordinate_quantum_mm", 0.01)
        ),
        allow_disconnected_conductor=bool(
            simulation.get("allow_disconnected_conductor", False)
        ),
        max_attempts=int(simulation.get("max_attempts", 3)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _config_from_args(args)
    if not args.prepare_only and not args.stop_after_proposal:
        require_confirmation = F5_REQUIRE_CONFIRMATION and not args.yes
        if require_confirmation:
            answer = input(
                f"Type RUN to start/resume qLogEHVI plan {config.plan_id} "
                f"({config.total_budget} target evaluations): "
            )
            if answer.strip() != "RUN":
                print("Cancelled; no solver was started.")
                return 1
    try:
        return run_campaign(
            config,
            prepare_only=args.prepare_only,
            stop_after_proposal=args.stop_after_proposal,
        )
    except KeyboardInterrupt:
        print("[qLogEHVI] interrupted; plan and active batch remain resumable")
        return 130
    except Exception as exc:
        print(f"qLogEHVI error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
