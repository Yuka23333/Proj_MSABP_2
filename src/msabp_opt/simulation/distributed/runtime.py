"""Princess run preparation, Maid deployment, and lifecycle supervision.

SSH/SCP are used only for short deployment and wake-up operations.  Once
started, each Maid is a device-local process: it opens the device-local CST
project with ``cst.interface`` and talks back to Princess over HTTP.
"""

from __future__ import annotations

import csv
import json
import math
import re
import secrets
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .config import DeviceConfig, DeviceRegistry, LaunchMode
from .http_api import PrincessHttpServer
from .maid import RUNTIME_SCHEMA_VERSION, write_runtime_config
from .princess import PrincessCoordinator
from .state import FrozenCsv, PrincessState, UnknownWorkerError, freeze_csv
from .transport import (
    DoctorReport,
    LaunchReceipt,
    doctor_device,
    launch_maid,
    push_file_atomic,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUN_SCHEMA_VERSION = 1
DEFAULT_LEASE_SECONDS = 90.0
DEFAULT_MONITOR_SECONDS = 1.0
DEFAULT_TERMINAL_GRACE_SECONDS = 5.0
DEFAULT_RESTART_DELAY_SECONDS = 2.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_STALE_IDLE_SECONDS = 30.0
DEFAULT_ARTIFACT_TIMEOUT_SECONDS = 600.0
DEFAULT_ARTIFACT_COMMIT_DEADLINE_SECONDS = 1800.0
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class PrincessRuntimeError(RuntimeError):
    """Raised when a run cannot be safely prepared, deployed, or resumed."""


@dataclass(frozen=True)
class ExcludedCsvRow:
    case_id: str
    source_row_index: int
    csv_line: int
    reason: str


@dataclass(frozen=True)
class PreparedWorklist:
    source: FrozenCsv
    worklist: FrozenCsv
    excluded_rows: tuple[ExcludedCsvRow, ...]


@dataclass(frozen=True)
class PrincessRunPaths:
    run_id: str
    run_root: Path
    source_csv: Path
    worklist_csv: Path
    exclusions_json: Path
    database: Path
    runtime_json: Path
    deployment_root: Path
    incoming_root: Path
    upload_root: Path
    results_root: Path

    @classmethod
    def for_run(
        cls,
        run_id: str,
        *,
        repository_root: str | Path = REPOSITORY_ROOT,
        results_root: str | Path | None = None,
    ) -> "PrincessRunPaths":
        resolved_id = validate_run_id(run_id)
        root = Path(repository_root).expanduser().resolve()
        run_root = root / "simulations" / "runs" / resolved_id
        raw_results = (
            Path(results_root).expanduser().resolve()
            if results_root is not None
            else root / "results" / "raw" / resolved_id
        )
        return cls(
            run_id=resolved_id,
            run_root=run_root,
            source_csv=run_root / "source.csv",
            worklist_csv=run_root / "worklist.csv",
            exclusions_json=run_root / "excluded_rows.json",
            database=run_root / "princess.sqlite3",
            runtime_json=run_root / "princess_runtime.json",
            deployment_root=run_root / "deployment",
            incoming_root=run_root / "incoming",
            upload_root=run_root / "incoming" / "upload",
            results_root=raw_results,
        )


@dataclass(frozen=True)
class DeviceRunPaths:
    worker_root: str
    csv_path: str
    project_path: str
    output_root: str
    runtime_config_path: str


@dataclass(frozen=True)
class DeviceDeployment:
    device: DeviceConfig
    paths: DeviceRunPaths
    runtime_staging_path: Path
    doctor: DoctorReport
    launch: LaunchReceipt


@dataclass(frozen=True)
class PendingLaunch:
    started_at: float
    deadline: float
    pid: int | None


@dataclass(frozen=True)
class RunPreparation:
    paths: PrincessRunPaths
    worklist: PreparedWorklist
    api_token: str
    created: bool


def validate_run_id(value: str) -> str:
    resolved = str(value).strip()
    if not _SAFE_ID.fullmatch(resolved) or resolved in {".", ".."}:
        raise ValueError(f"unsafe run id: {resolved!r}")
    return resolved


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _parse_csv_bool(value: Any, *, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{label} must be a boolean, got {value!r}")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_or_same(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise PrincessRuntimeError(
                f"existing {label} differs for this run: {path}"
            )
        return
    _atomic_write_bytes(path, payload)


def prepare_worklist(
    source_csv: str | Path,
    paths: PrincessRunPaths,
) -> PreparedWorklist:
    """Freeze the source CSV and exclude sampler-rejected geometry up front."""

    frozen_source = freeze_csv(source_csv, paths.source_csv)
    with frozen_source.path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("sampling CSV has no header")
        fieldnames = tuple(reader.fieldnames)
        accepted: list[dict[str, str]] = []
        excluded: list[ExcludedCsvRow] = []
        for source_row_index, row in enumerate(reader):
            case_id = str(row.get("sample_id", source_row_index)).strip()
            if not _SAFE_ID.fullmatch(case_id) or case_id in {".", ".."}:
                raise ValueError(
                    f"sampling CSV line {source_row_index + 2} has unsafe sample_id "
                    f"{case_id!r}"
                )
            geometry_valid = True
            if "geometry_valid" in fieldnames:
                geometry_valid = _parse_csv_bool(
                    row.get("geometry_valid", ""),
                    label=f"geometry_valid on CSV line {source_row_index + 2}",
                )
            if geometry_valid:
                accepted.append(dict(row))
                continue
            excluded.append(
                ExcludedCsvRow(
                    case_id=case_id,
                    source_row_index=source_row_index,
                    csv_line=source_row_index + 2,
                    reason=str(row.get("geometry_error", "")).strip()
                    or "sampler marked geometry invalid",
                )
            )

    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(accepted)
    _write_immutable_or_same(
        paths.worklist_csv,
        buffer.getvalue().encode("utf-8"),
        label="filtered worklist",
    )
    exclusions_payload = {
        "schema_version": 1,
        "source_csv_sha256": frozen_source.sha256,
        "excluded_count": len(excluded),
        "excluded_rows": [asdict(item) for item in excluded],
    }
    exclusions_bytes = (
        json.dumps(
            exclusions_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_immutable_or_same(
        paths.exclusions_json,
        exclusions_bytes,
        label="excluded-row manifest",
    )
    frozen_worklist = freeze_csv(paths.worklist_csv, paths.worklist_csv)
    return PreparedWorklist(
        source=frozen_source,
        worklist=frozen_worklist,
        excluded_rows=tuple(excluded),
    )


def select_devices(
    registry: DeviceRegistry,
    device_ids: Sequence[str] | None = None,
) -> tuple[DeviceConfig, ...]:
    """Select enabled devices, or explicitly opt disabled devices into one run."""

    if device_ids:
        unique_ids = tuple(dict.fromkeys(str(item).strip() for item in device_ids))
        devices = tuple(registry.get_device(device_id) for device_id in unique_ids)
        return tuple(
            device if device.enabled else replace(device, enabled=True)
            for device in devices
        )
    devices = registry.enabled_devices
    if not devices:
        raise PrincessRuntimeError(
            "no Maid devices are enabled; enable them in princess_devices.json "
            "or pass explicit --device values"
        )
    return devices


def device_run_paths(
    device: DeviceConfig,
    run_id: str,
    *,
    launch_generation: str | None = None,
) -> DeviceRunPaths:
    run_id = validate_run_id(run_id)
    if launch_generation is not None:
        launch_generation = validate_run_id(launch_generation)
    worker_root = (
        PureWindowsPath(device.repo_root)
        / "simulations"
        / "runs"
        / run_id
        / "workers"
        / device.id
    )
    launch_root = (
        worker_root
        if launch_generation is None
        else worker_root / "launches" / launch_generation
    )
    return DeviceRunPaths(
        worker_root=str(worker_root),
        csv_path=str(worker_root / "worklist.csv"),
        project_path=str(launch_root / "model" / "msa-bp.cst"),
        output_root=str(launch_root / "output"),
        runtime_config_path=str(launch_root / "maid_runtime.json"),
    )


def _load_or_create_run_secret(
    paths: PrincessRunPaths,
    *,
    worklist: PreparedWorklist,
    registry: DeviceRegistry,
    device_ids: Sequence[str],
) -> str:
    if paths.runtime_json.is_file():
        payload = json.loads(paths.runtime_json.read_text(encoding="utf-8"))
        if payload.get("schema_version") != RUN_SCHEMA_VERSION:
            raise PrincessRuntimeError("unsupported existing Princess runtime schema")
        expected = {
            "run_id": paths.run_id,
            "source_csv_sha256": worklist.source.sha256,
            "worklist_csv_sha256": worklist.worklist.sha256,
            "advertise_url": registry.advertise_url,
        }
        mismatches = [
            name for name, value in expected.items() if payload.get(name) != value
        ]
        if mismatches:
            raise PrincessRuntimeError(
                "existing Princess runtime does not match: " + ", ".join(mismatches)
            )
        token = payload.get("api_token")
        if not isinstance(token, str) or len(token) < 32:
            raise PrincessRuntimeError("existing Princess runtime has no valid API token")
        merged_devices = sorted(
            set(str(item) for item in payload.get("device_ids", [])) | set(device_ids)
        )
        if merged_devices != payload.get("device_ids"):
            payload["device_ids"] = merged_devices
            _atomic_write_bytes(
                paths.runtime_json,
                (
                    json.dumps(payload, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8"),
            )
        return token

    token = secrets.token_urlsafe(48)
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": paths.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv_sha256": worklist.source.sha256,
        "worklist_csv_sha256": worklist.worklist.sha256,
        "advertise_url": registry.advertise_url,
        "bind_host": registry.bind_host,
        "port": registry.port,
        "device_ids": sorted(set(device_ids)),
        "api_token": token,
    }
    _atomic_write_bytes(
        paths.runtime_json,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return token


def prepare_run(
    *,
    source_csv: str | Path,
    run_id: str,
    registry: DeviceRegistry,
    devices: Sequence[DeviceConfig],
    repository_root: str | Path = REPOSITORY_ROOT,
    results_root: str | Path | None = None,
) -> RunPreparation:
    paths = PrincessRunPaths.for_run(
        run_id,
        repository_root=repository_root,
        results_root=results_root,
    )
    paths.run_root.mkdir(parents=True, exist_ok=True)
    worklist = prepare_worklist(source_csv, paths)
    with PrincessState(paths.database) as state:
        created = state.initialize_run(
            paths.run_id,
            worklist.worklist,
            metadata={
                "source_csv_sha256": worklist.source.sha256,
                "excluded_geometry_count": len(worklist.excluded_rows),
            },
        )
    token = _load_or_create_run_secret(
        paths,
        worklist=worklist,
        registry=registry,
        device_ids=[device.id for device in devices],
    )
    return RunPreparation(
        paths=paths,
        worklist=worklist,
        api_token=token,
        created=created,
    )


def _maid_runtime_payload(
    *,
    device: DeviceConfig,
    paths: DeviceRunPaths,
    preparation: RunPreparation,
    registry: DeviceRegistry,
    dry_run: bool,
    coordinate_quantum_mm: float,
    allow_disconnected_conductor: bool,
    command_timeout_seconds: float,
    heartbeat_seconds: float,
    poll_seconds: float,
    max_consecutive_errors: int,
    save_project_after_case: bool,
    hello_timeout_seconds: float,
    artifact_timeout_seconds: float,
    artifact_commit_deadline_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "run_id": preparation.paths.run_id,
        "worker_id": device.id,
        "princess_url": registry.advertise_url,
        "api_token": preparation.api_token,
        "csv_path": paths.csv_path,
        "csv_sha256": preparation.worklist.worklist.sha256,
        "project_path": paths.project_path,
        "output_root": paths.output_root,
        "dry_run": dry_run,
        "coordinate_quantum_mm": coordinate_quantum_mm,
        "allow_disconnected_conductor": allow_disconnected_conductor,
        "command_timeout_seconds": command_timeout_seconds,
        "heartbeat_seconds": heartbeat_seconds,
        "poll_seconds": poll_seconds,
        "max_consecutive_errors": max_consecutive_errors,
        "save_project_after_case": save_project_after_case,
        "hello_timeout_seconds": hello_timeout_seconds,
        "artifact_timeout_seconds": artifact_timeout_seconds,
        "artifact_commit_deadline_seconds": artifact_commit_deadline_seconds,
    }


class PrincessRuntime:
    """Own one Princess HTTP server and its selected Maid processes."""

    def __init__(
        self,
        *,
        preparation: RunPreparation,
        registry: DeviceRegistry,
        devices: Sequence[DeviceConfig],
        project_template: str | Path,
        dry_run: bool = False,
        coordinate_quantum_mm: float = 0.01,
        allow_disconnected_conductor: bool = False,
        command_timeout_seconds: float = 15.0,
        heartbeat_seconds: float = 15.0,
        poll_seconds: float = 5.0,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        max_consecutive_errors: int = 5,
        max_attempts: int = 3,
        save_project_after_case: bool = False,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        stale_idle_seconds: float = DEFAULT_STALE_IDLE_SECONDS,
        artifact_timeout_seconds: float = DEFAULT_ARTIFACT_TIMEOUT_SECONDS,
        artifact_commit_deadline_seconds: float = (
            DEFAULT_ARTIFACT_COMMIT_DEADLINE_SECONDS
        ),
        max_recovery_launch_attempts: int = 3,
        push_file: Callable[..., Any] = push_file_atomic,
        doctor: Callable[..., DoctorReport] = doctor_device,
        launcher: Callable[..., LaunchReceipt] = launch_maid,
        server_factory: Callable[..., PrincessHttpServer] = PrincessHttpServer,
    ) -> None:
        self.preparation = preparation
        self.registry = registry
        self.devices = tuple(devices)
        self.project_template = Path(project_template).expanduser().resolve()
        self.dry_run = bool(dry_run)
        self.coordinate_quantum_mm = self._positive_finite(
            coordinate_quantum_mm,
            "coordinate_quantum_mm",
        )
        self.allow_disconnected_conductor = bool(allow_disconnected_conductor)
        self.command_timeout_seconds = self._positive_finite(
            command_timeout_seconds,
            "command_timeout_seconds",
        )
        self.heartbeat_seconds = self._positive_finite(
            heartbeat_seconds,
            "heartbeat_seconds",
        )
        self.poll_seconds = self._positive_finite(poll_seconds, "poll_seconds")
        self.lease_seconds = self._positive_finite(lease_seconds, "lease_seconds")
        self.startup_timeout_seconds = self._positive_finite(
            startup_timeout_seconds,
            "startup_timeout_seconds",
        )
        self.stale_idle_seconds = self._positive_finite(
            stale_idle_seconds,
            "stale_idle_seconds",
        )
        self.artifact_timeout_seconds = self._positive_finite(
            artifact_timeout_seconds,
            "artifact_timeout_seconds",
        )
        self.artifact_commit_deadline_seconds = self._positive_finite(
            artifact_commit_deadline_seconds,
            "artifact_commit_deadline_seconds",
        )
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("heartbeat_seconds must be shorter than lease_seconds")
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.max_attempts = int(max_attempts)
        self.max_recovery_launch_attempts = int(max_recovery_launch_attempts)
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be at least 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.max_recovery_launch_attempts < 1:
            raise ValueError("max_recovery_launch_attempts must be at least 1")
        self.save_project_after_case = bool(save_project_after_case)
        self._push_file = push_file
        self._doctor = doctor
        self._launcher = launcher
        self._server_factory = server_factory
        self._state: PrincessState | None = None
        self._coordinator: PrincessCoordinator | None = None
        self._server: PrincessHttpServer | None = None
        self._server_thread: threading.Thread | None = None
        self._server_started_at: float | None = None
        self._launch_pending: dict[str, PendingLaunch] = {}
        self._restart_due: dict[str, tuple[int, float]] = {}
        self._recovery_due: dict[str, float] = {}
        self._launch_failures: dict[str, int] = {}
        self._disabled_for_run: set[str] = set()

    @staticmethod
    def _positive_finite(value: float, label: str) -> float:
        resolved = float(value)
        if not math.isfinite(resolved) or resolved <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
        return resolved

    @property
    def state(self) -> PrincessState:
        if self._state is None:
            raise PrincessRuntimeError("Princess runtime is not started")
        return self._state

    def start_server(self) -> None:
        if self._server is not None:
            return
        paths = self.preparation.paths
        self._state = PrincessState(paths.database)
        self._coordinator = PrincessCoordinator(
            self._state,
            run_id=paths.run_id,
            csv_sha256=self.preparation.worklist.worklist.sha256,
            results_dir=paths.results_root,
            incoming_dir=paths.incoming_root,
            lease_seconds=self.lease_seconds,
            max_attempts=self.max_attempts,
            streak_threshold=self.max_consecutive_errors,
        )
        self._server = self._server_factory(
            (self.registry.bind_host, self.registry.port),
            token=self.preparation.api_token,
            upload_dir=paths.upload_root,
            message_handler=self._coordinator.handle_message,
            artifact_handler=self._coordinator.handle_artifact,
        )
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"PrincessHttp-{paths.run_id}",
            daemon=True,
        )
        self._server_started_at = time.time()
        self._server_thread.start()

    def _stage_runtime(self, device: DeviceConfig, paths: DeviceRunPaths) -> Path:
        payload = _maid_runtime_payload(
            device=device,
            paths=paths,
            preparation=self.preparation,
            registry=self.registry,
            dry_run=self.dry_run,
            coordinate_quantum_mm=self.coordinate_quantum_mm,
            allow_disconnected_conductor=self.allow_disconnected_conductor,
            command_timeout_seconds=self.command_timeout_seconds,
            heartbeat_seconds=self.heartbeat_seconds,
            poll_seconds=self.poll_seconds,
            max_consecutive_errors=self.max_consecutive_errors,
            save_project_after_case=self.save_project_after_case,
            hello_timeout_seconds=self.startup_timeout_seconds * 0.8,
            artifact_timeout_seconds=self.artifact_timeout_seconds,
            artifact_commit_deadline_seconds=(
                self.artifact_commit_deadline_seconds
            ),
        )
        destination = (
            self.preparation.paths.deployment_root / device.id / "maid_runtime.json"
        )
        return write_runtime_config(destination, payload)

    def _worker_is_known(self, device_id: str) -> bool:
        try:
            self.state.get_worker(self.preparation.paths.run_id, device_id)
        except UnknownWorkerError:
            return False
        return True

    def _fresh_launch_generation(self, reason: str) -> str:
        return validate_run_id(
            f"{reason}-{int(time.time() * 1000)}-{secrets.token_hex(3)}"
        )

    def deploy_device(
        self,
        device: DeviceConfig,
        *,
        launch_generation: str | None = None,
        force_project: bool = False,
    ) -> tuple[DeviceRunPaths, Path]:
        paths = device_run_paths(
            device,
            self.preparation.paths.run_id,
            launch_generation=launch_generation,
        )
        runtime_staging = self._stage_runtime(device, paths)
        self._push_file(
            device,
            self.preparation.paths.worklist_csv,
            paths.csv_path,
            overwrite=True,
        )
        needs_project = force_project or not self._worker_is_known(device.id)
        if not self.dry_run and needs_project:
            if not self.project_template.is_file():
                raise FileNotFoundError(
                    f"CST project template does not exist: {self.project_template}"
                )
            self._push_file(
                device,
                self.project_template,
                paths.project_path,
                overwrite=True,
            )
        # Runtime JSON is committed last: Maid never observes a half deployment.
        self._push_file(
            device,
            runtime_staging,
            paths.runtime_config_path,
            overwrite=True,
        )
        return paths, runtime_staging

    def deploy_and_launch(
        self,
        device: DeviceConfig,
        *,
        launch_generation: str | None = None,
        force_project: bool = False,
    ) -> DeviceDeployment:
        paths, runtime_staging = self.deploy_device(
            device,
            launch_generation=launch_generation,
            force_project=force_project,
        )
        launch_device = replace(
            device,
            runtime_config_path=paths.runtime_config_path,
        )
        report = self._doctor(launch_device)
        # doctor_device checks the repository template path.  The Maid actually
        # opens the separately verified per-launch copy just pushed above.
        ignored_missing = {"cst_project"}
        if self.dry_run:
            ignored_missing.add("cst.interface")
        missing = [
            item for item in report.missing_requirements if item not in ignored_missing
        ]
        version_errors: tuple[str, ...] = ()
        if report.python_version is not None:
            version_text = report.python_version.strip()
            if not re.match(r"^3\.11(?:\.|$)", version_text):
                version_errors = (
                    f"Python {version_text} is unsupported; CST 2025 Maid requires 3.11",
                )
        if report.errors or missing or version_errors:
            detail = "; ".join(
                (*report.errors, *version_errors, *(f"missing {x}" for x in missing))
            )
            raise PrincessRuntimeError(
                f"Maid doctor failed for {device.id}: {detail or 'unknown error'}"
            )
        if launch_device.launch_mode is LaunchMode.BELL:
            receipt = self._launcher(
                launch_device,
                bell_token=self.preparation.api_token,
            )
        else:
            receipt = self._launcher(launch_device)
        return DeviceDeployment(
            device=launch_device,
            paths=paths,
            runtime_staging_path=runtime_staging,
            doctor=report,
            launch=receipt,
        )

    @staticmethod
    def _worker_metadata(worker: Mapping[str, Any]) -> Mapping[str, Any]:
        decoded_metadata = worker.get("metadata")
        if isinstance(decoded_metadata, Mapping):
            return decoded_metadata
        raw = worker.get("metadata_json", "{}")
        if isinstance(raw, Mapping):
            return raw
        try:
            decoded = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, Mapping) else {}

    def _worker_refreshed_after(
        self,
        device_id: str,
        cutoff: float,
        *,
        expected_pid: int | None = None,
    ) -> bool:
        try:
            worker = self.state.get_worker(self.preparation.paths.run_id, device_id)
        except UnknownWorkerError:
            return False
        try:
            last_seen = float(worker["last_seen"])
        except (KeyError, TypeError, ValueError):
            return False
        if last_seen < cutoff:
            return False
        if expected_pid is None:
            return True
        raw_pid = self._worker_metadata(worker).get("pid")
        return not isinstance(raw_pid, bool) and raw_pid == expected_pid

    def _wait_for_worker_refreshes(
        self,
        expected: Mapping[str, tuple[float, int | None]],
        timeout_seconds: float,
    ) -> set[str]:
        remaining = dict(expected)
        confirmed: set[str] = set()
        deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
        while remaining:
            for device_id, (cutoff, expected_pid) in tuple(remaining.items()):
                if self._worker_refreshed_after(
                    device_id,
                    cutoff,
                    expected_pid=expected_pid,
                ):
                    confirmed.add(device_id)
                    remaining.pop(device_id)
            seconds_left = deadline - time.monotonic()
            if not remaining or seconds_left <= 0.0:
                break
            time.sleep(min(0.1, seconds_left))
        return confirmed

    def _record_pending_launch(
        self,
        device_id: str,
        *,
        started_at: float,
        receipt: LaunchReceipt,
    ) -> None:
        self._launch_pending[device_id] = PendingLaunch(
            started_at=started_at,
            deadline=time.time() + self.startup_timeout_seconds,
            pid=receipt.pid,
        )

    def _mark_launch_confirmed(self, device_id: str) -> None:
        self._launch_pending.pop(device_id, None)
        self._recovery_due.pop(device_id, None)
        self._launch_failures.pop(device_id, None)
        self._disabled_for_run.discard(device_id)

    def _mark_launch_failure(self, device_id: str) -> bool:
        failures = self._launch_failures.get(device_id, 0) + 1
        self._launch_failures[device_id] = failures
        exhausted = failures >= self.max_recovery_launch_attempts
        if exhausted:
            self._disabled_for_run.add(device_id)
            self._launch_pending.pop(device_id, None)
            self._recovery_due.pop(device_id, None)
            self._restart_due.pop(device_id, None)
        return exhausted

    def start_workers(
        self,
        *,
        resume_grace_seconds: float | None = None,
    ) -> tuple[DeviceDeployment, ...]:
        if self._server is None:
            raise PrincessRuntimeError("start_server() must run before Maid launch")
        progress = self.state.progress(self.preparation.paths.run_id)
        if progress.is_terminal:
            return ()

        deployments: list[DeviceDeployment] = []
        errors: list[str] = []
        live_workers: set[str] = set()
        if not self.preparation.created:
            known_devices = {
                device.id: (float(self._server_started_at or time.time()), None)
                for device in self.devices
                if self._worker_is_known(device.id)
            }
            grace = (
                max(self.heartbeat_seconds, self.poll_seconds) + 1.0
                if resume_grace_seconds is None
                else max(float(resume_grace_seconds), 0.0)
            )
            live_workers = self._wait_for_worker_refreshes(known_devices, grace)
            for device_id in sorted(live_workers):
                print(
                    f"[Princess] Maid {device_id} reconnected; duplicate launch skipped",
                    flush=True,
                )

        launch_cutoffs: dict[str, tuple[float, int | None]] = {}
        uncertain_launch_cutoffs: dict[str, tuple[float, int | None]] = {}
        uncertain_launch_errors: dict[str, str] = {}
        for device in self.devices:
            if device.id in live_workers:
                continue
            started_at: float | None = None
            try:
                known = self._worker_is_known(device.id)
                if known:
                    worker = self.state.get_worker(
                        self.preparation.paths.run_id,
                        device.id,
                    )
                    status = str(worker["status"])
                    if status in {"unhealthy", "quarantined", "stopped"}:
                        raise PrincessRuntimeError(
                            f"worker state is {status}; explicit intervention is required"
                        )
                    if status == "offline":
                        self.state.wake_worker(
                            self.preparation.paths.run_id,
                            device.id,
                        )
                generation = (
                    self._fresh_launch_generation("resume") if known else None
                )
                started_at = time.time()
                deployment = self.deploy_and_launch(
                    device,
                    launch_generation=generation,
                    force_project=known,
                )
                deployments.append(deployment)
                self._record_pending_launch(
                    device.id,
                    started_at=started_at,
                    receipt=deployment.launch,
                )
                launch_cutoffs[device.id] = (started_at, deployment.launch.pid)
            except Exception as exc:
                detail = f"{device.id}: {type(exc).__name__}: {exc}"
                if started_at is None:
                    self._mark_launch_failure(device.id)
                    errors.append(detail)
                else:
                    # The remote launcher can time out after Start-Process has
                    # already created the Maid.  Without a trustworthy receipt
                    # there is no PID to compare, so hello/last_seen after this
                    # launch round's cutoff is the authoritative confirmation.
                    uncertain_launch_cutoffs[device.id] = (started_at, None)
                    uncertain_launch_errors[device.id] = detail

        refresh_expectations = dict(launch_cutoffs)
        refresh_expectations.update(uncertain_launch_cutoffs)
        confirmed = self._wait_for_worker_refreshes(
            refresh_expectations,
            self.startup_timeout_seconds,
        )
        confirmed_launches = set(confirmed).intersection(launch_cutoffs)
        for device_id, (cutoff, _expected_pid) in launch_cutoffs.items():
            if device_id not in confirmed and self._worker_refreshed_after(
                device_id,
                cutoff,
            ):
                confirmed.add(device_id)
                print(
                    f"[Princess] Maid {device_id} refreshed from an existing PID; "
                    "replacement launch is not required",
                    flush=True,
                )
        for device_id in sorted(set(uncertain_launch_cutoffs).intersection(confirmed)):
            print(
                f"[Princess] Maid {device_id} registered after an inconclusive "
                "launcher/SSH receipt; hello accepted as authoritative",
                flush=True,
            )
        live_workers.update(confirmed)
        now = time.time()
        for device_id in confirmed:
            self._mark_launch_confirmed(device_id)
        for device_id in set(launch_cutoffs) - confirmed:
            self._launch_pending.pop(device_id, None)
            exhausted = self._mark_launch_failure(device_id)
            if not exhausted:
                self._recovery_due[device_id] = now + DEFAULT_RESTART_DELAY_SECONDS
            errors.append(
                f"{device_id}: Maid process did not register before the "
                f"{self.startup_timeout_seconds:g}s startup deadline"
                + ("; recovery budget exhausted" if exhausted else "")
            )
        for device_id in set(uncertain_launch_cutoffs) - confirmed:
            exhausted = self._mark_launch_failure(device_id)
            detail = uncertain_launch_errors[device_id]
            errors.append(
                detail
                + "; no Maid hello arrived before the startup deadline"
                + ("; recovery budget exhausted" if exhausted else "")
            )

        if not live_workers:
            raise PrincessRuntimeError(
                "no Maid registered with Princess: " + " | ".join(errors)
            )
        for message in errors:
            print(f"[Princess] Maid skipped: {message}", flush=True)
        return tuple(
            deployment
            for deployment in deployments
            if deployment.device.id in confirmed_launches
        )

    def _refresh_pending_launches(self, now: float) -> None:
        run_id = self.preparation.paths.run_id
        for device_id, pending in tuple(self._launch_pending.items()):
            if self._worker_refreshed_after(
                device_id,
                pending.started_at,
                expected_pid=pending.pid,
            ):
                self._mark_launch_confirmed(device_id)
                continue
            if self._worker_refreshed_after(device_id, pending.started_at):
                self._mark_launch_confirmed(device_id)
                print(
                    f"[Princess] Maid {device_id} recovered through an existing PID",
                    flush=True,
                )
                continue
            if now < pending.deadline:
                continue
            self._launch_pending.pop(device_id, None)
            if self._mark_launch_failure(device_id):
                print(
                    f"[Princess] Maid {device_id} exhausted its launch/hello "
                    "recovery budget",
                    flush=True,
                )
                continue
            try:
                worker = self.state.get_worker(run_id, device_id)
            except UnknownWorkerError:
                worker = None
            if worker is not None and worker["status"] == "restarting":
                restart_count = int(worker["restart_count"])
                self._restart_due[device_id] = (restart_count, now)
            else:
                self._recovery_due[device_id] = now

    def _schedule_restarts(self, now: float) -> None:
        run_id = self.preparation.paths.run_id
        for device in self.devices:
            if device.id in self._disabled_for_run:
                try:
                    late_worker = self.state.get_worker(run_id, device.id)
                except UnknownWorkerError:
                    continue
                if (
                    str(late_worker["status"]) in {"idle", "busy"}
                    and now - float(late_worker["last_seen"])
                    < max(self.poll_seconds * 2.0, self.heartbeat_seconds * 2.0)
                ):
                    self._mark_launch_confirmed(device.id)
                else:
                    continue
            try:
                worker = self.state.get_worker(run_id, device.id)
            except UnknownWorkerError:
                continue
            if worker["status"] != "restarting":
                self._restart_due.pop(device.id, None)
                continue
            if device.id in self._launch_pending:
                continue
            restart_count = int(worker["restart_count"])
            current = self._restart_due.get(device.id)
            if current is None or current[0] != restart_count:
                self._restart_due[device.id] = (
                    restart_count,
                    now + DEFAULT_RESTART_DELAY_SECONDS,
                )
                continue
            if now < current[1]:
                continue
            started_at = time.time()
            try:
                deployment = self.deploy_and_launch(
                    device,
                    launch_generation=self._fresh_launch_generation(
                        f"restart-{restart_count}"
                    ),
                    force_project=True,
                )
            except Exception as exc:
                exhausted = self._mark_launch_failure(device.id)
                if not exhausted:
                    self._restart_due[device.id] = (
                        restart_count,
                        now + DEFAULT_RESTART_DELAY_SECONDS,
                    )
                print(
                    f"[Princess] restart of Maid {device.id} failed; will retry: "
                    f"{type(exc).__name__}: {exc}"
                    + (" (budget exhausted)" if exhausted else ""),
                    flush=True,
                )
                continue
            self._record_pending_launch(
                device.id,
                started_at=started_at,
                receipt=deployment.launch,
            )
            self._restart_due.pop(device.id, None)
            print(
                f"[Princess] restarted Maid {device.id}, pid={deployment.launch.pid}",
                flush=True,
            )

    def _schedule_recoveries(
        self,
        now: float,
        progress: Mapping[str, int | bool],
    ) -> None:
        if int(progress["pending"]) <= 0:
            return
        run_id = self.preparation.paths.run_id
        for device in self.devices:
            if device.id in self._disabled_for_run:
                try:
                    late_worker = self.state.get_worker(run_id, device.id)
                except UnknownWorkerError:
                    continue
                if (
                    str(late_worker["status"]) in {"idle", "busy"}
                    and now - float(late_worker["last_seen"])
                    < max(self.poll_seconds * 2.0, self.heartbeat_seconds * 2.0)
                ):
                    self._mark_launch_confirmed(device.id)
                else:
                    continue
            if device.id in self._launch_pending:
                continue
            try:
                worker = self.state.get_worker(run_id, device.id)
            except UnknownWorkerError:
                worker = None
            status = "unknown" if worker is None else str(worker["status"])
            if status == "restarting":
                continue
            if status in {"unhealthy", "quarantined", "stopped", "busy"}:
                self._recovery_due.pop(device.id, None)
                continue
            if status == "idle" and now - float(worker["last_seen"]) < max(
                self.poll_seconds * 2.0,
                1.0,
            ):
                self._mark_launch_confirmed(device.id)
                continue
            stale_idle = (
                status == "idle"
                and now - float(worker["last_seen"]) >= self.stale_idle_seconds
            )
            retry_scheduled = device.id in self._recovery_due
            needs_recovery = (
                status in {"unknown", "offline"} or stale_idle or retry_scheduled
            )
            if not needs_recovery:
                self._recovery_due.pop(device.id, None)
                continue
            due = self._recovery_due.setdefault(
                device.id,
                now + DEFAULT_RESTART_DELAY_SECONDS,
            )
            if now < due:
                continue
            try:
                if status == "offline":
                    self.state.wake_worker(run_id, device.id)
                started_at = time.time()
                deployment = self.deploy_and_launch(
                    device,
                    launch_generation=self._fresh_launch_generation("recovery"),
                    force_project=True,
                )
                self._record_pending_launch(
                    device.id,
                    started_at=started_at,
                    receipt=deployment.launch,
                )
                self._recovery_due.pop(device.id, None)
                print(
                    f"[Princess] recovery Maid {device.id} launched, "
                    f"pid={deployment.launch.pid}",
                    flush=True,
                )
            except Exception as exc:
                exhausted = self._mark_launch_failure(device.id)
                if not exhausted:
                    self._recovery_due[device.id] = (
                        now + max(DEFAULT_RESTART_DELAY_SECONDS, self.poll_seconds)
                    )
                print(
                    f"[Princess] recovery launch for Maid {device.id} failed; "
                    f"will retry: {type(exc).__name__}: {exc}"
                    + (" (budget exhausted)" if exhausted else ""),
                    flush=True,
                )

    def monitor(
        self,
        *,
        interval_seconds: float = DEFAULT_MONITOR_SECONDS,
        terminal_grace_seconds: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, int | bool]:
        """Recover leases/restarts until every valid worklist row is terminal."""

        if terminal_grace_seconds is None:
            terminal_grace_seconds = max(
                DEFAULT_TERMINAL_GRACE_SECONDS,
                self.poll_seconds * 2.0 + 1.0,
            )
        if interval_seconds <= 0 or terminal_grace_seconds < 0:
            raise ValueError("monitor intervals must be positive/non-negative")
        stop_event = stop_event or threading.Event()
        last_snapshot: dict[str, int | bool] | None = None
        terminal_since: float | None = None
        run_id = self.preparation.paths.run_id
        while not stop_event.is_set():
            now = time.time()
            expired = self.state.release_expired_leases(
                run_id,
                max_attempts=self.max_attempts,
                now=now,
            )
            if expired:
                print(
                    f"[Princess] recovered expired leases: {', '.join(expired)}",
                    flush=True,
                )
            self._refresh_pending_launches(now)
            self._schedule_restarts(now)
            progress = self.state.progress(run_id).as_dict()
            self._schedule_recoveries(now, progress)
            if progress != last_snapshot:
                print(
                    "[Princess] progress "
                    f"completed={progress['completed']} running={progress['running']} "
                    f"pending={progress['pending']} failed={progress['failed']} "
                    f"total={progress['total']}",
                    flush=True,
                )
                last_snapshot = progress
            if bool(progress["is_terminal"]):
                terminal_since = terminal_since or now
                if now - terminal_since >= terminal_grace_seconds:
                    return progress
            else:
                terminal_since = None
                if int(progress["pending"]) > 0 and int(progress["running"]) == 0:
                    blocked_statuses: list[str] = []
                    for device in self.devices:
                        if device.id in self._disabled_for_run:
                            blocked_statuses.append("launch_exhausted")
                            continue
                        try:
                            status = str(
                                self.state.get_worker(run_id, device.id)["status"]
                            )
                        except UnknownWorkerError:
                            status = "unknown"
                        blocked_statuses.append(status)
                    if (
                        blocked_statuses
                        and all(
                            status in {"unhealthy", "quarantined", "stopped"}
                            or status == "launch_exhausted"
                            for status in blocked_statuses
                        )
                        and not self._launch_pending
                    ):
                        raise PrincessRuntimeError(
                            "run cannot progress: every selected Maid is "
                            + ", ".join(blocked_statuses)
                        )
            stop_event.wait(interval_seconds)
        return self.state.progress(run_id).as_dict()

    def close(self) -> None:
        server, thread, state = self._server, self._server_thread, self._state
        self._server = None
        self._server_thread = None
        self._state = None
        self._coordinator = None
        self._server_started_at = None
        self._launch_pending.clear()
        self._restart_due.clear()
        self._recovery_due.clear()
        self._launch_failures.clear()
        self._disabled_for_run.clear()
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)
        if state is not None:
            state.close()

    def __enter__(self) -> "PrincessRuntime":
        self.start_server()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_status(
    run_id: str,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    paths = PrincessRunPaths.for_run(run_id, repository_root=repository_root)
    if not paths.database.is_file():
        raise FileNotFoundError(f"Princess run database does not exist: {paths.database}")
    with PrincessState(paths.database) as state:
        progress = state.progress(paths.run_id).as_dict()
        workers: list[dict[str, Any]] = []
        if paths.runtime_json.is_file():
            runtime = json.loads(paths.runtime_json.read_text(encoding="utf-8"))
            for worker_id in runtime.get("device_ids", []):
                try:
                    worker = state.get_worker(paths.run_id, str(worker_id))
                except UnknownWorkerError:
                    worker = {"worker_id": str(worker_id), "status": "not_registered"}
                workers.append(worker)
    return {
        "run_id": paths.run_id,
        "progress": progress,
        "workers": workers,
        "results_root": str(paths.results_root),
    }


def copy_registry_with_device(
    registry_path: str | Path,
    device_mapping: Mapping[str, Any],
) -> DeviceRegistry:
    """Atomically append one validated device to a registry JSON file."""

    from .config import load_device_registry, registry_from_mapping

    path = Path(registry_path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    devices = raw.get("devices")
    if not isinstance(devices, list):
        raise PrincessRuntimeError("device registry has no devices array")
    device_id = str(device_mapping.get("id", "")).strip()
    if any(str(item.get("id", "")) == device_id for item in devices):
        raise PrincessRuntimeError(f"device already exists: {device_id!r}")
    candidate = dict(raw)
    candidate["devices"] = [*devices, dict(device_mapping)]
    registry_from_mapping(candidate)
    _atomic_write_bytes(
        path,
        (
            json.dumps(candidate, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8"),
    )
    return load_device_registry(path)


def mirror_project_for_local_debug(
    source: str | Path,
    destination: str | Path,
) -> Path:
    """Copy only a standalone CST container, never its generated sidecar."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path
