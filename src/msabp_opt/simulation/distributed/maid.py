"""Long-lived remote worker that asks Princess for CST simulation cases."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import threading
import time
import traceback
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import case_runner
from .http_api import ApiError, PrincessClient
from .protocol import make_message, validate_message
from .control_policy import CONTROL_MODES, control_mode


RUNTIME_SCHEMA_VERSION = 1
UNHEALTHY_EXIT_CODE = 75
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_HEARTBEAT_SECONDS = 15.0
DEFAULT_ARTIFACT_COMMIT_DEADLINE_SECONDS = 1800.0
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class MaidConfigError(ValueError):
    """Raised when the local runtime configuration is malformed."""


@dataclass(frozen=True)
class MaidRuntimeConfig:
    run_id: str
    worker_id: str
    princess_url: str
    api_token: str
    csv_path: Path
    csv_sha256: str
    project_path: Path
    output_root: Path
    dry_run: bool = False
    coordinate_quantum_mm: float = 0.01
    allow_disconnected_conductor: bool = False
    command_timeout_seconds: float = 15.0
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS
    poll_seconds: float = DEFAULT_POLL_SECONDS
    max_consecutive_errors: int = 5
    save_project_after_case: bool = False
    hello_timeout_seconds: float = 60.0
    artifact_timeout_seconds: float = 600.0
    artifact_commit_deadline_seconds: float = (
        DEFAULT_ARTIFACT_COMMIT_DEADLINE_SECONDS
    )
    protocol_mode: str = "legacy"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MaidRuntimeConfig":
        allowed = {
            "schema_version",
            "run_id",
            "worker_id",
            "princess_url",
            "api_token",
            "csv_path",
            "csv_sha256",
            "project_path",
            "output_root",
            "dry_run",
            "coordinate_quantum_mm",
            "allow_disconnected_conductor",
            "command_timeout_seconds",
            "heartbeat_seconds",
            "poll_seconds",
            "max_consecutive_errors",
            "save_project_after_case",
            "hello_timeout_seconds",
            "artifact_timeout_seconds",
            "artifact_commit_deadline_seconds",
            "protocol_mode",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise MaidConfigError(f"runtime config contains unknown keys: {sorted(unknown)}")
        if raw.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            raise MaidConfigError(
                f"runtime config schema_version must be {RUNTIME_SCHEMA_VERSION}"
            )

        def required_text(name: str) -> str:
            value = raw.get(name)
            if not isinstance(value, str) or not value.strip():
                raise MaidConfigError(f"{name} must be a non-empty string")
            return value.strip()

        def positive_float(name: str, default: float) -> float:
            value = raw.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MaidConfigError(f"{name} must be a number")
            result = float(value)
            if not math.isfinite(result) or result <= 0.0:
                raise MaidConfigError(f"{name} must be finite and positive")
            return result

        def boolean(name: str, default: bool) -> bool:
            value = raw.get(name, default)
            if not isinstance(value, bool):
                raise MaidConfigError(f"{name} must be boolean")
            return value

        max_errors = raw.get("max_consecutive_errors", 5)
        if isinstance(max_errors, bool) or not isinstance(max_errors, int):
            raise MaidConfigError("max_consecutive_errors must be an integer")
        if max_errors < 1:
            raise MaidConfigError("max_consecutive_errors must be positive")

        csv_sha256 = required_text("csv_sha256").lower()
        if not _SHA256_PATTERN.fullmatch(csv_sha256):
            raise MaidConfigError("csv_sha256 must contain exactly 64 hexadecimal digits")

        run_id = _validated_identifier(required_text("run_id"), "run_id")
        worker_id = _validated_identifier(required_text("worker_id"), "worker_id")
        protocol_mode = raw.get("protocol_mode", "legacy")
        if protocol_mode not in {"legacy", *CONTROL_MODES}:
            raise MaidConfigError("protocol_mode must be legacy or an explicit Shore mode")
        return cls(
            run_id=run_id,
            worker_id=worker_id,
            princess_url=required_text("princess_url"),
            api_token=required_text("api_token"),
            csv_path=Path(required_text("csv_path")).expanduser().resolve(),
            csv_sha256=csv_sha256,
            project_path=Path(required_text("project_path")).expanduser().resolve(),
            output_root=Path(required_text("output_root")).expanduser().resolve(),
            dry_run=boolean("dry_run", False),
            coordinate_quantum_mm=positive_float("coordinate_quantum_mm", 0.01),
            allow_disconnected_conductor=boolean(
                "allow_disconnected_conductor", False
            ),
            command_timeout_seconds=positive_float("command_timeout_seconds", 15.0),
            heartbeat_seconds=positive_float("heartbeat_seconds", 15.0),
            poll_seconds=positive_float("poll_seconds", 5.0),
            max_consecutive_errors=max_errors,
            save_project_after_case=boolean("save_project_after_case", False),
            hello_timeout_seconds=positive_float("hello_timeout_seconds", 60.0),
            artifact_timeout_seconds=positive_float(
                "artifact_timeout_seconds",
                600.0,
            ),
            artifact_commit_deadline_seconds=positive_float(
                "artifact_commit_deadline_seconds",
                DEFAULT_ARTIFACT_COMMIT_DEADLINE_SECONDS,
            ),
            protocol_mode=protocol_mode,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MaidRuntimeConfig":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise MaidConfigError("runtime config must contain one JSON object")
        return cls.from_mapping(payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv_table(
    path: str | Path,
    expected_sha256: str,
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    source = Path(path).expanduser().resolve()
    actual_sha256 = sha256_file(source)
    if actual_sha256.lower() != expected_sha256.lower():
        raise MaidConfigError(
            f"CSV SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise MaidConfigError("CSV has no header")
        return tuple(reader.fieldnames), [dict(row) for row in reader]


def load_csv_rows(path: str | Path, expected_sha256: str) -> list[dict[str, str]]:
    return load_csv_table(path, expected_sha256)[1]


def canonical_row_sha256(
    fieldnames: tuple[str, ...],
    row: Mapping[str, Any],
) -> str:
    canonical = json.dumps(
        [[name, row[name]] for name in fieldnames],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def doctor(
    config: MaidRuntimeConfig,
    *,
    require_cst: bool | None = None,
) -> dict[str, Any]:
    """Check the environment without opening CST or changing the project."""

    if require_cst is None:
        require_cst = not config.dry_run
    if require_cst:
        from .control_policy import reject_external_execution

        reject_external_execution("CST doctor import")
        if config.protocol_mode in CONTROL_MODES:
            raise MaidConfigError("Shore fake-worker doctor refuses CST checks")
    if require_cst and sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "CST Studio Suite 2025 Maid requires Python 3.11; "
            f"found {platform.python_version()}"
        )
    modules: dict[str, str] = {}
    required_modules = ("numpy", "scipy", "pandas", "shapely")
    if require_cst:
        required_modules += ("cst.interface",)
    for module_name in required_modules:
        module = __import__(module_name, fromlist=["*"])
        modules[module_name] = str(getattr(module, "__version__", "available"))
    if not config.csv_path.is_file():
        raise FileNotFoundError(f"Maid CSV does not exist: {config.csv_path}")
    rows = load_csv_rows(config.csv_path, config.csv_sha256)
    if not config.dry_run and not config.project_path.is_file():
        raise FileNotFoundError(f"Maid CST project does not exist: {config.project_path}")
    return {
        "worker_id": config.worker_id,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "modules": modules,
        "csv_rows": len(rows),
        "csv_sha256": config.csv_sha256,
        "project_path": str(config.project_path),
        "dry_run": config.dry_run,
    }


class OwnedCstSession:
    """One CST environment owned exclusively by this Maid process."""

    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path
        self.environment: Any | None = None
        self.project: Any | None = None

    def open(self) -> Any:
        from .control_policy import reject_external_execution

        reject_external_execution("CST session open")
        if self.project is not None:
            return self.project
        import cst.interface

        environment = cst.interface.DesignEnvironment.new()
        try:
            project = environment.open_project(str(self.project_path))
        except Exception:
            try:
                environment.close()
            except Exception:
                pass
            raise
        self.environment = environment
        self.project = project
        return project

    def close(self) -> None:
        project, environment = self.project, self.environment
        self.project = None
        self.environment = None
        if project is not None:
            try:
                project.close()
            except Exception:
                pass
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass


class _Heartbeat:
    def __init__(
        self,
        maid: "Maid",
        assignment: Mapping[str, Any],
    ) -> None:
        self.maid = maid
        self.assignment = assignment
        self.stage = "assigned"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "_Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(self.maid.config.heartbeat_seconds * 2.0, 1.0))

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def _loop(self) -> None:
        while not self.stop_event.wait(self.maid.config.heartbeat_seconds):
            try:
                self.maid._send(
                    "heartbeat",
                    {
                        "attempt_id": self.assignment["attempt_id"],
                        "lease_token": self.assignment["lease_token"],
                        "phase": self.stage,
                    },
                )
            except Exception as exc:
                self.maid._log(f"heartbeat failed: {type(exc).__name__}: {exc}")


class Maid:
    """Poll Princess, run assigned rows locally, and upload verified artifacts."""

    def __init__(
        self,
        config: MaidRuntimeConfig,
        *,
        client: PrincessClient | Any | None = None,
        artifact_client: PrincessClient | Any | None = None,
        session: OwnedCstSession | Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        control_worker: Any | None = None,
        run_csv_row: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.control_worker = control_worker
        controlled = config.protocol_mode in CONTROL_MODES
        selected_mode = control_mode()
        if selected_mode and config.protocol_mode != selected_mode:
            raise MaidConfigError("Controlled environment refuses legacy or different Shore mode")
        if controlled and (
            control_worker is None or run_csv_row is None
            or run_csv_row is case_runner.run_csv_row
            or session is None or isinstance(session, OwnedCstSession)
        ):
            raise MaidConfigError(
                "Shore synthetic mode requires a Shore gate and explicitly injected fake executor/session"
            )
        if not controlled and control_worker is not None:
            raise MaidConfigError("Shore control requires explicit synthetic protocol_mode")
        self._run_csv_row = run_csv_row or case_runner.run_csv_row
        self.client = client or PrincessClient(config.princess_url, config.api_token)
        self.artifact_client = artifact_client or (
            self.client
            if client is not None
            else PrincessClient(
                config.princess_url,
                config.api_token,
                timeout=config.artifact_timeout_seconds,
            )
        )
        self.fieldnames, self.rows = load_csv_table(
            config.csv_path,
            config.csv_sha256,
        )
        self.session = session or OwnedCstSession(config.project_path)
        self._sleep = sleeper
        self.failed_case_ids: list[str] = []
        self.config.output_root.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        print(f"[Maid:{self.config.worker_id}] {message}", flush=True)

    def _send(self, message_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._send_via(self.client, message_type, payload)

    def _send_via(
        self,
        client: Any,
        message_type: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        response = client.send(
            make_message(
                message_type,
                run_id=self.config.run_id,
                worker_id=self.config.worker_id,
                payload=payload,
                protocol_version=2 if self.control_worker is not None else 1,
            )
        )
        validated = validate_message(response)
        if self.control_worker is not None and validated["protocol_version"] != 2:
            raise MaidConfigError("Shore Maid refuses a protocol downgrade")
        return validated

    @staticmethod
    def _is_transient_http_error(error: Exception) -> bool:
        if isinstance(error, ApiError):
            return error.status in {408, 425, 429} or error.status >= 500
        return isinstance(error, (ConnectionError, TimeoutError, OSError))

    def _retry_transient(
        self,
        operation: Callable[[], Any],
        *,
        label: str,
        deadline_seconds: float | None = None,
        deadline_at: float | None = None,
    ) -> Any:
        started = time.monotonic()
        if deadline_seconds is not None and deadline_at is not None:
            raise ValueError("specify either deadline_seconds or deadline_at, not both")
        if deadline_seconds is not None:
            deadline_at = started + deadline_seconds
        retry_budget = None if deadline_at is None else max(deadline_at - started, 0.0)
        failures = 0
        last_error: Exception | None = None
        while True:
            if deadline_at is not None and time.monotonic() >= deadline_at:
                error = TimeoutError(
                    f"{label} did not succeed within {retry_budget:g}s"
                )
                if last_error is None:
                    raise error
                raise error from last_error
            try:
                return operation()
            except Exception as exc:
                if not self._is_transient_http_error(exc):
                    raise
                failures += 1
                last_error = exc
                now = time.monotonic()
                if deadline_at is not None and now >= deadline_at:
                    raise TimeoutError(
                        f"{label} did not succeed within {retry_budget:g}s"
                    ) from exc
                delay = min(self.config.poll_seconds * (2 ** min(failures - 1, 4)), 30.0)
                if deadline_at is not None:
                    delay = min(delay, max(deadline_at - now, 0.0))
                self._log(
                    f"{label} temporarily failed ({type(exc).__name__}: {exc}); "
                    f"retrying in {delay:g}s"
                )
                self._sleep(delay)

    def _send_with_retry(
        self,
        message_type: str,
        payload: Mapping[str, Any],
        *,
        deadline_seconds: float | None = None,
        deadline_at: float | None = None,
        artifact_channel: bool = False,
    ) -> dict[str, Any]:
        client = self.artifact_client if artifact_channel else self.client
        return self._retry_transient(
            lambda: self._send_via(client, message_type, payload),
            label=f"{message_type} request",
            deadline_seconds=deadline_seconds,
            deadline_at=deadline_at,
        )

    def run(self) -> int:
        try:
            hello = self._send_with_retry(
                "hello",
                {
                    "hostname": platform.node(),
                    "pid": os.getpid(),
                    "csv_sha256": self.config.csv_sha256,
                    "dry_run": self.config.dry_run,
                    "protocol_mode": self.config.protocol_mode,
                    "supported_simulation_modes": list(
                        case_runner.SUPPORTED_SIMULATION_MODES
                    ),
                },
                deadline_seconds=self.config.hello_timeout_seconds,
            )
            if hello["type"] != "welcome":
                raise RuntimeError(f"Princess rejected Maid hello: {hello['type']}")

            if self.control_worker is not None:
                # A collector whose local completion already committed may
                # now reply stop; replay our own durable terminal outbox first.
                # STARTING/unknown execution is refused by pending_assignments.
                for assignment in self.control_worker.pending_assignments():
                    outcome = self._run_assignment(assignment)
                    if outcome in {"failed", "not_started", "failed_unresolved", "completed_pending"}:
                        self._log("Shore terminal outcome reconciled; stopping without automatic restart"
                                  if outcome in {"failed", "not_started"}
                                  else "Recovered Shore outbox remains unresolved")
                        return 2

            while True:
                response = self._send_with_retry(
                    "request_task",
                    {"exclude_case_ids": list(self.failed_case_ids)},
                )
                response_type = response["type"]
                payload = response["payload"]
                if response_type == "stop":
                    self._log("Princess reports that the run is complete")
                    return 0
                if response_type == "retry_paused":
                    self._log(f"Princess paused controlled retries: {payload.get('reason')}; "
                              f"cases={payload.get('cases', [])}")
                    return 2
                if response_type == "wait":
                    self._sleep(
                        float(
                            payload.get(
                                "retry_after_seconds",
                                self.config.poll_seconds,
                            )
                        )
                    )
                    continue
                if response_type != "assignment":
                    raise RuntimeError(f"unexpected Princess response: {response_type}")

                outcome = self._run_assignment(payload)
                if self.control_worker is not None and outcome in {"failed", "not_started", "failed_unresolved", "completed_pending"}:
                    self._log("Shore terminal outcome reconciled; stopping without automatic restart"
                              if outcome in {"failed", "not_started"}
                              else "Shore work remains unresolved; explicit recovery required")
                    return 2
                if outcome == "restart":
                    self._log("five consecutive failures; shutting down for restart")
                    return UNHEALTHY_EXIT_CODE
        finally:
            self.session.close()

    def _run_assignment(self, assignment: Mapping[str, Any]) -> str:
        case_id = _validated_identifier(assignment["case_id"], "case_id")
        row_index = int(assignment["row_index"])
        attempt_id = _validated_identifier(
            assignment["attempt_id"],
            "attempt_id",
        )
        lease_token = str(assignment["lease_token"])
        if not 0 <= row_index < len(self.rows):
            raise RuntimeError(f"Princess assigned invalid CSV row index {row_index}")
        row = self.rows[row_index]
        if "sample_id" in self.fieldnames and str(row["sample_id"]).strip() != case_id:
            raise RuntimeError(
                f"assignment case_id {case_id!r} does not match CSV sample_id "
                f"{row['sample_id']!r} at row {row_index}"
            )
        expected_row_sha256 = str(assignment.get("row_sha256", "")).lower()
        if not _SHA256_PATTERN.fullmatch(expected_row_sha256):
            raise RuntimeError("Princess assignment has no valid row_sha256")
        actual_row_sha256 = canonical_row_sha256(self.fieldnames, row)
        if actual_row_sha256 != expected_row_sha256:
            raise RuntimeError(
                f"assignment row SHA-256 mismatch at row {row_index}: "
                f"expected {expected_row_sha256}, got {actual_row_sha256}"
            )

        if self.control_worker is not None:
            return self._run_controlled_assignment(assignment, row, actual_row_sha256)
        if assignment.get("control_authority") is not None:
            raise MaidConfigError("Legacy Maid refuses a Shore-controlled assignment")

        attempt_root = self.config.output_root / "attempts" / attempt_id
        attempt_root.mkdir(parents=True, exist_ok=True)
        with _Heartbeat(self, assignment) as heartbeat:
            try:
                heartbeat.set_stage("opening_project")
                project = None if self.config.dry_run else self.session.open()
                result = self._run_csv_row(
                    row,
                    project_path=self.config.project_path,
                    output_root=attempt_root,
                    project=project,
                    case_id=case_id,
                    coordinate_quantum_mm=self.config.coordinate_quantum_mm,
                    allow_disconnected_conductor=(
                        self.config.allow_disconnected_conductor
                    ),
                    command_timeout=self.config.command_timeout_seconds,
                    overwrite=True,
                    save_project_after_case=self.config.save_project_after_case,
                    dry_run=self.config.dry_run,
                    stage_callback=heartbeat.set_stage,
                    local_artifact_root=self.config.output_root / "local_only",
                )
            except Exception as exc:
                return self._report_failure(
                    assignment,
                    exc,
                    traceback.format_exc(),
                )

            try:
                heartbeat.set_stage("packaging_results")
                archive = self._archive_result(attempt_id, result.case_directory)
                archive_sha256 = sha256_file(archive)
            except Exception as exc:
                return self._report_failure(
                    assignment,
                    exc,
                    traceback.format_exc(),
                )

            # Upload and completion are commit operations.  A timeout means the
            # Princess side may still have accepted them, so retry the same
            # idempotency keys instead of misreporting a successful CST run as
            # a simulation failure.
            commit_deadline = (
                time.monotonic() + self.config.artifact_commit_deadline_seconds
            )
            heartbeat.set_stage("uploading_results")
            upload = self._retry_transient(
                lambda: self.artifact_client.upload_artifact(
                    attempt_id,
                    self.config.worker_id,
                    lease_token,
                    archive,
                ),
                label="artifact upload",
                deadline_at=commit_deadline,
            )
            if upload.get("sha256") != archive_sha256:
                raise RuntimeError("Princess artifact SHA-256 acknowledgement mismatch")
            heartbeat.set_stage("committing_results")
            response = self._send_with_retry(
                "complete",
                {
                    "attempt_id": attempt_id,
                    "lease_token": lease_token,
                    "archive_sha256": archive_sha256,
                    "manifest_sha256": sha256_file(result.manifest_path),
                },
                deadline_at=commit_deadline,
                artifact_channel=True,
            )
            if response["type"] != "completed_ack":
                raise RuntimeError(
                    f"Princess did not acknowledge completion: {response['type']}"
                )
            self._cleanup_completed_attempt(attempt_root, archive)

        self.failed_case_ids.clear()
        self._log(f"case {case_id} completed")
        return "completed"

    def _run_controlled_assignment(
        self, assignment: Mapping[str, Any], row: Mapping[str, Any], row_sha256: str,
    ) -> str:
        """Synthetic execution through Maid's normal entry with durable replay."""
        gate = self.control_worker
        attempt_id = str(assignment["attempt_id"])
        permit = gate.accept(assignment, run_id=self.config.run_id, worker_id=self.config.worker_id,
                             row_sha256=row_sha256, csv_sha256=self.config.csv_sha256)
        if gate.is_resolved(attempt_id):
            return gate.result(attempt_id)["receipt"]["outcome"]
        durable_result = gate.result(attempt_id)
        attempt_root = self.config.output_root / "attempts" / attempt_id
        attempt_root.mkdir(parents=True, exist_ok=True)
        with _Heartbeat(self, assignment) as heartbeat:
            if durable_result is None:
                # Persist STARTING before even opening the injected fake session.
                gate.start(attempt_id)
                try:
                    project = None if self.config.dry_run else self.session.open()
                    result = self._run_csv_row(
                        row, project_path=self.config.project_path, output_root=attempt_root,
                        project=project, case_id=str(assignment["case_id"]),
                        coordinate_quantum_mm=self.config.coordinate_quantum_mm,
                        allow_disconnected_conductor=self.config.allow_disconnected_conductor,
                        command_timeout=self.config.command_timeout_seconds, overwrite=False,
                        save_project_after_case=False, dry_run=self.config.dry_run,
                        stage_callback=heartbeat.set_stage,
                        local_artifact_root=self.config.output_root / "local_only",
                    )
                except Exception as exc:
                    # A returned synthetic exception is a confirmed termination;
                    # BaseException/process death remains STARTING and unresolved.
                    durable_result = gate.save_result(
                        attempt_id, attempt_root / "failure", outcome="failed",
                        failure_code=type(exc).__name__, error_message=str(exc),
                    )
                else:
                    durable_result = gate.save_result(
                        attempt_id, result.case_directory, outcome="completed",
                        manifest_path=result.manifest_path,
                    )
            durable_result = gate.checked_result(attempt_id)
            self._report_control_outbox(attempt_id)
            if durable_result["receipt"]["outcome"] != "completed":
                return self._deliver_control_terminal(assignment, durable_result, permit, attempt_root)
            delivery = gate.delivery(attempt_id)
            if delivery is None:
                archive = self._archive_result(attempt_id, Path(durable_result["root"]))
                with archive.open("r+b") as stream:
                    os.fsync(stream.fileno())
                delivery = {
                    "archive": str(archive),
                    "payload": {
                        "attempt_id": attempt_id, "lease_token": str(assignment["lease_token"]),
                        "archive_sha256": sha256_file(archive),
                        "manifest_sha256": sha256_file(durable_result["manifest_path"]),
                        "control_authority": {
                            "permit_id": durable_result["receipt"]["permit_id"],
                            "evidence": durable_result["evidence"],
                            "aliases": durable_result["aliases"],
                            "evidence_sha256": durable_result["receipt"]["evidence_sha256"],
                            "outcome": "completed",
                        },
                    },
                }
                gate.delivery(attempt_id, delivery)
            archive = Path(delivery["archive"])
            if sha256_file(archive) != delivery["payload"]["archive_sha256"]:
                raise ValueError("Durable Shore archive bytes changed")
            commit_deadline = time.monotonic() + self.config.artifact_commit_deadline_seconds
            heartbeat.set_stage("uploading_results")
            if not gate.uploaded(attempt_id):
                upload = self._retry_transient(
                    lambda: self.artifact_client.upload_artifact(
                        attempt_id, self.config.worker_id, str(assignment["lease_token"]), archive),
                    label="artifact upload", deadline_at=commit_deadline,
                )
                if upload.get("sha256") != delivery["payload"]["archive_sha256"]:
                    raise RuntimeError("Princess artifact SHA-256 acknowledgement mismatch")
                gate.uploaded(attempt_id, upload["sha256"])
            heartbeat.set_stage("committing_results")
            response = self._send_with_retry(
                "complete", delivery["payload"], deadline_at=commit_deadline, artifact_channel=True,
            )
            if response["type"] != "completed_ack":
                raise RuntimeError("Princess did not acknowledge controlled completion")
            self._report_control_outbox(attempt_id)
            try:
                resolved = gate.resolved(attempt_id)
            except Exception as exc:
                if not self._is_control_network_error(exc):
                    raise
                self._log(f"Shore resolution unavailable; retaining outbox: {exc}")
                resolved = False
            if resolved:
                self._cleanup_completed_attempt(attempt_root, archive)
                self.failed_case_ids.clear()
                return "completed"
            self._log("Collector replied, but Shore is not RESOLVED; retaining all result bytes")
            return "completed_pending"

    def _deliver_control_terminal(self, assignment, result, permit, attempt_root) -> str:
        from .control_failure import terminal_payload

        gate, attempt_id = self.control_worker, str(assignment["attempt_id"])
        delivery = gate.delivery(attempt_id)
        if delivery is None:
            delivery = {"terminal_payload": terminal_payload(dict(assignment), result, permit)}
            gate.delivery(attempt_id, delivery)
        elif "terminal_payload" not in delivery:
            raise ValueError("Terminal outcome conflicts with a completed delivery")
        response = self._send_with_retry(
            "control_terminal", delivery["terminal_payload"], artifact_channel=True,
            deadline_seconds=self.config.artifact_commit_deadline_seconds,
        )
        if (response["type"] != "terminal_ack"
                or response["payload"].get("outcome") != result["receipt"]["outcome"]):
            raise ValueError("Collector did not acknowledge the same terminal outcome")
        self._report_control_outbox(attempt_id)
        try:
            resolved = gate.resolved(attempt_id)
        except Exception as exc:
            if not self._is_control_network_error(exc):
                raise
            self._log(f"Shore terminal resolution pending; retaining record: {exc}")
            resolved = False
        if not resolved:
            return "failed_unresolved"
        # The durable inbox/tombstone remains. Only disposable per-attempt files
        # are cleaned, and only after our own signed RESOLVED verification.
        archive = self.config.output_root / "outbox" / f"{attempt_id}.zip"
        self._cleanup_completed_attempt(attempt_root, archive)
        return result["receipt"]["outcome"]

    def _report_control_outbox(self, attempt_id: str) -> None:
        try:
            self.control_worker.report(attempt_id)
        except Exception as exc:
            if not self._is_control_network_error(exc):
                raise
            self._log(f"Shore worker report pending in durable outbox: {exc}")

    @staticmethod
    def _is_control_network_error(error: Exception) -> bool:
        # Shore's RPC transport wraps network/invalid transport responses as
        # AuthorityError('RPC'); retaining bytes never treats that as authority.
        return isinstance(error, (ConnectionError, TimeoutError, OSError)) or getattr(error, "code", None) == "RPC"

    def _cleanup_completed_attempt(self, attempt_root: Path, archive: Path) -> None:
        try:
            archive.unlink(missing_ok=True)
            shutil.rmtree(attempt_root, ignore_errors=False)
        except OSError as exc:
            self._log(
                f"completed result cleanup failed ({type(exc).__name__}: {exc}); "
                "the retained copy is safe to remove later"
            )

    def _report_failure(
        self,
        assignment: Mapping[str, Any],
        error: Exception,
        traceback_text: str,
    ) -> str:
        case_id = str(assignment["case_id"])
        # Geometry rejected explicitly by the sampler is a permanent case
        # failure.  Other precheck exceptions can be dependency/code failures
        # and must not silently bypass the Maid health/retry policy.
        row_index = int(assignment["row_index"])
        row = self.rows[row_index]
        explicit_invalid_geometry = False
        if "geometry_valid" in row:
            try:
                explicit_invalid_geometry = str(row["geometry_valid"]).strip().lower() in {
                    "0",
                    "false",
                    "f",
                    "no",
                    "n",
                    "off",
                }
            except Exception:
                explicit_invalid_geometry = False
        permanent_precheck = (
            isinstance(error, case_runner.CaseRunError)
            and error.stage == "precheck"
            and explicit_invalid_geometry
        )
        response = self._send_with_retry(
            "failure",
            {
                "attempt_id": str(assignment["attempt_id"]),
                "lease_token": str(assignment["lease_token"]),
                "error_kind": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback_text,
                "retryable": not permanent_precheck,
                "counts_toward_streak": not permanent_precheck,
            },
        )
        if response["type"] not in {"failure_ack", "restart"}:
            raise RuntimeError(f"Princess rejected failure report: {response['type']}")
        if not permanent_precheck:
            self.failed_case_ids.append(case_id)
            self.failed_case_ids = self.failed_case_ids[-self.config.max_consecutive_errors :]
        self._log(f"case {case_id} failed: {type(error).__name__}: {error}")
        local_restart = (
            not permanent_precheck
            and len(self.failed_case_ids) >= self.config.max_consecutive_errors
        )
        return "restart" if response["type"] == "restart" or local_restart else "failed"

    def _archive_result(self, attempt_id: str, case_directory: Path) -> Path:
        attempt_id = _validated_identifier(attempt_id, "attempt_id")
        outbox = self.config.output_root / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        archive = outbox / f"{attempt_id}.zip"
        temporary = archive.with_suffix(".zip.tmp")
        temporary.unlink(missing_ok=True)
        try:
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as handle:
                for path in sorted(case_directory.rglob("*")):
                    if path.is_file():
                        handle.write(path, path.relative_to(case_directory))
            temporary.replace(archive)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return archive


def _validated_identifier(value: Any, label: str) -> str:
    resolved = str(value).strip()
    if not _SAFE_IDENTIFIER.fullmatch(resolved) or resolved in {".", ".."}:
        raise ValueError(f"{label} is unsafe: {resolved!r}")
    return resolved


def write_runtime_config(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write the ignored, per-run Maid launch configuration."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def copy_project_template(source: str | Path, destination: str | Path) -> Path:
    """Create a Maid-owned standalone project copy without its CST sidecar."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"CST template does not exist: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f"{destination_path.name}.tmp")
    shutil.copy2(source_path, temporary)
    temporary.replace(destination_path)
    return destination_path
