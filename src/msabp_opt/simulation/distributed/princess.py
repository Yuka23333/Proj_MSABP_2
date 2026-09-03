"""Princess coordinator: authoritative scheduling and artifact acceptance."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
import threading
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from pathlib import Path, PurePosixPath
from typing import Any

from . import case_runner
from .http_api import ApiError
from .protocol import make_message
from .state import StateError


MAX_ARCHIVE_FILES = 10_000
MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_COMPLETION_RECEIPT_SUFFIX = ".completed.json"


def _record_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"state record is not serializable: {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be a non-empty string")
    return value.strip()


def _safe_identifier(value: Any, name: str) -> str:
    resolved = str(value).strip()
    if not _SAFE_ID.fullmatch(resolved) or resolved in {".", ".."}:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} is unsafe")
    return resolved


def _required_sha256(payload: Mapping[str, Any], name: str) -> str:
    value = _required_text(payload, name).lower()
    if not _SHA256.fullmatch(value):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"{name} must contain exactly 64 hexadecimal digits",
        )
    return value


def _optional_bool(payload: Mapping[str, Any], name: str, default: bool) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be boolean")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{label} is unsafe")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{label} is unsafe")
    if any(
        ":" in part or part in {"", ".", ".."} or part.rstrip(" .") != part
        for part in relative.parts
    ):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{label} is unsafe")
    return relative


class PrincessCoordinator:
    """Map Maid protocol messages onto one durable :class:`PrincessState`."""

    def __init__(
        self,
        state: Any,
        *,
        run_id: str,
        csv_sha256: str,
        results_dir: str | Path,
        incoming_dir: str | Path,
        lease_seconds: float = 90.0,
        max_attempts: int = 3,
        streak_threshold: int = 5,
        max_automatic_restarts_without_success: int = 1,
        required_simulation_modes: Sequence[str] = (
            case_runner.DEFAULT_SIMULATION_MODE,
        ),
    ) -> None:
        self.state = state
        self.run_id = run_id
        self.csv_sha256 = csv_sha256.lower()
        self.results_dir = Path(results_dir).expanduser().resolve()
        self.incoming_dir = Path(incoming_dir).expanduser().resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = float(lease_seconds)
        self.max_attempts = int(max_attempts)
        self.streak_threshold = int(streak_threshold)
        self.max_automatic_restarts_without_success = int(
            max_automatic_restarts_without_success
        )
        self.required_simulation_modes = frozenset(
            str(mode).strip() for mode in required_simulation_modes
        )
        if not self.required_simulation_modes:
            raise ValueError("required_simulation_modes must not be empty")
        self._attempt_locks_guard = threading.Lock()
        self._attempt_locks: dict[str, threading.Lock] = {}

    @staticmethod
    def _state_call(operation: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except StateError as exc:
            raise ApiError(HTTPStatus.CONFLICT, str(exc)) from exc

    def _completion_receipt_path(self, attempt_id: str) -> Path:
        return self.incoming_dir / f"{attempt_id}{_COMPLETION_RECEIPT_SUFFIX}"

    def _attempt_lock(self, attempt_id: str) -> threading.Lock:
        """Return the process-local lock that serializes one attempt's commit path."""
        with self._attempt_locks_guard:
            lock = self._attempt_locks.get(attempt_id)
            if lock is None:
                lock = threading.Lock()
                self._attempt_locks[attempt_id] = lock
            return lock

    def _load_completion_receipt(
        self,
        *,
        attempt_id: str,
        worker_id: str,
        lease_token: str,
        archive_sha256: str,
        manifest_sha256: str,
    ) -> dict[str, Any] | None:
        path = self._completion_receipt_path(attempt_id)
        if not path.is_file():
            return None
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"completion receipt is unreadable: {path}",
            ) from exc
        if not isinstance(receipt, Mapping):
            raise ApiError(HTTPStatus.CONFLICT, "completion receipt is invalid")

        expected = {
            "run_id": self.run_id,
            "attempt_id": attempt_id,
            "worker_id": worker_id,
            "lease_token_sha256": hashlib.sha256(
                lease_token.encode("utf-8")
            ).hexdigest(),
            "archive_sha256": archive_sha256,
            "manifest_sha256": manifest_sha256,
        }
        for name, value in expected.items():
            if not hmac.compare_digest(str(receipt.get(name, "")), value):
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"duplicate completion does not match receipt field {name}",
                )

        artifact_path = Path(str(receipt.get("artifact_path", ""))).resolve()
        try:
            artifact_path.relative_to(self.results_dir)
        except ValueError as exc:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "completion receipt artifact path is outside the results directory",
            ) from exc
        manifest_path = artifact_path / case_runner.MANIFEST_FILENAME
        if (
            not manifest_path.is_file()
            or _sha256_file(manifest_path) != manifest_sha256
        ):
            raise ApiError(
                HTTPStatus.CONFLICT,
                "completed artifact no longer matches its receipt",
            )
        return dict(receipt)

    def _reply(
        self,
        request: Mapping[str, Any],
        message_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return make_message(
            message_type,
            run_id=self.run_id,
            worker_id=str(request.get("worker_id", "")) or None,
            payload=payload,
        )

    def handle_message(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["run_id"] != self.run_id:
            raise ApiError(HTTPStatus.CONFLICT, "Maid run_id does not match Princess")
        worker_id = _safe_identifier(
            _required_text(request, "worker_id"),
            "worker_id",
        )
        payload = request["payload"]
        message_type = request["type"]

        if message_type == "hello":
            maid_csv_sha256 = _required_text(payload, "csv_sha256").lower()
            if maid_csv_sha256 != self.csv_sha256:
                raise ApiError(HTTPStatus.CONFLICT, "Maid CSV SHA-256 does not match")
            raw_modes = payload.get("supported_simulation_modes")
            if raw_modes is None:
                # Compatibility for older Maids is safe only for the original
                # single-antenna workflow.  A propagation campaign must never
                # silently fall back to that runner.
                supported_modes = {case_runner.DEFAULT_SIMULATION_MODE}
            elif not isinstance(raw_modes, list) or not all(
                isinstance(mode, str) and mode.strip() for mode in raw_modes
            ):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "Maid supported_simulation_modes must be an array of strings",
                )
            else:
                supported_modes = {mode.strip() for mode in raw_modes}
            missing_modes = self.required_simulation_modes - supported_modes
            if missing_modes:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "Maid code is too old for this worklist; missing simulation "
                    f"modes: {sorted(missing_modes)}. Pull the current commit on "
                    "the Maid host before restarting Princess.",
                )
            self.state.register_worker(
                self.run_id,
                worker_id,
                host=str(payload.get("hostname", "")),
                transport="http",
                metadata={
                    "pid": payload.get("pid"),
                    "dry_run": _optional_bool(payload, "dry_run", False),
                    "supported_simulation_modes": sorted(supported_modes),
                },
            )
            return self._reply(
                request,
                "welcome",
                {
                    "csv_sha256": self.csv_sha256,
                    "progress": _record_mapping(self.state.progress(self.run_id)),
                },
            )

        if message_type == "request_task":
            self.state.mark_worker_ready(self.run_id, worker_id)
            raw_exclusions = payload.get("exclude_case_ids", [])
            if not isinstance(raw_exclusions, list) or not all(
                isinstance(item, str) for item in raw_exclusions
            ):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "exclude_case_ids must be an array of strings",
                )
            claim = self.state.claim_next(
                self.run_id,
                worker_id,
                lease_seconds=self.lease_seconds,
                max_attempts=self.max_attempts,
                exclude_case_ids=tuple(raw_exclusions),
            )
            if claim is None and raw_exclusions:
                progress = _record_mapping(self.state.progress(self.run_id))
                if int(progress.get("pending", 0)) > 0:
                    # Prefer a fresh case, but do not deadlock a Maid when every
                    # remaining pending case is in its short local exclusion list.
                    claim = self.state.claim_next(
                        self.run_id,
                        worker_id,
                        lease_seconds=self.lease_seconds,
                        max_attempts=self.max_attempts,
                        exclude_case_ids=(),
                    )
            if claim is not None:
                return self._reply(request, "assignment", _record_mapping(claim))
            progress = _record_mapping(self.state.progress(self.run_id))
            unfinished = int(progress.get("pending", 0)) + int(
                progress.get("running", 0)
            )
            if unfinished == 0:
                return self._reply(request, "stop", {"progress": progress})
            return self._reply(
                request,
                "wait",
                {"retry_after_seconds": 5.0, "progress": progress},
            )

        if message_type == "heartbeat":
            expires_at = self.state.heartbeat(
                self.run_id,
                worker_id,
                _required_text(payload, "attempt_id"),
                _required_text(payload, "lease_token"),
                lease_seconds=self.lease_seconds,
                phase=str(payload.get("phase", "")),
            )
            return self._reply(
                request,
                "heartbeat_ack",
                {"lease_expires_at": expires_at},
            )

        if message_type == "failure":
            attempt_id = _safe_identifier(
                _required_text(payload, "attempt_id"),
                "attempt_id",
            )
            lease_token = _required_text(payload, "lease_token")
            with self._attempt_lock(attempt_id):
                # A completion that won the race is authoritative.  Rechecking
                # under the attempt lock prevents a late failure from changing
                # an already committed task.
                if self._completion_receipt_path(attempt_id).is_file():
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        "attempt is already completed",
                    )
                outcome = self.state.fail_task(
                    self.run_id,
                    worker_id,
                    attempt_id,
                    lease_token,
                    error_kind=_required_text(payload, "error_kind"),
                    error_message=_required_text(payload, "error_message"),
                    traceback_text=str(payload.get("traceback", "")),
                    log_path=str(payload.get("log_path", "")),
                    retryable=_optional_bool(payload, "retryable", True),
                    max_attempts=self.max_attempts,
                    counts_toward_streak=_optional_bool(
                        payload,
                        "counts_toward_streak",
                        True,
                    ),
                    streak_threshold=self.streak_threshold,
                )
                outcome_payload = _record_mapping(outcome)
                should_restart = bool(
                    outcome_payload.get("restart_worker")
                    or outcome_payload.get("should_restart")
                    or outcome_payload.get("streak_reached")
                )
                if should_restart:
                    failures = self.state.failure_bundle(
                        self.run_id,
                        worker_id,
                        limit=self.streak_threshold,
                    )
                    outcome_payload["failure_bundle"] = [
                        _record_mapping(item) for item in failures
                    ]
                    decision = self.state.prepare_worker_restart(
                        self.run_id,
                        worker_id,
                        max_automatic_restarts_without_success=(
                            self.max_automatic_restarts_without_success
                        ),
                    )
                    outcome_payload["restart_decision"] = _record_mapping(decision)
                    return self._reply(request, "restart", outcome_payload)
                return self._reply(request, "failure_ack", outcome_payload)

        if message_type == "complete":
            attempt_id = _safe_identifier(
                _required_text(payload, "attempt_id"),
                "attempt_id",
            )
            lease_token = _required_text(payload, "lease_token")
            expected_archive_sha256 = _required_sha256(payload, "archive_sha256")
            expected_manifest_sha256 = _required_sha256(
                payload,
                "manifest_sha256",
            )
            with self._attempt_lock(attempt_id):
                # Receipt and staged-archive checks belong inside the lock so a
                # duplicate request observes the first request's committed state.
                receipt = self._load_completion_receipt(
                    attempt_id=attempt_id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                    archive_sha256=expected_archive_sha256,
                    manifest_sha256=expected_manifest_sha256,
                )
                if receipt is not None:
                    return self._reply(
                        request,
                        "completed_ack",
                        {
                            "artifact_path": str(receipt["artifact_path"]),
                            "manifest_sha256": expected_manifest_sha256,
                            "progress": _record_mapping(
                                self.state.progress(self.run_id)
                            ),
                            "duplicate": True,
                        },
                    )

                archive = self.incoming_dir / f"{attempt_id}.zip"
                if not archive.is_file():
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        "artifact has not been uploaded",
                    )
                if _sha256_file(archive) != expected_archive_sha256:
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        "staged artifact SHA-256 mismatch",
                    )

                self._state_call(
                    self.state.heartbeat,
                    self.run_id,
                    worker_id,
                    attempt_id,
                    lease_token,
                    lease_seconds=self.lease_seconds,
                    phase="verifying_artifact",
                )
                worker = self._state_call(
                    self.state.get_worker,
                    self.run_id,
                    worker_id,
                )
                if str(worker.get("current_attempt_id", "")) != attempt_id:
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        "worker no longer owns the uploaded attempt",
                    )
                expected_case_id = str(worker.get("current_case_id", ""))
                artifact_path, manifest_sha256 = self._accept_archive(
                    attempt_id,
                    archive,
                    expected_manifest_sha256,
                    expected_case_id=expected_case_id,
                )
                self._state_call(
                    self.state.complete_task,
                    self.run_id,
                    worker_id,
                    attempt_id,
                    lease_token,
                    artifact_path=str(artifact_path),
                    manifest_sha256=manifest_sha256,
                )
                receipt_payload = {
                    "run_id": self.run_id,
                    "attempt_id": attempt_id,
                    "worker_id": worker_id,
                    "lease_token_sha256": hashlib.sha256(
                        lease_token.encode("utf-8")
                    ).hexdigest(),
                    "archive_sha256": expected_archive_sha256,
                    "manifest_sha256": manifest_sha256,
                    "artifact_path": str(artifact_path),
                    "case_id": expected_case_id,
                }
                _write_json_atomic(
                    self._completion_receipt_path(attempt_id),
                    receipt_payload,
                )
                archive.unlink(missing_ok=True)
                return self._reply(
                    request,
                    "completed_ack",
                    {
                        "artifact_path": str(artifact_path),
                        "manifest_sha256": manifest_sha256,
                        "progress": _record_mapping(
                            self.state.progress(self.run_id)
                        ),
                    },
                )

        raise ApiError(HTTPStatus.BAD_REQUEST, f"unknown Maid message: {message_type}")

    def handle_artifact(
        self,
        attempt_id: str,
        worker_id: str,
        lease_token: str,
        temporary_path: Path,
        sha256: str,
        size: int,
    ) -> dict[str, Any]:
        attempt_id = _safe_identifier(attempt_id, "attempt_id")
        worker_id = _safe_identifier(worker_id, "worker_id")
        if not _SHA256.fullmatch(str(sha256)):
            raise ApiError(HTTPStatus.BAD_REQUEST, "artifact SHA-256 is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "artifact size is invalid")
        if not temporary_path.is_file():
            raise ApiError(HTTPStatus.BAD_REQUEST, "uploaded artifact is missing")
        actual_size = temporary_path.stat().st_size
        actual_sha256 = _sha256_file(temporary_path)
        if actual_size != size or not hmac.compare_digest(
            actual_sha256,
            str(sha256).lower(),
        ):
            raise ApiError(
                HTTPStatus.CONFLICT,
                "uploaded artifact size or SHA-256 does not match",
            )

        with self._attempt_lock(attempt_id):
            # Recheck committed and staged state only after acquiring the attempt
            # lock.  This makes identical upload retries safe while completion or
            # another upload is in flight.
            if self._completion_receipt_path(attempt_id).is_file():
                raise ApiError(HTTPStatus.CONFLICT, "attempt is already completed")

            # Validate and renew the worker-bound lease before touching an
            # existing staged artifact.  A spoofed worker must never overwrite it.
            self._state_call(
                self.state.heartbeat,
                self.run_id,
                worker_id,
                attempt_id,
                lease_token,
                lease_seconds=self.lease_seconds,
                phase="uploading_artifact",
            )
            destination = self.incoming_dir / f"{attempt_id}.zip"
            if destination.exists():
                if (
                    destination.stat().st_size != actual_size
                    or not hmac.compare_digest(
                        _sha256_file(destination),
                        actual_sha256,
                    )
                ):
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        "attempt already has a different staged artifact",
                    )
                temporary_path.unlink(missing_ok=True)
            else:
                os.replace(temporary_path, destination)
            self._state_call(
                self.state.mark_artifact_ready,
                self.run_id,
                worker_id,
                attempt_id,
                lease_token,
                artifact_path=str(destination),
                manifest_sha256="",
            )
            return {
                "sha256": actual_sha256,
                "size": actual_size,
                "status": "staged",
            }

    def _accept_archive(
        self,
        attempt_id: str,
        archive: Path,
        expected_manifest_sha256: str,
        *,
        expected_case_id: str,
    ) -> tuple[Path, str]:
        with tempfile.TemporaryDirectory(
            prefix=f"extract-{attempt_id}-",
            dir=self.incoming_dir,
        ) as temporary_name:
            temporary = Path(temporary_name)
            try:
                with zipfile.ZipFile(archive) as handle:
                    members = handle.infolist()
                    if len(members) > MAX_ARCHIVE_FILES:
                        raise ApiError(
                            HTTPStatus.BAD_REQUEST,
                            "artifact contains too many files",
                        )
                    if sum(member.file_size for member in members) > MAX_EXTRACTED_BYTES:
                        raise ApiError(
                            HTTPStatus.BAD_REQUEST,
                            "artifact expands beyond safety limit",
                        )

                    seen_paths: set[str] = set()
                    extracted_bytes = 0
                    for member in members:
                        relative = _safe_relative_path(
                            member.filename,
                            "artifact member path",
                        )
                        normalized = "/".join(relative.parts).casefold()
                        if normalized in seen_paths:
                            raise ApiError(
                                HTTPStatus.BAD_REQUEST,
                                f"artifact contains a duplicate path: {relative}",
                            )
                        seen_paths.add(normalized)
                        destination = temporary.joinpath(*relative.parts)
                        try:
                            destination.resolve().relative_to(temporary.resolve())
                        except ValueError as exc:
                            raise ApiError(
                                HTTPStatus.BAD_REQUEST,
                                "artifact member escapes extraction directory",
                            ) from exc
                        if member.is_dir():
                            destination.mkdir(parents=True, exist_ok=True)
                            continue
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with handle.open(member) as source, destination.open(
                            "wb"
                        ) as target:
                            while chunk := source.read(1024 * 1024):
                                extracted_bytes += len(chunk)
                                if extracted_bytes > MAX_EXTRACTED_BYTES:
                                    raise ApiError(
                                        HTTPStatus.BAD_REQUEST,
                                        "artifact expands beyond safety limit",
                                    )
                                target.write(chunk)
            except zipfile.BadZipFile as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, "artifact is not a valid ZIP") from exc

            manifest_path = temporary / case_runner.MANIFEST_FILENAME
            if not manifest_path.is_file():
                raise ApiError(HTTPStatus.BAD_REQUEST, "artifact has no manifest.json")
            manifest_sha256 = _sha256_file(manifest_path)
            if manifest_sha256 != expected_manifest_sha256:
                raise ApiError(HTTPStatus.CONFLICT, "manifest SHA-256 mismatch")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "artifact manifest is not valid UTF-8 JSON",
                ) from exc
            if not isinstance(manifest, Mapping):
                raise ApiError(HTTPStatus.BAD_REQUEST, "artifact manifest is invalid")
            if manifest.get("schema_version") != case_runner.MANIFEST_SCHEMA_VERSION:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "artifact manifest schema_version is unsupported",
                )
            status = manifest.get("status")
            if status not in {"completed", "dry_run"}:
                raise ApiError(HTTPStatus.BAD_REQUEST, "artifact manifest is invalid")
            dry_run = manifest.get("dry_run")
            if not isinstance(dry_run, bool) or dry_run != (status == "dry_run"):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "artifact manifest status and dry_run disagree",
                )
            case_id = _safe_identifier(manifest.get("case_id", ""), "manifest case_id")
            if case_id != expected_case_id:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"manifest case_id {case_id!r} does not match leased case "
                    f"{expected_case_id!r}",
                )
            self._verify_manifest_artifacts(temporary, manifest)

            final_directory = self.results_dir / (
                f"case_{int(case_id):04d}" if case_id.isdecimal() else f"case_{case_id}"
            )
            if final_directory.exists():
                existing_manifest = final_directory / case_runner.MANIFEST_FILENAME
                if (
                    existing_manifest.is_file()
                    and _sha256_file(existing_manifest) == manifest_sha256
                ):
                    return final_directory, manifest_sha256
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"result directory already exists with different content: {final_directory}",
                )
            os.replace(temporary, final_directory)
            return final_directory, manifest_sha256

    @staticmethod
    def _verify_manifest_artifacts(
        root: Path,
        manifest: Mapping[str, Any],
    ) -> None:
        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            raise ApiError(HTTPStatus.BAD_REQUEST, "manifest artifacts must be an object")
        status = manifest.get("status")
        simulation_mode = manifest.get(
            "simulation_mode",
            case_runner.DEFAULT_SIMULATION_MODE,
        )
        required_by_mode = {
            case_runner.DEFAULT_SIMULATION_MODE: {
                "s11",
                "rad_eff",
                "tot_eff",
                "farfield_source",
            },
            "propagation_s21": {"s21"},
        }
        if simulation_mode not in required_by_mode:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"manifest simulation_mode is unsupported: {simulation_mode!r}",
            )
        required = required_by_mode[simulation_mode] if status == "completed" else set()
        missing = required - set(artifacts)
        if missing:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"manifest is missing required artifacts: {sorted(missing)}",
            )
        if status == "dry_run" and artifacts:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "dry-run manifest must not claim result artifacts",
            )

        seen_paths: set[str] = set()
        for name, raw_record in artifacts.items():
            if not isinstance(raw_record, Mapping):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"manifest artifact {name!r} is invalid",
                )
            relative = _safe_relative_path(
                raw_record.get("path"),
                "manifest artifact path",
            )
            normalized = "/".join(relative.parts).casefold()
            if normalized in seen_paths:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "manifest artifacts must reference distinct files",
                )
            seen_paths.add(normalized)
            path = root.joinpath(*relative.parts)
            if not path.is_file():
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"manifest artifact is missing: {relative}",
                )
            expected_size = raw_record.get("size_bytes")
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
            ):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"manifest artifact size is invalid: {relative}",
                )
            if path.stat().st_size != expected_size:
                raise ApiError(HTTPStatus.CONFLICT, f"artifact size mismatch: {relative}")
            expected_sha256 = raw_record.get("sha256")
            if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(
                expected_sha256
            ):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"manifest artifact SHA-256 is invalid: {relative}",
                )
            if not hmac.compare_digest(
                _sha256_file(path),
                expected_sha256.lower(),
            ):
                raise ApiError(HTTPStatus.CONFLICT, f"artifact SHA-256 mismatch: {relative}")
