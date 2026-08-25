"""Durable task state for the Princess distributed simulation coordinator.

Princess is the only logical writer.  SQLite transactions make task claims and
lease transitions safe even when the HTTP server handles several Maid requests
concurrently.  Timestamps are Unix seconds so tests and recovery code can
inject a deterministic clock.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = 1
TASK_STATUSES = ("pending", "running", "completed", "failed")
ACTIVE_ATTEMPT_STATUSES = ("running", "artifact_ready")
BLOCKED_WORKER_STATUSES = ("unhealthy", "quarantined", "stopped", "offline")


class StateError(RuntimeError):
    """Base exception for invalid coordinator state transitions."""


class RunMismatchError(StateError):
    """Raised when a run is resumed with a different frozen CSV."""


class UnknownRunError(StateError):
    """Raised when a run id has not been initialized."""


class UnknownWorkerError(StateError):
    """Raised when a Maid has not registered with the run."""


class LeaseError(StateError):
    """Raised when an attempt no longer owns the task lease."""


class WorkerUnavailableError(StateError):
    """Raised when an unhealthy, quarantined, or stopped Maid requests work."""


@dataclass(frozen=True)
class FrozenCsvRow:
    """Stable identity for one row in a frozen sampling CSV."""

    case_id: str
    row_index: int
    csv_line: int
    row_sha256: str


@dataclass(frozen=True)
class FrozenCsv:
    """Validated immutable worklist description."""

    path: Path
    sha256: str
    row_count: int
    fieldnames: tuple[str, ...]
    case_id_column: str | None
    rows: tuple[FrozenCsvRow, ...]


@dataclass(frozen=True)
class ClaimedTask:
    """A lease granted to one Maid."""

    run_id: str
    case_id: str
    row_index: int
    csv_line: int
    row_sha256: str
    attempt_id: str
    attempt_number: int
    lease_token: str
    lease_expires_at: float

    @property
    def csv_row_index(self) -> int:
        """Compatibility name matching the database column."""

        return self.row_index


@dataclass(frozen=True)
class Progress:
    """User-visible run progress counts."""

    total: int
    pending: int
    running: int
    completed: int
    failed: int

    @property
    def finished(self) -> int:
        return self.completed + self.failed

    @property
    def is_terminal(self) -> bool:
        return self.total == self.finished

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "total": self.total,
            "pending": self.pending,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "finished": self.finished,
            "is_terminal": self.is_terminal,
        }


@dataclass(frozen=True)
class FailureRecord:
    """One failed simulation attempt returned in a Maid health bundle."""

    attempt_id: str
    case_id: str
    worker_id: str
    error_kind: str
    error_message: str
    traceback_text: str
    log_path: str
    started_at: float
    finished_at: float


@dataclass(frozen=True)
class FailureOutcome:
    """Result of atomically recording one failed attempt."""

    case_id: str
    task_status: str
    attempt_count: int
    consecutive_errors: int
    threshold_reached: bool
    streak_reached: bool
    failure_bundle: tuple[FailureRecord, ...]


@dataclass(frozen=True)
class RestartDecision:
    """Whether Princess may automatically wake a failed Maid."""

    worker_id: str
    should_restart: bool
    status: str
    restart_count: int
    automatic_restarts_without_success: int


@dataclass(frozen=True)
class EventRecord:
    """Append-only audit event."""

    event_id: int
    run_id: str
    created_at: float
    event_type: str
    worker_id: str | None
    case_id: str | None
    attempt_id: str | None
    payload: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_sha256(fieldnames: tuple[str, ...], row: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        [[name, row[name]] for name in fieldnames],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _result_directory_identity(case_id: str) -> str:
    """Return the Windows-insensitive identity used by Princess results."""

    directory_name = (
        f"case_{int(case_id):04d}"
        if case_id.isdecimal()
        else f"case_{case_id}"
    )
    # Win32 treats names case-insensitively and discards terminal spaces/dots.
    return directory_name.rstrip(" .").casefold()


def validate_frozen_csv(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    case_id_column: str = "sample_id",
) -> FrozenCsv:
    """Validate a sampling CSV and derive stable case and row identities."""

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"sampling CSV does not exist: {csv_path}")
    digest = sha256_file(csv_path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise RunMismatchError(
            f"CSV SHA-256 mismatch for {csv_path}: expected {expected_sha256}, got {digest}"
        )

    rows: list[FrozenCsvRow] = []
    seen_case_ids: set[str] = set()
    seen_result_identities: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"sampling CSV has no header: {csv_path}")
        fieldnames = tuple(reader.fieldnames)
        if not fieldnames:
            raise ValueError(f"sampling CSV has an empty header: {csv_path}")
        if any(not isinstance(name, str) or not name.strip() for name in fieldnames):
            raise ValueError("sampling CSV contains an empty column name")
        if len(set(fieldnames)) != len(fieldnames):
            raise ValueError("sampling CSV contains duplicate column names")
        resolved_id_column = case_id_column if case_id_column in fieldnames else None

        for row_index, row in enumerate(reader):
            if None in row:
                raise ValueError(
                    f"sampling CSV line {row_index + 2} contains extra columns"
                )
            missing = [name for name in fieldnames if row.get(name) is None]
            if missing:
                raise ValueError(
                    f"sampling CSV line {row_index + 2} is missing columns: "
                    f"{', '.join(missing)}"
                )
            raw_case_id = (
                str(row[resolved_id_column]).strip()
                if resolved_id_column is not None
                else str(row_index)
            )
            if not raw_case_id:
                raise ValueError(
                    f"sampling CSV line {row_index + 2} has an empty case id"
                )
            if raw_case_id in seen_case_ids:
                raise ValueError(f"duplicate case id in sampling CSV: {raw_case_id!r}")
            seen_case_ids.add(raw_case_id)
            result_identity = _result_directory_identity(raw_case_id)
            conflicting_case_id = seen_result_identities.get(result_identity)
            if conflicting_case_id is not None:
                raise ValueError(
                    f"case ids {conflicting_case_id!r} and {raw_case_id!r} map "
                    "to the same Princess result directory on Windows"
                )
            seen_result_identities[result_identity] = raw_case_id
            rows.append(
                FrozenCsvRow(
                    case_id=raw_case_id,
                    row_index=row_index,
                    csv_line=row_index + 2,
                    row_sha256=_row_sha256(fieldnames, row),
                )
            )

    return FrozenCsv(
        path=csv_path,
        sha256=digest,
        row_count=len(rows),
        fieldnames=fieldnames,
        case_id_column=resolved_id_column,
        rows=tuple(rows),
    )


def freeze_csv(
    source: str | Path,
    destination: str | Path,
    *,
    case_id_column: str = "sample_id",
    overwrite: bool = False,
) -> FrozenCsv:
    """Copy a sampling CSV atomically, then validate the frozen copy.

    Repeating the operation is idempotent when the existing destination has the
    same digest.  A different existing file is never overwritten implicitly.
    """

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    source_info = validate_frozen_csv(source_path, case_id_column=case_id_column)
    if source_path == destination_path:
        return source_info

    if destination_path.exists():
        destination_digest = sha256_file(destination_path)
        if destination_digest == source_info.sha256:
            return validate_frozen_csv(
                destination_path,
                expected_sha256=source_info.sha256,
                case_id_column=case_id_column,
            )
        if not overwrite:
            raise FileExistsError(
                f"frozen CSV already exists with different content: {destination_path}"
            )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.{uuid4().hex}.part"
    )
    try:
        shutil.copyfile(source_path, temporary_path)
        with temporary_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(temporary_path) != source_info.sha256:
            raise OSError("frozen CSV copy failed SHA-256 verification")
        os.replace(temporary_path, destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return validate_frozen_csv(
        destination_path,
        expected_sha256=source_info.sha256,
        case_id_column=case_id_column,
    )


class PrincessState:
    """SQLite-backed run, worker, task, attempt, and event ledger."""

    def __init__(self, database_path: str | Path, *, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("SQLite timeout must be positive")
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=timeout,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()

    def __enter__(self) -> PrincessState:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                schema_version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                csv_path TEXT NOT NULL,
                csv_sha256 TEXT NOT NULL,
                csv_row_count INTEGER NOT NULL,
                csv_fieldnames_json TEXT NOT NULL,
                case_id_column TEXT,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_control (
                run_id TEXT PRIMARY KEY,
                stop_requested INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tasks (
                run_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                csv_row_index INTEGER NOT NULL,
                csv_line INTEGER NOT NULL,
                row_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                worker_id TEXT,
                attempt_id TEXT,
                lease_token TEXT,
                lease_expires_at REAL,
                last_error_kind TEXT NOT NULL DEFAULT '',
                last_error_message TEXT NOT NULL DEFAULT '',
                artifact_path TEXT NOT NULL DEFAULT '',
                manifest_sha256 TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL,
                PRIMARY KEY (run_id, case_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS workers (
                run_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                host TEXT NOT NULL,
                transport TEXT NOT NULL,
                status TEXT NOT NULL,
                current_case_id TEXT,
                current_attempt_id TEXT,
                consecutive_errors INTEGER NOT NULL DEFAULT 0,
                restart_count INTEGER NOT NULL DEFAULT 0,
                automatic_restarts_without_success INTEGER NOT NULL DEFAULT 0,
                last_seen REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, worker_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                lease_token TEXT NOT NULL,
                status TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT '',
                lease_expires_at REAL NOT NULL,
                started_at REAL NOT NULL,
                last_heartbeat REAL NOT NULL,
                finished_at REAL,
                counts_toward_streak INTEGER NOT NULL DEFAULT 0,
                error_kind TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                traceback_text TEXT NOT NULL DEFAULT '',
                log_path TEXT NOT NULL DEFAULT '',
                artifact_path TEXT NOT NULL DEFAULT '',
                manifest_sha256 TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (run_id, case_id)
                    REFERENCES tasks(run_id, case_id) ON DELETE CASCADE,
                FOREIGN KEY (run_id, worker_id)
                    REFERENCES workers(run_id, worker_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                event_type TEXT NOT NULL,
                worker_id TEXT,
                case_id TEXT,
                attempt_id TEXT,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_claim
                ON tasks(run_id, status, csv_row_index);
            CREATE INDEX IF NOT EXISTS idx_tasks_lease
                ON tasks(run_id, status, lease_expires_at);
            CREATE INDEX IF NOT EXISTS idx_attempts_worker
                ON attempts(run_id, worker_id, started_at);
            CREATE INDEX IF NOT EXISTS idx_events_run
                ON events(run_id, event_id);
            """
        )
        row = self._connection.execute(
            "SELECT schema_version FROM schema_info"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO schema_info(schema_version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        elif int(row["schema_version"]) != SCHEMA_VERSION:
            raise StateError(
                f"unsupported state schema {row['schema_version']}; "
                f"expected {SCHEMA_VERSION}"
            )

    @staticmethod
    def _now(now: float | None) -> float:
        resolved = time.time() if now is None else float(now)
        if not resolved == resolved or resolved in (float("inf"), float("-inf")):
            raise ValueError("timestamp must be finite")
        return resolved

    @staticmethod
    def _positive_seconds(value: float, label: str) -> float:
        resolved = float(value)
        if resolved <= 0 or not resolved < float("inf"):
            raise ValueError(f"{label} must be a positive finite number")
        return resolved

    @staticmethod
    def _require_text(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
        return value.strip()

    def _require_run_locked(self, connection: sqlite3.Connection, run_id: str) -> None:
        if connection.execute(
            "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone() is None:
            raise UnknownRunError(f"unknown run: {run_id}")

    def _require_worker_locked(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        worker_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM workers WHERE run_id = ? AND worker_id = ?",
            (run_id, worker_id),
        ).fetchone()
        if row is None:
            raise UnknownWorkerError(f"unknown worker {worker_id!r} for run {run_id!r}")
        return row

    def _event_locked(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        event_type: str,
        *,
        now: float,
        worker_id: str | None = None,
        case_id: str | None = None,
        attempt_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(
                run_id, created_at, event_type, worker_id, case_id,
                attempt_id, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now,
                event_type,
                worker_id,
                case_id,
                attempt_id,
                json.dumps(
                    dict(payload or {}),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                ),
            ),
        )

    def initialize_run(
        self,
        run_id: str,
        frozen_csv: FrozenCsv,
        *,
        metadata: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> bool:
        """Create all tasks once, or verify an idempotent resume."""

        run_id = self._require_text(run_id, "run_id")
        timestamp = self._now(now)
        metadata_json = json.dumps(
            dict(metadata or {}),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["csv_sha256"] != frozen_csv.sha256
                    or int(existing["csv_row_count"]) != frozen_csv.row_count
                ):
                    raise RunMismatchError(
                        f"run {run_id!r} was initialized from a different CSV"
                    )
                stored_tasks = connection.execute(
                    "SELECT case_id, row_sha256 FROM tasks WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
                expected = {row.case_id: row.row_sha256 for row in frozen_csv.rows}
                actual = {row["case_id"]: row["row_sha256"] for row in stored_tasks}
                if actual != expected:
                    raise RunMismatchError(
                        f"run {run_id!r} task rows do not match its frozen CSV"
                    )
                return False

            connection.execute(
                """
                INSERT INTO runs(
                    run_id, csv_path, csv_sha256, csv_row_count,
                    csv_fieldnames_json, case_id_column, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(frozen_csv.path),
                    frozen_csv.sha256,
                    frozen_csv.row_count,
                    json.dumps(frozen_csv.fieldnames, ensure_ascii=False),
                    frozen_csv.case_id_column,
                    metadata_json,
                    timestamp,
                ),
            )
            connection.executemany(
                """
                INSERT INTO tasks(
                    run_id, case_id, csv_row_index, csv_line, row_sha256,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                [
                    (
                        run_id,
                        row.case_id,
                        row.row_index,
                        row.csv_line,
                        row.row_sha256,
                        timestamp,
                        timestamp,
                    )
                    for row in frozen_csv.rows
                ],
            )
            self._event_locked(
                connection,
                run_id,
                "run_initialized",
                now=timestamp,
                payload={
                    "csv_sha256": frozen_csv.sha256,
                    "row_count": frozen_csv.row_count,
                },
            )
            connection.execute(
                """
                INSERT INTO run_control(run_id, stop_requested, reason, updated_at)
                VALUES (?, 0, '', ?)
                """,
                (run_id, timestamp),
            )
        return True

    def request_stop(
        self,
        run_id: str,
        *,
        reason: str,
        now: float | None = None,
    ) -> tuple[str, ...]:
        """Stop scheduling and refund every currently leased attempt."""

        run_id = self._require_text(run_id, "run_id")
        reason = self._require_text(reason, "reason")
        timestamp = self._now(now)
        with self._transaction() as connection:
            self._require_run_locked(connection, run_id)
            active = connection.execute(
                """
                SELECT t.case_id, t.attempt_id
                FROM tasks AS t
                JOIN attempts AS a ON a.attempt_id = t.attempt_id
                WHERE t.run_id = ? AND t.status = 'running'
                    AND a.status IN ('running', 'artifact_ready')
                ORDER BY t.csv_row_index
                """,
                (run_id,),
            ).fetchall()
            for task in active:
                connection.execute(
                    """
                    UPDATE attempts
                    SET status = 'released', finished_at = ?,
                        error_kind = 'Released', error_message = ?
                    WHERE attempt_id = ? AND status IN ('running', 'artifact_ready')
                    """,
                    (timestamp, reason, task["attempt_id"]),
                )
                connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'pending',
                        attempt_count = CASE
                            WHEN attempt_count > 0 THEN attempt_count - 1
                            ELSE 0
                        END,
                        worker_id = NULL, attempt_id = NULL,
                        lease_token = NULL, lease_expires_at = NULL,
                        last_error_kind = 'Released', last_error_message = ?,
                        updated_at = ?
                    WHERE run_id = ? AND case_id = ?
                    """,
                    (reason, timestamp, run_id, task["case_id"]),
                )
            connection.execute(
                """
                UPDATE workers
                SET status = 'stopped', current_case_id = NULL,
                    current_attempt_id = NULL, updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, run_id),
            )
            connection.execute(
                """
                INSERT INTO run_control(run_id, stop_requested, reason, updated_at)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    stop_requested = 1,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at
                """,
                (run_id, reason, timestamp),
            )
            case_ids = tuple(str(task["case_id"]) for task in active)
            self._event_locked(
                connection,
                run_id,
                "run_stop_requested",
                now=timestamp,
                payload={"reason": reason, "released_case_ids": list(case_ids)},
            )
        return case_ids

    def stop_request(self, run_id: str) -> str | None:
        """Return the durable stop reason, if emergency dismissal is active."""

        run_id = self._require_text(run_id, "run_id")
        with self._lock:
            self._require_run_locked(self._connection, run_id)
            row = self._connection.execute(
                """
                SELECT stop_requested, reason FROM run_control WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None or not bool(row["stop_requested"]):
            return None
        return str(row["reason"])

    def resume_run(self, run_id: str, *, now: float | None = None) -> bool:
        """Clear an earlier emergency stop before an explicit new start."""

        run_id = self._require_text(run_id, "run_id")
        timestamp = self._now(now)
        with self._transaction() as connection:
            self._require_run_locked(connection, run_id)
            row = connection.execute(
                """
                SELECT stop_requested, reason FROM run_control WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None or not bool(row["stop_requested"]):
                return False
            previous_reason = str(row["reason"])
            connection.execute(
                """
                UPDATE run_control
                SET stop_requested = 0, reason = '', updated_at = ?
                WHERE run_id = ?
                """,
                (timestamp, run_id),
            )
            connection.execute(
                """
                UPDATE workers
                SET status = 'offline', consecutive_errors = 0,
                    automatic_restarts_without_success = 0, updated_at = ?
                WHERE run_id = ? AND status = 'stopped'
                """,
                (timestamp, run_id),
            )
            self._event_locked(
                connection,
                run_id,
                "run_resumed_after_stop",
                now=timestamp,
                payload={"previous_reason": previous_reason},
            )
        return True

    def register_worker(
        self,
        run_id: str,
        worker_id: str,
        *,
        host: str = "",
        transport: str = "http",
        metadata: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Register a Maid idempotently without reviving a quarantined worker."""

        run_id = self._require_text(run_id, "run_id")
        worker_id = self._require_text(worker_id, "worker_id")
        transport = self._require_text(transport, "transport")
        timestamp = self._now(now)
        metadata_json = json.dumps(
            dict(metadata or {}),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        with self._transaction() as connection:
            self._require_run_locked(connection, run_id)
            existing = connection.execute(
                "SELECT 1 FROM workers WHERE run_id = ? AND worker_id = ?",
                (run_id, worker_id),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO workers(
                        run_id, worker_id, host, transport, status, last_seen,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'idle', ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        worker_id,
                        str(host),
                        transport,
                        timestamp,
                        metadata_json,
                        timestamp,
                        timestamp,
                    ),
                )
                event_type = "worker_registered"
            else:
                connection.execute(
                    """
                    UPDATE workers
                    SET host = ?, transport = ?, metadata_json = ?,
                        last_seen = ?, updated_at = ?
                    WHERE run_id = ? AND worker_id = ?
                    """,
                    (
                        str(host),
                        transport,
                        metadata_json,
                        timestamp,
                        timestamp,
                        run_id,
                        worker_id,
                    ),
                )
                event_type = "worker_reconnected"
            self._event_locked(
                connection,
                run_id,
                event_type,
                now=timestamp,
                worker_id=worker_id,
                payload={"host": str(host), "transport": transport},
            )
            row = self._require_worker_locked(connection, run_id, worker_id)
        return dict(row)

    def mark_worker_ready(
        self,
        run_id: str,
        worker_id: str,
        *,
        now: float | None = None,
    ) -> None:
        """Mark a newly launched or restarted Maid ready to request work."""

        timestamp = self._now(now)
        with self._transaction() as connection:
            worker = self._require_worker_locked(connection, run_id, worker_id)
            if worker["current_attempt_id"] is not None:
                if worker["status"] != "busy":
                    raise StateError(
                        f"worker {worker_id!r} owns an attempt while {worker['status']}"
                    )
                connection.execute(
                    """
                    UPDATE workers SET last_seen = ?, updated_at = ?
                    WHERE run_id = ? AND worker_id = ?
                    """,
                    (timestamp, timestamp, run_id, worker_id),
                )
                return
            if worker["status"] in {"unhealthy", "quarantined", "stopped"}:
                raise WorkerUnavailableError(
                    f"worker {worker_id!r} is {worker['status']} and must be woken explicitly"
                )
            connection.execute(
                """
                UPDATE workers SET status = 'idle', last_seen = ?, updated_at = ?
                WHERE run_id = ? AND worker_id = ?
                """,
                (timestamp, timestamp, run_id, worker_id),
            )
            self._event_locked(
                connection,
                run_id,
                "worker_ready",
                now=timestamp,
                worker_id=worker_id,
            )

    def claim_next(
        self,
        run_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 90.0,
        max_attempts: int = 3,
        exclude_case_ids: Sequence[str] = (),
        now: float | None = None,
    ) -> ClaimedTask | None:
        """Atomically claim the lowest-index pending task for a Maid."""

        timestamp = self._now(now)
        lease_duration = self._positive_seconds(lease_seconds, "lease_seconds")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        exclusions = tuple(dict.fromkeys(str(case_id) for case_id in exclude_case_ids))
        with self._transaction() as connection:
            self._require_run_locked(connection, run_id)
            self._requeue_expired_locked(
                connection,
                run_id,
                timestamp,
                max_attempts=max_attempts,
            )
            exhausted = connection.execute(
                """
                SELECT * FROM tasks
                WHERE run_id = ? AND status = 'pending' AND attempt_count >= ?
                ORDER BY csv_row_index
                """,
                (run_id, max_attempts),
            ).fetchall()
            for task in exhausted:
                connection.execute(
                    """
                    UPDATE tasks SET status = 'failed', updated_at = ?
                    WHERE run_id = ? AND case_id = ? AND status = 'pending'
                    """,
                    (timestamp, run_id, task["case_id"]),
                )
                self._event_locked(
                    connection,
                    run_id,
                    "task_attempts_exhausted",
                    now=timestamp,
                    case_id=str(task["case_id"]),
                    payload={"attempt_count": int(task["attempt_count"])},
                )
            worker = self._require_worker_locked(connection, run_id, worker_id)
            if worker["status"] in BLOCKED_WORKER_STATUSES:
                raise WorkerUnavailableError(
                    f"worker {worker_id!r} cannot claim while {worker['status']}"
                )

            current = connection.execute(
                """
                SELECT t.*, a.attempt_number, a.status AS attempt_status
                FROM tasks AS t
                JOIN attempts AS a ON a.attempt_id = t.attempt_id
                WHERE t.run_id = ? AND t.worker_id = ? AND t.status = 'running'
                ORDER BY t.csv_row_index LIMIT 1
                """,
                (run_id, worker_id),
            ).fetchone()
            if current is not None:
                return self._claimed_task_from_row(current)

            claim_sql = (
                "SELECT * FROM tasks "
                "WHERE run_id = ? AND status = 'pending'"
            )
            claim_parameters: list[Any] = [run_id]
            if exclusions:
                placeholders = ",".join("?" for _ in exclusions)
                claim_sql += f" AND case_id NOT IN ({placeholders})"
                claim_parameters.extend(exclusions)
            claim_sql += " ORDER BY csv_row_index, case_id LIMIT 1"
            task = connection.execute(claim_sql, claim_parameters).fetchone()
            if task is None:
                connection.execute(
                    """
                    UPDATE workers
                    SET status = 'idle', current_case_id = NULL,
                        current_attempt_id = NULL, last_seen = ?, updated_at = ?
                    WHERE run_id = ? AND worker_id = ?
                    """,
                    (timestamp, timestamp, run_id, worker_id),
                )
                return None

            attempt_id = uuid4().hex
            lease_token = uuid4().hex
            attempt_number = int(task["attempt_count"]) + 1
            expires_at = timestamp + lease_duration
            connection.execute(
                """
                INSERT INTO attempts(
                    attempt_id, run_id, case_id, worker_id, attempt_number,
                    lease_token, status, lease_expires_at, started_at,
                    last_heartbeat
                ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    attempt_id,
                    run_id,
                    task["case_id"],
                    worker_id,
                    attempt_number,
                    lease_token,
                    expires_at,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                UPDATE tasks
                SET status = 'running', attempt_count = ?, worker_id = ?,
                    attempt_id = ?, lease_token = ?, lease_expires_at = ?,
                    updated_at = ?
                WHERE run_id = ? AND case_id = ? AND status = 'pending'
                """,
                (
                    attempt_number,
                    worker_id,
                    attempt_id,
                    lease_token,
                    expires_at,
                    timestamp,
                    run_id,
                    task["case_id"],
                ),
            )
            connection.execute(
                """
                UPDATE workers
                SET status = 'busy', current_case_id = ?, current_attempt_id = ?,
                    last_seen = ?, updated_at = ?
                WHERE run_id = ? AND worker_id = ?
                """,
                (
                    task["case_id"],
                    attempt_id,
                    timestamp,
                    timestamp,
                    run_id,
                    worker_id,
                ),
            )
            self._event_locked(
                connection,
                run_id,
                "task_claimed",
                now=timestamp,
                worker_id=worker_id,
                case_id=task["case_id"],
                attempt_id=attempt_id,
                payload={
                    "attempt_number": attempt_number,
                    "lease_expires_at": expires_at,
                },
            )
            claimed = ClaimedTask(
                run_id=run_id,
                case_id=str(task["case_id"]),
                row_index=int(task["csv_row_index"]),
                csv_line=int(task["csv_line"]),
                row_sha256=str(task["row_sha256"]),
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                lease_token=lease_token,
                lease_expires_at=expires_at,
            )
        return claimed

    @staticmethod
    def _claimed_task_from_row(row: sqlite3.Row) -> ClaimedTask:
        return ClaimedTask(
            run_id=str(row["run_id"]),
            case_id=str(row["case_id"]),
            row_index=int(row["csv_row_index"]),
            csv_line=int(row["csv_line"]),
            row_sha256=str(row["row_sha256"]),
            attempt_id=str(row["attempt_id"]),
            attempt_number=int(row["attempt_number"]),
            lease_token=str(row["lease_token"]),
            lease_expires_at=float(row["lease_expires_at"]),
        )

    def _owned_attempt_locked(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        worker_id: str,
        attempt_id: str,
        lease_token: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT t.*, a.status AS attempt_status, a.attempt_number
            FROM tasks AS t
            JOIN attempts AS a ON a.attempt_id = t.attempt_id
            WHERE t.run_id = ? AND t.worker_id = ? AND t.attempt_id = ?
                AND t.lease_token = ? AND a.lease_token = ?
                AND t.status = 'running'
            """,
            (run_id, worker_id, attempt_id, lease_token, lease_token),
        ).fetchone()
        if row is None or row["attempt_status"] not in ACTIVE_ATTEMPT_STATUSES:
            raise LeaseError(
                f"attempt {attempt_id!r} does not own an active lease for worker "
                f"{worker_id!r}"
            )
        return row

    def _expire_task_locked(
        self,
        connection: sqlite3.Connection,
        task: sqlite3.Row,
        now: float,
        *,
        event_type: str = "lease_expired",
        reason: str = "lease expired",
        worker_status: str = "offline",
        max_attempts: int | None = None,
    ) -> None:
        task_status = (
            "failed"
            if max_attempts is not None
            and int(task["attempt_count"]) >= max_attempts
            else "pending"
        )
        connection.execute(
            """
            UPDATE attempts
            SET status = 'expired', finished_at = ?, error_kind = 'LeaseExpired',
                error_message = ?
            WHERE attempt_id = ? AND status IN ('running', 'artifact_ready')
            """,
            (now, reason, task["attempt_id"]),
        )
        connection.execute(
            """
            UPDATE tasks
            SET status = ?, worker_id = NULL, attempt_id = NULL,
                lease_token = NULL, lease_expires_at = NULL,
                last_error_kind = 'LeaseExpired', last_error_message = ?,
                updated_at = ?
            WHERE run_id = ? AND case_id = ? AND attempt_id = ?
            """,
            (
                task_status,
                reason,
                now,
                task["run_id"],
                task["case_id"],
                task["attempt_id"],
            ),
        )
        connection.execute(
            """
            UPDATE workers
            SET status = ?, current_case_id = NULL, current_attempt_id = NULL,
                updated_at = ?
            WHERE run_id = ? AND worker_id = ? AND current_attempt_id = ?
            """,
            (
                worker_status,
                now,
                task["run_id"],
                task["worker_id"],
                task["attempt_id"],
            ),
        )
        self._event_locked(
            connection,
            str(task["run_id"]),
            event_type,
            now=now,
            worker_id=str(task["worker_id"]),
            case_id=str(task["case_id"]),
            attempt_id=str(task["attempt_id"]),
            payload={
                "reason": reason,
                "task_status": task_status,
                "attempt_count": int(task["attempt_count"]),
            },
        )

    def _requeue_expired_locked(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        now: float,
        *,
        max_attempts: int | None = None,
    ) -> list[str]:
        expired = connection.execute(
            """
            SELECT t.* FROM tasks AS t
            JOIN attempts AS a ON a.attempt_id = t.attempt_id
            WHERE t.run_id = ? AND t.status = 'running'
                AND t.lease_expires_at <= ?
                AND a.status IN ('running', 'artifact_ready')
            ORDER BY t.csv_row_index
            """,
            (run_id, now),
        ).fetchall()
        for task in expired:
            self._expire_task_locked(
                connection,
                task,
                now,
                max_attempts=max_attempts,
            )
        return [str(task["case_id"]) for task in expired]

    def heartbeat(
        self,
        run_id: str,
        worker_id: str,
        attempt_id: str,
        lease_token: str,
        *,
        lease_seconds: float = 90.0,
        phase: str | None = None,
        now: float | None = None,
    ) -> float:
        """Renew an active lease and return its new expiry time."""

        timestamp = self._now(now)
        lease_duration = self._positive_seconds(lease_seconds, "lease_seconds")
        expired = False
        with self._transaction() as connection:
            task = self._owned_attempt_locked(
                connection, run_id, worker_id, attempt_id, lease_token
            )
            if float(task["lease_expires_at"]) <= timestamp:
                self._expire_task_locked(connection, task, timestamp)
                expired = True
                new_expiry = float(task["lease_expires_at"])
            else:
                new_expiry = timestamp + lease_duration
                connection.execute(
                    """
                    UPDATE tasks SET lease_expires_at = ?, updated_at = ?
                    WHERE run_id = ? AND case_id = ? AND attempt_id = ?
                    """,
                    (new_expiry, timestamp, run_id, task["case_id"], attempt_id),
                )
                connection.execute(
                    """
                    UPDATE attempts
                    SET lease_expires_at = ?, last_heartbeat = ?, phase = ?
                    WHERE attempt_id = ?
                    """,
                    (new_expiry, timestamp, phase or task["attempt_status"], attempt_id),
                )
                connection.execute(
                    """
                    UPDATE workers SET last_seen = ?, updated_at = ?
                    WHERE run_id = ? AND worker_id = ?
                    """,
                    (timestamp, timestamp, run_id, worker_id),
                )
        if expired:
            raise LeaseError(f"attempt {attempt_id!r} lease has expired")
        return new_expiry

    def mark_artifact_ready(
        self,
        run_id: str,
        worker_id: str,
        attempt_id: str,
        lease_token: str,
        *,
        artifact_path: str | Path,
        manifest_sha256: str = "",
        now: float | None = None,
    ) -> None:
        """Record remote completion while Princess verifies uploaded artifacts."""

        timestamp = self._now(now)
        artifact_text = self._require_text(str(artifact_path), "artifact_path")
        with self._transaction() as connection:
            task = self._owned_attempt_locked(
                connection, run_id, worker_id, attempt_id, lease_token
            )
            connection.execute(
                """
                UPDATE attempts
                SET status = 'artifact_ready', artifact_path = ?,
                    manifest_sha256 = ?, last_heartbeat = ?
                WHERE attempt_id = ?
                """,
                (artifact_text, str(manifest_sha256), timestamp, attempt_id),
            )
            connection.execute(
                """
                UPDATE tasks
                SET artifact_path = ?, manifest_sha256 = ?, updated_at = ?
                WHERE run_id = ? AND case_id = ? AND attempt_id = ?
                """,
                (
                    artifact_text,
                    str(manifest_sha256),
                    timestamp,
                    run_id,
                    task["case_id"],
                    attempt_id,
                ),
            )
            self._event_locked(
                connection,
                run_id,
                "artifact_ready",
                now=timestamp,
                worker_id=worker_id,
                case_id=str(task["case_id"]),
                attempt_id=attempt_id,
                payload={
                    "artifact_path": artifact_text,
                    "manifest_sha256": str(manifest_sha256),
                },
            )

    def complete_task(
        self,
        run_id: str,
        worker_id: str,
        attempt_id: str,
        lease_token: str,
        *,
        artifact_path: str | Path | None = None,
        manifest_sha256: str | None = None,
        now: float | None = None,
    ) -> None:
        """Atomically acknowledge artifacts and complete an owned task."""

        timestamp = self._now(now)
        with self._transaction() as connection:
            task = self._owned_attempt_locked(
                connection, run_id, worker_id, attempt_id, lease_token
            )
            resolved_artifact = (
                str(artifact_path) if artifact_path is not None else task["artifact_path"]
            )
            resolved_manifest = (
                str(manifest_sha256)
                if manifest_sha256 is not None
                else task["manifest_sha256"]
            )
            connection.execute(
                """
                UPDATE attempts
                SET status = 'completed', finished_at = ?, artifact_path = ?,
                    manifest_sha256 = ?
                WHERE attempt_id = ?
                """,
                (timestamp, resolved_artifact, resolved_manifest, attempt_id),
            )
            connection.execute(
                """
                UPDATE tasks
                SET status = 'completed', worker_id = NULL, attempt_id = NULL,
                    lease_token = NULL, lease_expires_at = NULL,
                    artifact_path = ?, manifest_sha256 = ?, updated_at = ?,
                    completed_at = ?
                WHERE run_id = ? AND case_id = ? AND attempt_id = ?
                """,
                (
                    resolved_artifact,
                    resolved_manifest,
                    timestamp,
                    timestamp,
                    run_id,
                    task["case_id"],
                    attempt_id,
                ),
            )
            connection.execute(
                """
                UPDATE workers
                SET status = 'idle', current_case_id = NULL,
                    current_attempt_id = NULL, consecutive_errors = 0,
                    automatic_restarts_without_success = 0,
                    last_seen = ?, updated_at = ?
                WHERE run_id = ? AND worker_id = ?
                """,
                (timestamp, timestamp, run_id, worker_id),
            )
            self._event_locked(
                connection,
                run_id,
                "task_completed",
                now=timestamp,
                worker_id=worker_id,
                case_id=str(task["case_id"]),
                attempt_id=attempt_id,
                payload={
                    "artifact_path": resolved_artifact,
                    "manifest_sha256": resolved_manifest,
                },
            )

    @staticmethod
    def _failure_record_from_row(row: sqlite3.Row) -> FailureRecord:
        return FailureRecord(
            attempt_id=str(row["attempt_id"]),
            case_id=str(row["case_id"]),
            worker_id=str(row["worker_id"]),
            error_kind=str(row["error_kind"]),
            error_message=str(row["error_message"]),
            traceback_text=str(row["traceback_text"]),
            log_path=str(row["log_path"]),
            started_at=float(row["started_at"]),
            finished_at=float(row["finished_at"]),
        )

    def _failure_bundle_locked(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        worker_id: str,
        limit: int,
    ) -> tuple[FailureRecord, ...]:
        worker = self._require_worker_locked(connection, run_id, worker_id)
        streak_length = min(int(worker["consecutive_errors"]), limit)
        if streak_length <= 0:
            return ()
        rows = connection.execute(
            """
            SELECT * FROM attempts
            WHERE run_id = ? AND worker_id = ? AND status = 'failed'
                AND counts_toward_streak = 1
            ORDER BY finished_at DESC, rowid DESC
            LIMIT ?
            """,
            (run_id, worker_id, streak_length),
        ).fetchall()
        return tuple(
            self._failure_record_from_row(row) for row in reversed(rows)
        )

    def fail_task(
        self,
        run_id: str,
        worker_id: str,
        attempt_id: str,
        lease_token: str,
        *,
        error_kind: str,
        error_message: str,
        traceback_text: str = "",
        log_path: str | Path = "",
        retryable: bool = True,
        max_attempts: int = 3,
        counts_toward_streak: bool = True,
        streak_threshold: int = 5,
        now: float | None = None,
    ) -> FailureOutcome:
        """Record one failure, requeue or terminate the task, and update health."""

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if streak_threshold < 1:
            raise ValueError("streak_threshold must be at least 1")
        error_kind = self._require_text(error_kind, "error_kind")
        error_message = self._require_text(error_message, "error_message")
        timestamp = self._now(now)
        with self._transaction() as connection:
            task = self._owned_attempt_locked(
                connection, run_id, worker_id, attempt_id, lease_token
            )
            attempt_count = int(task["attempt_count"])
            task_status = (
                "pending" if retryable and attempt_count < max_attempts else "failed"
            )
            connection.execute(
                """
                UPDATE attempts
                SET status = 'failed', finished_at = ?, counts_toward_streak = ?,
                    error_kind = ?, error_message = ?, traceback_text = ?,
                    log_path = ?
                WHERE attempt_id = ?
                """,
                (
                    timestamp,
                    int(counts_toward_streak),
                    error_kind,
                    error_message,
                    str(traceback_text),
                    str(log_path),
                    attempt_id,
                ),
            )
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, worker_id = NULL, attempt_id = NULL,
                    lease_token = NULL, lease_expires_at = NULL,
                    last_error_kind = ?, last_error_message = ?, updated_at = ?
                WHERE run_id = ? AND case_id = ? AND attempt_id = ?
                """,
                (
                    task_status,
                    error_kind,
                    error_message,
                    timestamp,
                    run_id,
                    task["case_id"],
                    attempt_id,
                ),
            )
            worker = self._require_worker_locked(connection, run_id, worker_id)
            consecutive_errors = int(worker["consecutive_errors"]) + int(
                counts_toward_streak
            )
            threshold_reached = (
                counts_toward_streak
                and consecutive_errors >= streak_threshold
            )
            worker_status = "unhealthy" if threshold_reached else "idle"
            connection.execute(
                """
                UPDATE workers
                SET status = ?, current_case_id = NULL,
                    current_attempt_id = NULL, consecutive_errors = ?,
                    last_seen = ?, updated_at = ?
                WHERE run_id = ? AND worker_id = ?
                """,
                (
                    worker_status,
                    consecutive_errors,
                    timestamp,
                    timestamp,
                    run_id,
                    worker_id,
                ),
            )
            self._event_locked(
                connection,
                run_id,
                "task_failed",
                now=timestamp,
                worker_id=worker_id,
                case_id=str(task["case_id"]),
                attempt_id=attempt_id,
                payload={
                    "error_kind": error_kind,
                    "error_message": error_message,
                    "retryable": bool(retryable),
                    "task_status": task_status,
                    "counts_toward_streak": bool(counts_toward_streak),
                    "consecutive_errors": consecutive_errors,
                    "threshold_reached": bool(threshold_reached),
                },
            )
            bundle = (
                self._failure_bundle_locked(
                    connection, run_id, worker_id, streak_threshold
                )
                if threshold_reached
                else ()
            )
        return FailureOutcome(
            case_id=str(task["case_id"]),
            task_status=task_status,
            attempt_count=attempt_count,
            consecutive_errors=consecutive_errors,
            threshold_reached=bool(threshold_reached),
            streak_reached=bool(threshold_reached),
            failure_bundle=bundle,
        )

    def release_task(
        self,
        run_id: str,
        worker_id: str,
        attempt_id: str,
        lease_token: str,
        *,
        reason: str,
        worker_status: str = "offline",
        now: float | None = None,
    ) -> None:
        """Return an unfinished task to pending after a controlled disconnect."""

        reason = self._require_text(reason, "reason")
        worker_status = self._require_text(worker_status, "worker_status")
        timestamp = self._now(now)
        with self._transaction() as connection:
            task = self._owned_attempt_locked(
                connection, run_id, worker_id, attempt_id, lease_token
            )
            connection.execute(
                """
                UPDATE attempts
                SET status = 'released', finished_at = ?,
                    error_kind = 'Released', error_message = ?
                WHERE attempt_id = ?
                """,
                (timestamp, reason, attempt_id),
            )
            connection.execute(
                """
                UPDATE tasks
                SET status = 'pending', worker_id = NULL, attempt_id = NULL,
                    lease_token = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE run_id = ? AND case_id = ? AND attempt_id = ?
                """,
                (timestamp, run_id, task["case_id"], attempt_id),
            )
            connection.execute(
                """
                UPDATE workers
                SET status = ?, current_case_id = NULL,
                    current_attempt_id = NULL, updated_at = ?
                WHERE run_id = ? AND worker_id = ?
                """,
                (worker_status, timestamp, run_id, worker_id),
            )
            self._event_locked(
                connection,
                run_id,
                "task_released",
                now=timestamp,
                worker_id=worker_id,
                case_id=str(task["case_id"]),
                attempt_id=attempt_id,
                payload={"reason": reason, "worker_status": worker_status},
            )

    def release_expired_leases(
        self,
        run_id: str,
        *,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> tuple[str, ...]:
        """Recover expired attempts, failing cases at the attempt limit."""

        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        timestamp = self._now(now)
        with self._transaction() as connection:
            self._require_run_locked(connection, run_id)
            case_ids = self._requeue_expired_locked(
                connection,
                run_id,
                timestamp,
                max_attempts=max_attempts,
            )
        return tuple(case_ids)

    def progress(self, run_id: str) -> Progress:
        """Return a consistent status count snapshot."""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT status, COUNT(*) AS count FROM tasks
                WHERE run_id = ? GROUP BY status
                """,
                (run_id,),
            ).fetchall()
            if self._connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone() is None:
                raise UnknownRunError(f"unknown run: {run_id}")
        counts = {status: 0 for status in TASK_STATUSES}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        return Progress(
            total=sum(counts.values()),
            pending=counts["pending"],
            running=counts["running"],
            completed=counts["completed"],
            failed=counts["failed"],
        )

    def failure_bundle(
        self,
        run_id: str,
        worker_id: str,
        *,
        limit: int = 5,
    ) -> tuple[FailureRecord, ...]:
        """Return the latest failures belonging to the current error streak."""

        if limit < 1:
            raise ValueError("failure bundle limit must be at least 1")
        with self._transaction() as connection:
            return self._failure_bundle_locked(connection, run_id, worker_id, limit)

    def prepare_worker_restart(
        self,
        run_id: str,
        worker_id: str,
        *,
        max_automatic_restarts_without_success: int = 1,
        now: float | None = None,
    ) -> RestartDecision:
        """Allow a bounded automatic restart, otherwise quarantine the Maid."""

        if max_automatic_restarts_without_success < 0:
            raise ValueError("automatic restart limit cannot be negative")
        timestamp = self._now(now)
        with self._transaction() as connection:
            worker = self._require_worker_locked(connection, run_id, worker_id)
            if worker["current_attempt_id"] is not None:
                raise StateError("cannot restart a worker that still owns an attempt")
            used = int(worker["automatic_restarts_without_success"])
            restart_count = int(worker["restart_count"])
            should_restart = used < max_automatic_restarts_without_success
            if should_restart:
                used += 1
                restart_count += 1
                status = "restarting"
                event_type = "worker_restart_scheduled"
            else:
                status = "quarantined"
                event_type = "worker_quarantined"
            connection.execute(
                """
                UPDATE workers
                SET status = ?, consecutive_errors = 0, restart_count = ?,
                    automatic_restarts_without_success = ?, updated_at = ?
                WHERE run_id = ? AND worker_id = ?
                """,
                (status, restart_count, used, timestamp, run_id, worker_id),
            )
            self._event_locked(
                connection,
                run_id,
                event_type,
                now=timestamp,
                worker_id=worker_id,
                payload={
                    "restart_count": restart_count,
                    "automatic_restarts_without_success": used,
                },
            )
        return RestartDecision(
            worker_id=worker_id,
            should_restart=should_restart,
            status=status,
            restart_count=restart_count,
            automatic_restarts_without_success=used,
        )

    def wake_worker(
        self,
        run_id: str,
        worker_id: str,
        *,
        now: float | None = None,
    ) -> None:
        """Explicitly clear quarantine after human intervention."""

        timestamp = self._now(now)
        with self._transaction() as connection:
            worker = self._require_worker_locked(connection, run_id, worker_id)
            if worker["current_attempt_id"] is not None:
                raise StateError("cannot wake a worker that still owns an attempt")
            connection.execute(
                """
                UPDATE workers
                SET status = 'idle', consecutive_errors = 0,
                    automatic_restarts_without_success = 0,
                    last_seen = ?, updated_at = ?
                WHERE run_id = ? AND worker_id = ?
                """,
                (timestamp, timestamp, run_id, worker_id),
            )
            self._event_locked(
                connection,
                run_id,
                "worker_woken",
                now=timestamp,
                worker_id=worker_id,
            )

    def get_task(self, run_id: str, case_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM tasks WHERE run_id = ? AND case_id = ?",
                (run_id, str(case_id)),
            ).fetchone()
        if row is None:
            raise StateError(f"unknown task {case_id!r} for run {run_id!r}")
        return dict(row)

    def get_worker(self, run_id: str, worker_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM workers WHERE run_id = ? AND worker_id = ?",
                (run_id, worker_id),
            ).fetchone()
        if row is None:
            raise UnknownWorkerError(f"unknown worker {worker_id!r} for run {run_id!r}")
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def list_events(
        self,
        run_id: str,
        *,
        after_event_id: int = 0,
        limit: int | None = None,
    ) -> tuple[EventRecord, ...]:
        if after_event_id < 0:
            raise ValueError("after_event_id cannot be negative")
        if limit is not None and limit < 1:
            raise ValueError("event limit must be at least 1")
        sql = (
            "SELECT * FROM events WHERE run_id = ? AND event_id > ? "
            "ORDER BY event_id"
        )
        parameters: list[Any] = [run_id, after_event_id]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        return tuple(
            EventRecord(
                event_id=int(row["event_id"]),
                run_id=str(row["run_id"]),
                created_at=float(row["created_at"]),
                event_type=str(row["event_type"]),
                worker_id=row["worker_id"],
                case_id=row["case_id"],
                attempt_id=row["attempt_id"],
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        )


# A short alias keeps integration code readable without hiding the role.
StateStore = PrincessState
