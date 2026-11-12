"""Offline supervisor material journal; NOT yet a live service entry point.

Explicit first-install initialization, conservative restart, and immutable
four-file commits are implemented here. Authority discovery, activation,
process hosting and terminal-job release are deliberately not wired yet. An
existing job remains blocking even when the network reports an empty pool.
Installation/ACL checks and fixed production paths belong to the future host.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from uuid import uuid4

from .control_job_material import MAX_MATERIAL_BYTES, validate_material
from .control_release import safe_job_id, validate_profile
from .control_remote_worker import _json_bytes, checked_path

STATE_SCHEMA = "msabp.supervisor-state"
STATE_VERSION = 1
_FILES = {"authority-bootstrap.json", "samples.csv", "maid-runtime.json", "bell-config.json"}


class MaintenanceRequired(RuntimeError):
    """Missing, inconsistent or ambiguous state must not be silently rebuilt."""


def _regular(path: Path) -> Path:
    path = checked_path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise MaintenanceRequired("State file must be regular and non-aliased")
    return path


@contextmanager
def _maintenance_database(path: Path):
    """Read a stopped, externally locked database without SQLite sidecar writes.

    SQLite mode=ro alone can update/create WAL shared memory. Immutable reads
    avoid that, but ignore WAL contents: never use them with a nonempty WAL or
    rollback journal. Do not checkpoint, delete, or repair ambiguous history.
    Empty WAL and existing SHM left by normal read-only checks are harmless.
    """
    path = _regular(path)

    def observed():
        state = []
        for suffix in ("", "-wal", "-shm", "-journal"):
            candidate = checked_path(path.with_name(path.name + suffix))
            if suffix and not candidate.exists():
                state.append(None)
                continue
            info = _regular(candidate).stat()
            if suffix in {"-wal", "-journal"} and info.st_size:
                raise MaintenanceRequired("Nonempty SQLite recovery journal requires maintenance")
            state.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
        return state

    before = observed()
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            yield db
    finally:
        if observed() != before:
            raise MaintenanceRequired("Original database or sidecars changed during maintenance")


def _write_new(path: Path, data: bytes) -> None:
    checked_path(path)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _same_or_new(path: Path, data: bytes) -> None:
    checked_path(path)
    if path.exists():
        if _regular(path).read_bytes() != data:
            raise MaintenanceRequired("Immutable material changed")
    else:
        _write_new(path, data)


def initialize_state(root: str | Path, config_parent: str | Path, profile: dict) -> None:
    """Explicit installation step, never called as automatic startup recovery.

    A failed partial initialization is preserved and requires maintenance. The
    caller must enforce administrator access/ACLs and an idle installation.
    """
    root, config = checked_path(root), checked_path(config_parent)
    profile = validate_profile(profile)
    if root == config or root in config.parents or config in root.parents:
        raise ValueError("Supervisor state and committed config roots must be disjoint")
    if not config.is_dir():
        raise ValueError("Existing trusted configuration parent required")
    root.mkdir(exist_ok=False)
    checked_path(root)
    marker = {"schema": STATE_SCHEMA, "schema_version": STATE_VERSION,
        "installation_id": uuid4().hex, "config_parent": str(config), "profile": profile}
    _write_new(root / "instance.lock", b"0")
    db = sqlite3.connect(root / "supervisor.sqlite")
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE install (id INTEGER PRIMARY KEY CHECK(id=1), record TEXT NOT NULL)")
        db.execute("INSERT INTO install VALUES(1,?)", (_json_bytes(marker).decode(),))
        db.execute("""CREATE TABLE materials (job_id TEXT PRIMARY KEY COLLATE NOCASE,
            binding TEXT NOT NULL, material_sha256 TEXT NOT NULL, raw BLOB NOT NULL,
            stage TEXT NOT NULL CHECK(stage IN ('DOWNLOADED','VERIFIED','COMMITTED')),
            verified TEXT)""")
        db.commit()
    finally:
        db.close()
    # Last install marker is not recreated if interrupted or deleted.
    _write_new(root / "installed.json", _json_bytes(marker))


class MaterialJournal:
    """Single active job, byte-identical retries, no network or process launch.

    Mutating operations and the runner handoff acquire the durable instance
    lock. ``original_readonly`` instead requires the caller's maintenance
    exclusion and never takes a lock or initializes/reconciles state.
    """

    def __init__(self, root: str | Path, *, profile: dict, bootstrap_data: bytes,
                 maintenance_readonly=False):
        self.root = checked_path(root)
        self.maintenance_readonly = maintenance_readonly
        self.profile = validate_profile(profile)
        self.bootstrap_data = bootstrap_data
        if hashlib.sha256(bootstrap_data).hexdigest() != self.profile["bootstrap_sha256"]:
            raise MaintenanceRequired("Local bootstrap pin mismatch")
        self._installation()

    def _installation(self, *, maintenance=False) -> dict:
        try:
            marker_raw = _regular(self.root / "installed.json").read_bytes()
            marker = json.loads(marker_raw)
            if (type(marker) is not dict or set(marker) != {
                    "schema", "schema_version", "installation_id", "config_parent", "profile"}
                    or marker["schema"] != STATE_SCHEMA or type(marker["schema_version"]) is not int
                    or marker["schema_version"] != STATE_VERSION or marker["profile"] != self.profile
                    or _json_bytes(marker) != marker_raw):
                raise MaintenanceRequired("Invalid or changed supervisor installation")
            # The exact initialization record must survive in BOTH places.
            with self._database(readonly=True, maintenance=maintenance) as db:
                rows = db.execute("SELECT id, record FROM install").fetchall()
                if rows != [(1, marker_raw.decode())]:
                    raise MaintenanceRequired("Installation marker differs from the original ledger")
            config = checked_path(marker["config_parent"])
            if (not config.is_dir() or config == self.root or config in self.root.parents
                    or self.root in config.parents):
                raise MaintenanceRequired("Invalid local config root")
            _regular(self.root / "instance.lock")
            return marker
        except (OSError, ValueError, sqlite3.Error, KeyError) as exc:
            raise MaintenanceRequired("Original supervisor installation/ledger is missing or damaged") from exc

    @contextmanager
    def _database(self, *, readonly=False, maintenance=False):
        path = _regular(self.root / "supervisor.sqlite")
        if maintenance or self.maintenance_readonly:
            if not readonly:
                raise MaintenanceRequired("Maintenance cannot open a writable material journal")
            with _maintenance_database(path) as db:
                yield db
            return
        db = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"),
                             uri=True, timeout=2)
        try:
            db.execute("PRAGMA synchronous=FULL")
            if not readonly:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if not readonly:
                db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def exclusive(self):
        """Non-blocking OS lock; missing lock is maintenance, not reinitialize."""
        self._installation()
        path = _regular(self.root / "instance.lock")
        with path.open("r+b") as stream:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise MaintenanceRequired("Another supervisor owns the instance lock") from exc
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def status(self) -> dict:
        self._installation()
        with self._database(readonly=True) as db:
            rows = db.execute("SELECT job_id,stage,material_sha256 FROM materials ORDER BY job_id").fetchall()
        return {"schema": STATE_SCHEMA, "schema_version": STATE_VERSION,
            "mode": "offline-foundation", "discovery_enabled": False,
            "jobs": [{"job_id": j, "stage": s, "material_sha256": h} for j, s, h in rows]}

    def download(self, job_id: str, raw: bytes, *, expected_binding: dict) -> None:
        """Journal bounded response loss/retry before semantic verification.

        job_id/binding come from the caller's original verified authority
        snapshot, not from a downloaded URL/path. No new job can displace this
        record, even if it subsequently fails validation.
        """
        job_id = safe_job_id(job_id)
        if type(expected_binding) is not dict or expected_binding.get("job_id") != job_id:
            raise ValueError("Downloaded job differs from the expected binding")
        binding_raw = _json_bytes(expected_binding).decode()
        if len(binding_raw.encode()) > 64 * 1024:
            raise ValueError("Oversized job binding")
        if type(raw) is not bytes or not raw or len(raw) > MAX_MATERIAL_BYTES:
            raise ValueError("Oversized or missing download")
        digest = hashlib.sha256(raw).hexdigest()
        with self.exclusive(), self._database() as db:
            rows = db.execute("SELECT job_id,binding,material_sha256,raw FROM materials").fetchall()
            if rows:
                if rows != [(job_id, binding_raw, digest, raw)]:
                    raise MaintenanceRequired("Old job/material must be reconciled before new work")
                return
            target = checked_path(Path(self._installation()["config_parent"]) / job_id)
            if target.exists():
                raise MaintenanceRequired("Unjournaled existing config cannot be adopted")
            db.execute("INSERT INTO materials VALUES(?,?,?,?,?,NULL)",
                        (job_id, binding_raw, digest, raw, "DOWNLOADED"))

    def _verified(self, job_id: str, *, maintenance=False):
        with self._database(readonly=True, maintenance=maintenance) as db:
            row = db.execute("SELECT binding,material_sha256,raw,stage,verified FROM materials WHERE job_id=?",
                             (safe_job_id(job_id),)).fetchone()
        if row is None:
            raise MaintenanceRequired("Missing original material journal")
        binding, digest, raw, stage, record = row
        if hashlib.sha256(raw).hexdigest() != digest:
            raise MaintenanceRequired("Journal material digest changed")
        verified = validate_material(raw, profile=self.profile, bootstrap_data=self.bootstrap_data,
                                     expected_binding=json.loads(binding))
        if verified.job_id != job_id:
            raise MaintenanceRequired("Journal job ID differs from the original material binding")
        if (stage == "DOWNLOADED") != (record is None):
            raise MaintenanceRequired("Verification stage and durable record disagree")
        if record is not None and record != _json_bytes(verified.record).decode():
            raise MaintenanceRequired("Original verification record changed")
        if stage not in {"DOWNLOADED", "VERIFIED", "COMMITTED"}:
            raise MaintenanceRequired("Unknown material state")
        return verified, stage

    def verify(self, job_id: str) -> dict:
        with self.exclusive():
            verified, stage = self._verified(job_id)
            if stage == "DOWNLOADED":
                with self._database() as db:
                    db.execute("UPDATE materials SET stage='VERIFIED',verified=? WHERE job_id=?",
                               (_json_bytes(verified.record).decode(), job_id))
            # Return only public counters/digests, never runtime's API token.
            return {"job_id": job_id, "release_id": verified.release_id,
                    "material_sha256": verified.material_sha256}

    def original(self, job_id: str):
        """Return reverified original material only after immutable commit.

        This is an internal runner handoff, not a public status response: the
        four file bytes include the legacy runtime token. Never log this value.
        """
        with self.exclusive():
            return self._original(job_id)

    def original_readonly(self, job_id: str):
        """Reverify committed bytes without acquiring locks or writing state.

        The maintenance caller must already exclude the host and all runtime
        mutation. As with ``original``, the returned material contains the
        legacy runtime token and must never be logged or used as authorization.
        """
        return self._original(job_id, maintenance=True)

    def _original(self, job_id, *, maintenance=False):
        marker = self._installation(maintenance=maintenance)
        verified, stage = self._verified(job_id, maintenance=maintenance)
        if stage != "COMMITTED":
            raise MaintenanceRequired("Original material has not been committed")
        target = Path(marker["config_parent"]) / safe_job_id(job_id)
        self._directory(target, verified.files, complete=True)
        return verified

    @staticmethod
    def _directory(root: Path, files: dict, *, complete: bool):
        root = checked_path(root)
        names = {p.name for p in root.iterdir()}
        if names - _FILES or (complete and names != _FILES):
            raise MaintenanceRequired("Unexpected or missing material files")
        for name in names:
            if _regular(root / name).read_bytes() != files[name]:
                raise MaintenanceRequired("Immutable material bytes differ")

    def commit(self, job_id: str) -> Path:
        """Expose all four files by directory rename; no runner is called.

        Repeated commit after a lost reply validates every original byte. A
        crash after rename but before the final ledger transaction is safe to
        reconcile; missing committed files are not silently regenerated.
        """
        with self.exclusive():
            verified, stage = self._verified(job_id)
            if stage == "DOWNLOADED":
                raise MaintenanceRequired("Material must be verified before commit")
            config = Path(self._installation()["config_parent"])
            target = checked_path(config / job_id)
            staging = checked_path(self.root / "staging" / job_id)
            if stage == "COMMITTED" and not target.is_dir():
                raise MaintenanceRequired("Committed material disappeared")
            if target.exists():
                self._directory(target, verified.files, complete=True)
                if staging.exists():
                    raise MaintenanceRequired("Ambiguous staging and committed directories")
            else:
                staging.mkdir(parents=True, exist_ok=True)
                self._directory(staging, verified.files, complete=False)
                for name, data in verified.files.items():
                    _same_or_new(staging / name, data)
                self._directory(staging, verified.files, complete=True)
                # Trusted disjoint roots must share a filesystem for rename;
                # an unsupported deployment fails, never falls back to copying.
                os.rename(staging, target)
                self._directory(target, verified.files, complete=True)
            with self._database() as db:
                db.execute("UPDATE materials SET stage='COMMITTED' WHERE job_id=?", (job_id,))
            return target
