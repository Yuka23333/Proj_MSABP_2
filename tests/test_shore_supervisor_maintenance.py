"""Transaction-local maintenance observations: no host, network, keys or repair."""
# ruff: noqa: F811 -- imported pytest fixtures
from __future__ import annotations

from contextlib import ExitStack, closing, contextmanager
import json
import os
import sqlite3

import pytest

from test_shore_job_material import materials  # noqa: F401
from test_shore_remote_worker import remote_worker  # noqa: F401
from test_shore_supervisor_runtime import lifecycle  # noqa: F401
from msabp_opt.simulation.distributed import control_supervisor_runtime as runtime
from msabp_opt.simulation.distributed.control_remote_worker import _json_bytes
from msabp_opt.simulation.distributed.control_supervisor import MaintenanceRequired, MaterialJournal


def tree_state(root):
    """Exclude access times, which a genuinely read-only file read can update."""
    return {str(path.relative_to(root)): (
        path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns,
        path.read_bytes() if path.is_file() else None)
        for path in [root, *root.rglob("*")]}


def require_readonly_connections(monkeypatch):
    connect, opened = sqlite3.connect, []

    def readonly(database, *args, **kwargs):
        assert kwargs.get("uri") is True
        assert str(database).endswith(("?mode=ro", "?mode=ro&immutable=1"))
        opened.append(str(database))
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", readonly)
    return opened


@contextmanager
def held_lock(path):
    """Installer's existing lock file, immutable b'0', offset 1/length 1."""
    with path.open("r+b", buffering=0) as stream:
        stream.seek(1)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(1)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def test_empty_snapshot_exact_contract_is_readonly_under_callers_three_locks(lifecycle, monkeypatch):
    p = lifecycle
    marker = json.loads((p.root / "runtime-installed.json").read_bytes())

    def forbidden(*_args, **_kwargs):
        pytest.fail("Maintenance must not acquire locks or invoke lifecycle/identity/initialization")

    for name in ("step", "run", "_clients", "_material_client", "_discover"):
        monkeypatch.setattr(runtime.SupervisorRuntime, name, forbidden)
    monkeypatch.setattr(runtime.RuntimeLedger, "exclusive", forbidden)
    monkeypatch.setattr(MaterialJournal, "exclusive", forbidden)
    monkeypatch.setattr(runtime, "initialize_runtime_state", forbidden)
    monkeypatch.setattr(runtime, "initialize_state", forbidden)
    opened = require_readonly_connections(monkeypatch)
    before = tree_state(p.tmp_path)
    with ExitStack() as locks:
        for name in ("host.lock", "runtime.lock", "maintenance.lock"):
            locks.enter_context(held_lock(p.root / name))
        first = runtime.maintenance_snapshot(p.install)
        assert runtime.maintenance_snapshot(p.install) == first
    assert first == {"schema": "msabp.supervisor-maintenance-snapshot", "schema_version": 1,
        "installation_id": marker["installation_id"], "release_id": p.materials.profile["release_id"],
        "checked_jobs": 0, "checked_processes": 0, "quiescent": True}
    assert opened and p.calls == []
    assert tree_state(p.tmp_path) == before
    assert all((p.root / name).read_bytes() == b"0"
               for name in ("host.lock", "runtime.lock", "maintenance.lock"))


def test_empty_new_ledger_does_not_attest_legacy_jobs_or_authorize_replay(lifecycle):
    p = lifecycle
    # Deliberately outside the new supervisor's inventory. The installer must
    # separately block migration on this unknown legacy state; this API cannot
    # make it safe simply because the new runtime has no records.
    legacy = p.tmp_path / "legacy-003-unknown.json"
    legacy.write_bytes(b'{"state":"UNKNOWN","original_exit":null}')
    first = runtime.maintenance_snapshot(p.install)
    assert first["quiescent"] is True and first["checked_jobs"] == first["checked_processes"] == 0
    assert "legacy" not in first and "authorization" not in first
    ledger = runtime.RuntimeLedger(p.root, profile=p.materials.profile)
    with ledger.db() as db:
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,'SESSION',NULL,NULL,NULL)",
                   ("job-1", _json_bytes(p.case.binding).decode(), "{}", _json_bytes(p.materials.profile).decode()))
    with pytest.raises(MaintenanceRequired, match="release and idle"):
        runtime.maintenance_snapshot(p.install)
    assert legacy.read_bytes() == b'{"state":"UNKNOWN","original_exit":null}'


@pytest.mark.parametrize("name", ["runtime.sqlite", "runtime-installed.json", "host.lock", "runtime.lock", "maintenance.lock"])
def test_missing_snapshot_inputs_are_never_created(lifecycle, monkeypatch, name):
    p = lifecycle
    (p.root / name).unlink()
    before = tree_state(p.tmp_path)
    require_readonly_connections(monkeypatch)
    with pytest.raises(MaintenanceRequired):
        runtime.maintenance_snapshot(p.install)
    assert tree_state(p.tmp_path) == before


@pytest.mark.parametrize("damage", ["bytes", "jobs-table", "processes-table", "marker"])
def test_corrupt_runtime_history_is_maintenance_without_repair(lifecycle, monkeypatch, damage):
    p = lifecycle
    if damage == "bytes":
        (p.root / "runtime.sqlite").write_bytes(b"corrupt original database")
    elif damage == "marker":
        (p.root / "runtime-installed.json").write_bytes(b"[]")
    else:
        with sqlite3.connect(p.root / "runtime.sqlite") as db:
            db.execute("DROP TABLE " + ("jobs" if damage == "jobs-table" else "processes"))
    before = tree_state(p.tmp_path)
    require_readonly_connections(monkeypatch)
    with pytest.raises(MaintenanceRequired):
        runtime.maintenance_snapshot(p.install)
    assert tree_state(p.tmp_path) == before


@pytest.mark.parametrize("damage", ["INTENT", "RUNNING", "UNKNOWN", "missing-identity", "missing-exit",
    "pid-mismatch", "invalid-json", "nonobject-json", "exit-before-start", "orphan"])
def test_unresolved_or_damaged_original_process_blocks_snapshot(lifecycle, monkeypatch, damage):
    p = lifecycle
    ledger = runtime.RuntimeLedger(p.root, profile=p.materials.profile)
    identity = _json_bytes({"pid": 100, "creation_time": 555}).decode()
    exit_record = _json_bytes({"pid": 100, "creation_time": 555, "exit_time": 999, "exit_code": 0}).decode()
    state = damage if damage in {"INTENT", "RUNNING", "UNKNOWN"} else "EXITED"
    if damage == "missing-identity":
        identity = None
    elif damage == "missing-exit":
        exit_record = None
    elif damage == "pid-mismatch":
        identity = _json_bytes({"pid": 101, "creation_time": 555}).decode()
    elif damage == "invalid-json":
        identity = "{"
    elif damage == "nonobject-json":
        identity = "null"
    elif damage == "exit-before-start":
        exit_record = _json_bytes({"pid": 100, "creation_time": 555, "exit_time": 554, "exit_code": 0}).decode()
    with ledger.db() as db:
        if damage != "orphan":
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,'RELEASED',NULL,'{}','{}')",
                ("job-1", _json_bytes(p.case.binding).decode(), "{}", _json_bytes(p.materials.profile).decode()))
        db.execute("INSERT INTO processes VALUES('original','job-1','execute',?,?,?)", (state, identity, exit_record))
    before = tree_state(p.tmp_path)
    require_readonly_connections(monkeypatch)
    with pytest.raises(MaintenanceRequired):
        runtime.maintenance_snapshot(p.install)
    assert tree_state(p.tmp_path) == before


@pytest.mark.parametrize("stage", ["RESERVED", "DOWNLOADING", "COMMITTED", "SESSION_INIT", "SESSION", "UNKNOWN", "RELEASED"])
def test_unreleased_or_missing_terminal_history_blocks_snapshot(lifecycle, monkeypatch, stage):
    p = lifecycle
    ledger = runtime.RuntimeLedger(p.root, profile=p.materials.profile)
    with ledger.db() as db:
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,NULL,NULL,NULL)",
            ("job-1", _json_bytes(p.case.binding).decode(), "{}", _json_bytes(p.materials.profile).decode(), stage))
    before = tree_state(p.tmp_path)
    require_readonly_connections(monkeypatch)
    with pytest.raises(MaintenanceRequired, match="release and idle"):
        runtime.maintenance_snapshot(p.install)
    assert tree_state(p.tmp_path) == before


def test_uncheckpointed_wal_cannot_hide_new_job_from_immutable_snapshot(lifecycle, monkeypatch):
    p = lifecycle
    # The main database is empty; the newly unresolved job exists only in WAL.
    # Unconditional immutable=1 would incorrectly report zero checked jobs.
    with closing(sqlite3.connect(p.root / "runtime.sqlite")) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("INSERT INTO jobs VALUES(?,?,?,?,'SESSION',NULL,NULL,NULL)",
            ("job-1", _json_bytes(p.case.binding).decode(), "{}", _json_bytes(p.materials.profile).decode()))
        writer.commit()
        assert (p.root / "runtime.sqlite-wal").stat().st_size > 0
        before = tree_state(p.tmp_path)
        require_readonly_connections(monkeypatch)
        with pytest.raises(MaintenanceRequired, match="Nonempty SQLite recovery journal"):
            runtime.maintenance_snapshot(p.install)
        assert tree_state(p.tmp_path) == before


def test_empty_wal_and_shm_left_by_normal_readonly_check_are_untouched(lifecycle, monkeypatch):
    p = lifecycle
    database = p.root / "runtime.sqlite"
    with closing(sqlite3.connect(database)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as reader:
        assert reader.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert (p.root / "runtime.sqlite-wal").stat().st_size == 0
    assert (p.root / "runtime.sqlite-shm").stat().st_size > 0
    before = tree_state(p.tmp_path)
    opened = require_readonly_connections(monkeypatch)
    assert runtime.maintenance_snapshot(p.install)["checked_jobs"] == 0
    assert all(uri.endswith("?mode=ro&immutable=1") for uri in opened)
    assert tree_state(p.tmp_path) == before


def test_nonempty_rollback_journal_is_not_recovered_or_deleted(lifecycle, monkeypatch):
    p = lifecycle
    (p.root / "runtime.sqlite-journal").write_bytes(b"unresolved rollback history")
    before = tree_state(p.tmp_path)
    require_readonly_connections(monkeypatch)
    with pytest.raises(MaintenanceRequired, match="Nonempty SQLite recovery journal"):
        runtime.maintenance_snapshot(p.install)
    assert tree_state(p.tmp_path) == before
