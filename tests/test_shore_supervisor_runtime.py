"""Local fake identities only: lifecycle restart, original work, and fail closure."""
# ruff: noqa: F811 -- imported pytest fixtures
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("shore_control")
from shore_control.client import Client, MemorySigner

from test_shore_job_material import materials  # noqa: F401
from test_shore_remote_worker import remote_worker  # noqa: F401
from test_shore_remote_controller import project, DirectSignedClient  # noqa: F401
from msabp_opt.simulation.distributed import control_supervisor_runtime as runtime
from msabp_opt.simulation.distributed.control_remote_worker import BELL_ACTOR, MAID_ACTOR, _json_bytes
from msabp_opt.simulation.distributed.control_supervisor import MaintenanceRequired


@pytest.fixture
def lifecycle(materials, tmp_path, monkeypatch):
    case = materials.make(1)
    config = tmp_path / "config"
    config.mkdir()
    root = tmp_path / "supervisor"
    runtime.initialize_runtime_state(root, config, materials.profile)
    bootstrap = json.loads(materials.bootstrap_data)
    install = {"profile": materials.profile, "bootstrap": bootstrap,
        "bootstrap_data": materials.bootstrap_data, "policy": {"discovery_enabled": True},
        "paths": {"state": root, "config": config, "app": tmp_path, "python": Path(r"C:\fake\python.exe"),
                  "policy": tmp_path / "policy.json"}}
    keys = case.keys
    bell = Client(case.authority, BELL_ACTOR, MemorySigner(keys[BELL_ACTOR]),
                  keys["authority"].public_key(), bootstrap["fleet_id"])
    seen, calls, sessions = [], [], []
    original_call = bell.call
    def call(method, **kwargs):
        calls.append(method)
        return original_call(method, **kwargs)
    bell.call = call
    transport = SimpleNamespace(case=case, terminal_proof=None, fail_material=False)
    class MaterialClient:
        def __init__(self, **kwargs):
            seen.append(copy.deepcopy(kwargs["expected_binding"]))
        def offer(self):
            return {"permit_id": transport.case.permit["payload"]["permit_id"],
                    "terminal_proof": transport.terminal_proof}
        def material(self, offer):
            if transport.fail_material:
                raise TimeoutError("lost material reply")
            return transport.case.raw
    class Session:
        def __init__(self, installation, ledger, material, client, verifier, *, create):
            self.material, self.create = material, create
            self.result = {"state": "UNKNOWN", "terminal": None}
            sessions.append(self)
        def step(self):
            return self.result
    verified = []
    def verify(proof, bootstrap, expected_binding):
        assert proof == {"request": "original", "response": "signed"}
        verified.append(copy.deepcopy(expected_binding))
    monkeypatch.setattr("msabp_opt.simulation.distributed.control_job_http.verify_release_proof", verify)
    def open_runtime():
        return runtime.SupervisorRuntime(install, bell_client=bell,
            material_client_factory=MaterialClient, session_factory=Session)
    return SimpleNamespace(**locals())


def test_maintenance_is_default_and_never_opens_identity(lifecycle):
    p = lifecycle
    p.install["policy"] = {}
    service = runtime.SupervisorRuntime(p.install)
    assert service.step()["state"] == "MAINTENANCE"
    assert p.calls == [] and service.client is None


def test_explicit_idle_stop_returns_without_identity_or_killing(lifecycle):
    p = lifecycle
    p.install["policy"] = {"discovery_enabled": False, "stop_when_idle": True}
    service = runtime.SupervisorRuntime(p.install)
    assert service.step()["state"] == "STOPPED"
    assert service.run() == 0
    assert service.client is None


def test_stop_flag_keeps_draining_old_unknown(lifecycle):
    p = lifecycle
    service = p.open_runtime()
    for _ in range(3):
        service.step()
    p.install["policy"] = {"discovery_enabled": False, "stop_when_idle": True}
    assert service.step()["state"] == "UNKNOWN"


def test_idle_upgrade_preserves_original_marker_and_ledger(lifecycle):
    from msabp_opt.simulation.distributed.control_release import public_profile
    p = lifecycle
    marker = (p.root / "runtime-installed.json").read_bytes()
    manifest = copy.deepcopy(p.materials.profile["release_manifest"])
    first = next(iter(manifest["source_hashes"]))
    manifest["source_hashes"][first] = "f" * 64
    upgraded = public_profile(manifest, p.materials.profile["bootstrap_sha256"])
    changed = {**p.install, "profile": upgraded, "policy": {"discovery_enabled": False}}
    assert runtime.SupervisorRuntime(changed).step()["state"] == "MAINTENANCE"
    assert (p.root / "runtime-installed.json").read_bytes() == marker
    service = p.open_runtime()
    service.step()
    with pytest.raises(MaintenanceRequired, match="original release"):
        runtime.SupervisorRuntime(changed)


def test_upgraded_current_release_running_process_is_not_an_old_process(lifecycle):
    from msabp_opt.simulation.distributed.control_release import public_profile
    p = lifecycle
    manifest = copy.deepcopy(p.materials.profile["release_manifest"])
    manifest["source_hashes"][next(iter(manifest["source_hashes"]))] = "f" * 64
    upgraded = public_profile(manifest, p.materials.profile["bootstrap_sha256"])
    ledger = runtime.RuntimeLedger(p.root, profile=upgraded)
    # The immutable installation still records release A; a real durable
    # current-release B job/process row must remain legal during B execution.
    with ledger.db() as db:
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,'SESSION',NULL,NULL,NULL)",
            ("job-1", _json_bytes(p.case.binding).decode(), "{}", _json_bytes(upgraded).decode()))
        db.execute("INSERT INTO processes VALUES('original-handle','job-1','execute','RUNNING',?,NULL)",
            (_json_bytes({"pid": 100, "creation_time": 555}).decode(),))
    assert runtime.RuntimeLedger(p.root, profile=upgraded).active()["stage"] == "SESSION"
    assert json.loads((p.root / "runtime-installed.json").read_bytes())["profile"] == p.materials.profile
    # A prior-release subtree remains blocking even if its job was incorrectly
    # marked released: only the same current owning-job profile is allowed.
    with ledger.db() as db:
        db.execute("UPDATE jobs SET stage='RELEASED',profile=?", (_json_bytes(p.materials.profile).decode(),))
    with pytest.raises(MaintenanceRequired, match="old subtree"):
        runtime.RuntimeLedger(p.root, profile=upgraded)


@pytest.mark.parametrize("name", ["runtime.sqlite", "runtime-installed.json", "runtime.lock"])
def test_missing_original_state_never_reinitialized(lifecycle, name):
    p = lifecycle
    (p.root / name).unlink()
    with pytest.raises(MaintenanceRequired):
        p.open_runtime()
    assert not (p.root / name).exists()


def test_two_instances_compete_for_original_os_lock(lifecycle):
    one, two = lifecycle.open_runtime(), lifecycle.open_runtime()
    with one.ledger.exclusive():
        with (lifecycle.root / "runtime.lock").open("rb", buffering=0) as stream:
            assert stream.read(1) == b"0"
        with pytest.raises(MaintenanceRequired, match="owns"):
            two.step()
    assert two.step()["state"] == "DISCOVERED"


def test_download_loss_restart_preserves_full_original_binding(lifecycle):
    p = lifecycle
    service = p.open_runtime()
    assert service.step()["state"] == "DISCOVERED"
    original = service.ledger.active()
    assert json.loads(original["snapshot"])["bound_job"]["session_id"] == "original-session"
    p.transport.fail_material = True
    with pytest.raises(TimeoutError):
        service.step()
    p.transport.fail_material = False
    service = p.open_runtime()
    assert service.step()["state"] == "COMMITTED"
    assert p.calls == ["status"]
    assert all(binding == p.case.binding for binding in p.seen)
    assert service.step()["state"] == "UNKNOWN"
    p.install["policy"]["discovery_enabled"] = False
    assert service.step()["state"] == "UNKNOWN"
    assert len(p.sessions) == 1
    assert p.calls == ["status"]


def test_unknown_old_job_blocks_fresh_empty_or_other_status(lifecycle):
    p = lifecycle
    service = p.open_runtime()
    assert [service.step()["state"] for _ in range(3)] == ["DISCOVERED", "COMMITTED", "UNKNOWN"]
    p.bell.call = lambda *_a, **_k: pytest.fail("Old job must not discover or clear from status")
    assert p.open_runtime().step()["state"] == "UNKNOWN"
    assert p.sessions[-1].create is False
    assert len(list(p.config.iterdir())) == 1


@pytest.mark.parametrize("missing", ["material", "wake.sqlite", "rpc.sqlite", "release"])
def test_retired_original_damage_blocks_discovery_and_clean_stop(lifecycle, monkeypatch, missing):
    p = lifecycle
    service = p.open_runtime()
    for _ in range(3):
        service.step()
    p.sessions[-1].result = terminal()
    p.transport.terminal_proof = {"request": "original", "response": "signed"}
    assert service.step()["state"] == "RELEASED"
    # No real installed paths are touched. Inject the missing-original check
    # at the readonly boundary, and assert neither new discovery nor stop runs.
    if missing == "release":
        monkeypatch.setattr(runtime, "verify_release", lambda *_a: (_ for _ in ()).throw(FileNotFoundError("old release")))
    else:
        monkeypatch.setattr(runtime, "verify_release", lambda *_a: p.materials.profile)
        if missing == "material":
            (p.root / "jobs/job-1/material/supervisor.sqlite").unlink()
        else:
            # Existing synthetic fixture has no real runner DB; this remains
            # missing and cannot be initialized as a new completed worker.
            from msabp_opt.simulation.distributed import control_remote_worker
            monkeypatch.setattr(control_remote_worker, "_read_idle_permits",
                lambda *_a: (_ for _ in ()).throw(FileNotFoundError(missing)))
    p.bell.call = lambda *_a, **_k: pytest.fail("Damaged old state cannot discover")
    with pytest.raises((ValueError, OSError, MaintenanceRequired)):
        service.step()
    p.install["policy"] = {"discovery_enabled": False, "stop_when_idle": True}
    with pytest.raises((ValueError, OSError, MaintenanceRequired)):
        p.open_runtime().step()


@pytest.mark.parametrize("stage", ["RESERVED", "SESSION_INIT", "BAD"])
def test_interrupted_initialization_never_recreates_journal(lifecycle, stage):
    p = lifecycle
    service = p.open_runtime()
    service.step()
    with service.ledger.db() as db:
        db.execute("UPDATE jobs SET stage=?", (stage,))
    with pytest.raises(MaintenanceRequired):
        p.open_runtime().step()
    assert not p.sessions


def terminal():
    return {"state": "IDLE", "terminal": {"bell_idle": {"signed": "bell"},
            "maid_idle": {"signed": "maid"}, "outboxes_pending": 0,
            "unknown": False, "subtree_exited": True, "processes": []}}


def test_next_job_requires_release_local_idle_and_original_tree_exit(lifecycle):
    p = lifecycle
    service = p.open_runtime()
    for _ in range(3):
        service.step()
    p.sessions[-1].result = terminal()
    assert service.step()["state"] == "AWAITING_RELEASE"
    assert p.calls == ["status"]
    p.transport.terminal_proof = {"request": "original", "response": "signed"}
    p.sessions[-1].result["terminal"]["subtree_exited"] = False
    with pytest.raises(MaintenanceRequired, match="incomplete"):
        service.step()
    p.sessions[-1].result = terminal()
    assert service.step()["state"] == "RELEASED"
    assert p.verified == [p.case.binding]
    assert service.ledger.active() is None
    original_bytes = {path.name: path.read_bytes() for path in (p.config / "job-1").iterdir()}
    second = p.materials.make(4)
    p.transport.case = second
    p.bell.transport = second.authority
    # This test isolates global transitions; the real original idle/proof
    # validation is exercised by the 1/4-point adapter regression below.
    service._verify_retired = lambda row: None
    assert service.step()["state"] == "DISCOVERED"
    assert service.step()["state"] == "COMMITTED"
    assert service.ledger.active()["job_id"] == "job-4"
    assert {path.name: path.read_bytes() for path in (p.config / "job-1").iterdir()} == original_bytes
    assert p.calls == ["status", "status"]


@pytest.mark.parametrize("field,value", [("phase", "DRAINING"), ("phase", "PAUSED_BOUND")])
def test_only_active_new_job_can_be_discovered(lifecycle, field, value):
    p = lifecycle
    status = p.bell.call("status")
    status["bound_job"][field] = value
    assert runtime.discovery_binding(status, p.bootstrap) is None


def test_unexpected_host_scope_and_source_paths_rejected(lifecycle):
    p = lifecycle
    status = p.bell.call("status")
    status["bound_job"]["host_id"] = "other-host"
    with pytest.raises(ValueError):
        runtime.discovery_binding(status, p.bootstrap)


def test_process_intent_and_original_handle_exit_are_durable(lifecycle):
    p = lifecycle
    service = p.open_runtime()
    service.step()
    child = SimpleNamespace(pid=100, creation_time=555, poll=lambda: None)
    tree = SimpleNamespace(active_processes=lambda: 1, close=lambda: None)
    captured = []
    def spawn(argv, *, cwd):
        assert service.ledger.processes("job-1")[0]["state"] == "INTENT"
        captured.append(argv)
        return child
    tree.spawn = spawn
    launcher = runtime.ManagedMaidLauncher(p.install, service.ledger, "job-1", job_factory=lambda: tree)
    launcher.spawn(["--recover-terminal"], purpose="recover-terminal")
    assert captured[0][1:4] == ["-I", "-B", "-u"]
    assert launcher.observe() == "RUNNING"
    # A new runtime has no original handle. PID 100 must not be reopened.
    restarted = runtime.ManagedMaidLauncher(p.install, service.ledger, "job-1", job_factory=lambda: tree)
    assert restarted.observe() == "UNKNOWN"
    child.poll = lambda: 0
    child.exit_record = SimpleNamespace(to_dict=lambda: {
        "pid": 100, "creation_time": 555, "exit_time": 999, "exit_code": 0})
    assert launcher.observe() == "RUNNING"  # Descendant still owns tree membership.
    tree.active_processes = lambda: 0
    assert launcher.observe() == "EXITED"
    record = service.ledger.processes("job-1")[0]
    assert json.loads(record["exit_record"])["creation_time"] == 555
    assert restarted.observe() == "EXITED"


@pytest.mark.parametrize("project", [1, 4], indirect=True)
@pytest.mark.parametrize("damage", [None, "wake.sqlite", "rpc.sqlite", "worker.sqlite", "worker-rpc.sqlite",
    "corrupt-wake", "corrupt-worker", "bell-outbox", "worker-outbox", "wake-unknown",
    "terminal-unknown", "terminal-outbox", "maid-signature", "old-release", "process-history"])
def test_real_synthetic_bell_maid_idle_and_release_adapter(project, monkeypatch, damage):
    """Real authority signatures, synthetic executor and both durable journals.

    Only process creation/WinJob observations are fake here; WinJob itself has
    a separate native regression suite. Nothing contacts an installed identity.
    """
    from msabp_opt.simulation.distributed import control_remote_http
    from msabp_opt.simulation.distributed.control_remote_worker import (
        MODE, RemoteSyntheticMaidConfig, _persist_worker_idle,
    )
    from msabp_opt.simulation.distributed.control_synthetic import SyntheticSession, synthetic_case
    from msabp_opt.simulation.distributed.control_worker import ShoreWorker
    from msabp_opt.simulation.distributed.maid import Maid, MaidRuntimeConfig
    from msabp_opt.simulation.distributed.control_job_http import verify_release_proof
    p = project
    wake = p.started()
    p.bell.report(wake["attempt_id"])
    assert p.complete()["status"] == "RESOLVED"
    assert p.bell.resolved(wake["attempt_id"])
    policy = control_remote_http._CollectorPolicy(p.coordinator, p.bootstrap,
        SimpleNamespace(network=p.bootstrap["network"]), p.prepared.paths.upload_root, p.queue.handle)
    client = DirectSignedClient(p, policy, MAID_ACTOR)
    config = RemoteSyntheticMaidConfig.from_mapping(p.runtime)
    worker = ShoreWorker(config.state_root / "control", p.clients[MAID_ACTOR],
        p.keys["authority"].public_key(), binding=p.runtime["binding"],
        registration=p.registrations[MAID_ACTOR], csv_sha256=p.runtime["csv_sha256"])
    maid_config = MaidRuntimeConfig(run_id="run", worker_id=MAID_ACTOR,
        princess_url=p.bootstrap["collector"]["endpoint"], api_token=p.prepared.api_token,
        csv_path=p.prepared.paths.worklist_csv, csv_sha256=p.runtime["csv_sha256"],
        project_path=p.tmp_path / "NEVER-OPEN.cst", output_root=Path(p.runtime["output_root"]),
        dry_run=True, protocol_mode=MODE, poll_seconds=0.001, heartbeat_seconds=1)
    maid = Maid(maid_config, client=client, artifact_client=client, control_worker=worker,
                session=SyntheticSession(), run_csv_row=synthetic_case)
    assert maid.run() == 0
    _persist_worker_idle(config, worker, p.clients[MAID_ACTOR], client)
    session = object.__new__(runtime.OriginalJobSession)
    session.runtime, session.bootstrap, session.coordinator = config, p.bootstrap, p.bell
    session.permit, session.material = wake, SimpleNamespace(job_id="job")
    session.collector = DirectSignedClient(p, policy, BELL_ACTOR)
    session.launcher = SimpleNamespace(observe=lambda: "EXITED", spawn=lambda *_a, **_k: pytest.fail("No relaunch"))
    process = {"process_id": "test-original", "job_id": "job", "purpose": "execute", "state": "EXITED",
        "identity": _json_bytes({"pid": 100, "creation_time": 555}).decode(),
        "exit_record": _json_bytes({"pid": 100, "creation_time": 555, "exit_time": 999, "exit_code": 0}).decode()}
    session.ledger = SimpleNamespace(root=p.tmp_path / "global", processes=lambda _job: [process])
    result = session.step()
    assert result["state"] == "IDLE" and result["terminal"]["subtree_exited"]
    assert p.state.progress("run").completed == p.points
    assert p.queue.idle_pending() == 0
    p.gate.begin_drain("Synthetic local supervisor adapter test")
    assert p.gate.finish_drain(worker_outboxes_pending=0)["phase"] == "PAUSED_BOUND"
    assert p.gate.release(disposition="completed", reason="Local test")["phase"] == "RELEASED"
    with p.gate.journal.transaction() as db:
        key = db.execute("SELECT key FROM rpc_journal WHERE method='release_job'").fetchone()[0]
    proof = p.gate.journal.proof(key)
    assert verify_release_proof(proof, p.bootstrap, p.runtime["binding"])["phase"] == "RELEASED"
    assert len(list(p.prepared.paths.results_root.rglob("manifest.json"))) == p.points
    # Re-audit real signed idle + real original Bell/Maid journals after
    # release. Only immutable installation/material handoff are test seams.
    from msabp_opt.simulation.distributed.control_release import RELEASE_SOURCE_FILES, public_profile
    from msabp_opt.simulation.distributed.control_remote_worker import SHORE_SOURCE_FILES
    from test_shore_supervisor_maintenance import require_readonly_connections, tree_state
    bootstrap_data = _json_bytes(p.bootstrap)
    profile = public_profile({"schema": "msabp.supervisor-release", "schema_version": 1,
        "mode": MODE, "project_id": "MSABP_2", "source_hashes": {name: "a" * 64 for name in RELEASE_SOURCE_FILES},
        "shore_source_hashes": {name: "b" * 64 for name in SHORE_SOURCE_FILES}},
        runtime.hashlib.sha256(bootstrap_data).hexdigest())
    material = SimpleNamespace(files={"maid-runtime.json": _json_bytes(p.runtime)},
                               record={"permit_id": wake["permit_id"]})
    monkeypatch.setattr(runtime, "MaterialJournal", lambda *_a, **_k: SimpleNamespace(
        original=lambda _j: material, original_readonly=lambda _j: material))
    monkeypatch.setattr(runtime, "verify_release", lambda *_a: profile)
    monkeypatch.setattr(runtime, "runtime_paths", lambda *_a: {"source_root": p.tmp_path / "source",
        "shore_source_root": p.tmp_path / "shore-source", "bell_state_root": p.bell.root})
    state, committed_config = p.tmp_path / "maintenance-runtime", p.tmp_path / "maintenance-config"
    committed_config.mkdir()
    runtime.initialize_runtime_state(state, committed_config, profile)
    install = {"profile": profile, "bootstrap": p.bootstrap, "bootstrap_data": bootstrap_data,
        "policy": {"discovery_enabled": False}, "paths": {"state": state, "config": committed_config}}
    service = runtime.SupervisorRuntime(install)
    retired = {"job_id": "job", "profile": _json_bytes(profile).decode(),
        "binding": _json_bytes(p.runtime["binding"]).decode(),
        "release_proof": _json_bytes(proof).decode(), "terminal": _json_bytes(result["terminal"]).decode()}
    with service.ledger.db() as db:
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,'RELEASED',NULL,?,?)",
            ("job", retired["binding"], "{}", retired["profile"], retired["release_proof"], retired["terminal"]))
        db.execute("INSERT INTO processes VALUES(?,?,?,?,?,?)", tuple(process.values()))
    service._verify_retired(retired)
    before = tree_state(p.tmp_path)
    with monkeypatch.context() as readonly:
        opened = require_readonly_connections(readonly)
        snapshot = runtime.maintenance_snapshot(install)
    assert snapshot["quiescent"] is True and snapshot["checked_jobs"] == snapshot["checked_processes"] == 1
    assert {Path(uri.removeprefix("file:/").split("?")[0]).name for uri in opened} >= {
        "runtime.sqlite", "wake.sqlite", "rpc.sqlite", "worker.sqlite"}
    assert tree_state(p.tmp_path) == before
    if damage is not None:
        if damage in {"wake.sqlite", "rpc.sqlite", "worker.sqlite", "worker-rpc.sqlite"}:
            missing = (config.state_root / "control" / damage.removeprefix("worker-")
                       if damage in {"worker.sqlite", "worker-rpc.sqlite"} else p.bell.root / damage)
            missing.unlink()
        elif damage in {"corrupt-wake", "corrupt-worker"}:
            damaged = p.bell.root / "wake.sqlite" if damage == "corrupt-wake" else config.state_root / "control/worker.sqlite"
            damaged.write_bytes(b"corrupt original history")
        elif damage in {"bell-outbox", "worker-outbox", "wake-unknown"}:
            import sqlite3
            damaged = ((config.state_root / "control/rpc.sqlite") if damage == "worker-outbox"
                       else p.bell.root / ("wake.sqlite" if damage == "wake-unknown" else "rpc.sqlite"))
            with sqlite3.connect(damaged) as db:
                if damage == "wake-unknown":
                    db.execute("UPDATE wake SET state='STARTING'")
                else:
                    assert db.execute("UPDATE rpc_journal SET result=NULL,error=NULL").rowcount > 0
        elif damage == "old-release":
            monkeypatch.setattr(runtime, "verify_release", lambda *_a: (_ for _ in ()).throw(FileNotFoundError("old release")))
        elif damage == "process-history":
            with service.ledger.db() as db:
                db.execute("DELETE FROM processes")
        else:
            altered = copy.deepcopy(result["terminal"])
            if damage == "terminal-unknown":
                altered["unknown"] = True
            elif damage == "terminal-outbox":
                altered["outboxes_pending"] = 1
            else:
                altered["maid_idle"]["signature"] = "a" * 88
            with service.ledger.db() as db:
                db.execute("UPDATE jobs SET terminal=?", (_json_bytes(altered).decode(),))
        before = tree_state(p.tmp_path)
        with monkeypatch.context() as readonly:
            require_readonly_connections(readonly)
            with pytest.raises(MaintenanceRequired):
                runtime.maintenance_snapshot(install)
        assert tree_state(p.tmp_path) == before


def test_never_started_drained_wake_does_not_invent_maid_idle(project):
    p = project
    assignment = p.poll()["assignment"]
    permit = p.bell.prepare(assignment)
    p.gate.begin_drain("Before accept")
    p.bell.not_started(permit["attempt_id"])
    p.bell.report(permit["attempt_id"])
    assert p.complete()["status"] == "RESOLVED"
    assert p.bell.resolved(permit["attempt_id"])
    session = object.__new__(runtime.OriginalJobSession)
    session.coordinator, session.permit = p.bell, permit
    session.launcher = SimpleNamespace(observe=lambda: "EXITED", spawn=lambda *_a, **_k: pytest.fail("No launch"))
    assert session.step() == {"state": "MAINTENANCE_NOT_STARTED", "terminal": None}
