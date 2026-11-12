"""Actual run_worker/Maid.run recovery; temporary identities and in-process RPC only."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
import zipfile

import pytest

pytest.importorskip("shore_control")

from shore_control.client import Client, MemorySigner
from shore_control.crypto import verify_object
from shore_control.evidence import verify_artifacts
from shore_control.permits import DESCRIPTOR_FIELDS

from msabp_opt.simulation.distributed import control_remote_worker as runtime
from msabp_opt.simulation.distributed import control_synthetic
from msabp_opt.simulation.distributed.control_rpc import SignedRpcJournal
from msabp_opt.simulation.distributed.control_worker import ShoreWorker
from msabp_opt.simulation.distributed.maid import Maid, canonical_row_sha256
from msabp_opt.simulation.distributed.protocol import make_message

from test_shore_remote_worker import remote_worker as _remote_worker

remote_worker = _remote_worker


class ProcessLost(BaseException):
    """Simulated process loss, never caught as a returned execution failure."""


class DropReads:
    def __init__(self, authority):
        self.authority, self.calls, self.losses, self.lost = authority, [], [], []

    def challenge(self, actor):
        return self.authority.challenge(actor)

    def handle(self, request):
        self.calls.append(request)
        loss = self.losses.pop(0) if request["payload"]["method"] == "get_operation" and self.losses else None
        if loss == "undelivered":
            self.lost.append(request)
            raise TimeoutError("original signed read not delivered")
        response = self.authority.handle(request)
        if loss == "committed":
            self.lost.append(request)
            raise TimeoutError("committed get_operation reply lost")
        return response


@pytest.fixture
def lab(remote_worker, monkeypatch):
    env = remote_worker
    monkeypatch.setattr("socket.socket", lambda *_a, **_k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("Subprocess forbidden"))
    monkeypatch.setattr("platform.node", lambda: "offline-recovery-test")
    monkeypatch.setattr(runtime.time, "sleep", lambda _delay: None)
    env.clock = [time.time() + 1]
    env.authority.clock = lambda: env.clock[0]
    env.authority.challenge_ms = 1000
    transport = DropReads(env.authority)
    env.worker_client.transport = transport
    cfg = env.config
    bound = {name: cfg["binding"][name] for name in ("job_id", "epoch", "host_id", "session_id")}
    session = Client(env.authority, "session:" + bound["session_id"], MemorySigner(env.keys["session"]),
                     env.keys["authority"].public_key(), cfg["binding"]["fleet_id"])
    row_hash = canonical_row_sha256(("sample_id", "value"), {"sample_id": "case-1", "value": "1"})
    permit = env.controller.call("validate_dispatch", **bound, operation={
        "operation_id": "terminal-recovery-attempt", "run_id": cfg["run_id"],
        "worker_id": runtime.MAID_ACTOR, "attempt_id": "attempt-1", "action": "attempt",
        "input_sha256": row_hash, "action_context": None})
    assignment = {"case_id": "case-1", "row_index": 0, "attempt_id": "attempt-1",
        "lease_token": "lease-original", "row_sha256": row_hash,
        "control_authority": {"descriptor": {key: permit["payload"][key] for key in DESCRIPTOR_FIELDS},
                              "permit": permit}}
    stats = SimpleNamespace(executions=0, run_returns=[], messages=[], idles=[], uploads=0, ack=True)
    real_executor, real_run = control_synthetic.synthetic_case, Maid.run

    def executor(*args, **kwargs):
        stats.executions += 1
        return real_executor(*args, **kwargs)

    def tracked_run(maid):
        result = real_run(maid)
        stats.run_returns.append(result)
        return result

    monkeypatch.setattr(control_synthetic, "synthetic_case", executor)
    monkeypatch.setattr(Maid, "run", tracked_run)

    class Collector:
        def send(self, message):
            kind = message["type"]
            stats.messages.append(kind)
            assert message["protocol_version"] == 2
            if kind == "hello":
                reply, payload = "welcome", {}
            elif kind == "request_task":
                assert stats.messages.count("request_task") == 1, "Recovery must never claim new work"
                reply, payload = "assignment", assignment
            elif kind == "heartbeat":
                reply, payload = "heartbeat_ack", {}
            elif kind == "complete":
                control = message["payload"]["control_authority"]
                checked = verify_artifacts(control["evidence"], permit["payload"],
                    env.root / "collector-artifacts", control["aliases"])
                assert checked == control["evidence_sha256"]
                if stats.ack:
                    session.call("controller_ack", permit_id=control["permit_id"], durable=True,
                                 outcome=control["outcome"], evidence_sha256=checked)
                reply, payload = "completed_ack", {}
            else:
                raise AssertionError("Unexpected recovery message " + kind)
            return make_message(reply, run_id=cfg["run_id"], worker_id=runtime.MAID_ACTOR,
                                protocol_version=2, payload=payload)

        def upload_artifact(self, _attempt_id, _worker_id, _lease_token, archive):
            stats.uploads += 1
            target = env.root / "collector-artifacts"
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive) as stream:
                stream.extractall(target)
            return {"sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest()}

        def request_json(self, path, body):
            assert path == "/v1/control/idle"
            stats.idles.append(body["record"])
            return {"status": "RECORDED", "actor_id": runtime.MAID_ACTOR}

    class Verifier:
        def verify_local(self):
            pass

        def verify_peer(self, *_args):
            pass

    collector = Collector()
    factories = dict(worker_client_factory=lambda *_a, **_k: env.worker_client,
        collector_client_factory=lambda *_a, **_k: collector,
        verifier_factory=lambda _network: Verifier())
    ready = cfg.state_root / ("ready." + "1" * 32 + ".json")
    env.run = lambda: runtime.run_worker(cfg, startup_record=env.startup, ready_record=ready, **factories)
    env.recover = lambda: runtime.recover_worker_terminal(cfg, **factories)
    env.gate = lambda: ShoreWorker(cfg.state_root / "control", env.worker_client,
        env.keys["authority"].public_key(), binding=cfg["binding"],
        registration=env.bootstrap["workers"][runtime.MAID_ACTOR], csv_sha256=cfg["csv_sha256"])
    env.journal = lambda: SignedRpcJournal(cfg.state_root / "control" / "rpc.sqlite")
    env.transport, env.stats, env.factories, env.assignment = transport, stats, factories, assignment
    env.ready = ready
    return env


def operation_count(env):
    with sqlite3.connect(env.root / "authority.sqlite") as db:
        return db.execute("SELECT COUNT(*) FROM operations").fetchone()[0]


def history(journal, key):
    with journal.transaction() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM rpc_generations WHERE key=? ORDER BY generation", (key,))]


def assert_finished_once(env, before):
    assert env.stats.executions == env.stats.uploads == 1
    assert env.stats.run_returns == [2], "The real Maid.run must take completed_pending exit"
    assert env.stats.messages.count("hello") == env.stats.messages.count("request_task") == 1
    assert operation_count(env) == before
    assert env.gate().pending_assignments() == []
    assert env.journal().pending() == []
    assert len(env.stats.idles) == 1
    idle = verify_object(env.bootstrap["workers"][runtime.MAID_ACTOR]["public_key"],
                         runtime.IDLE_DOMAIN, env.stats.idles[0])
    assert idle["permit_ids"] == [env.assignment["control_authority"]["permit"]["payload"]["permit_id"]]
    assert idle["outboxes_pending"] == 0
    methods = [call["payload"]["method"] for call in env.transport.calls]
    assert methods.count("accept_permit") == methods.count("worker_report") == 1
    assert not set(methods) & {"acquire", "validate_dispatch", "renew", "register_run"}


def crash_before_retry(env, monkeypatch):
    original = runtime._recover_terminal_outboxes

    def crash(*_args):
        raise ProcessLost("process exited after Maid.run returned 2, before its first recovery")

    monkeypatch.setattr(runtime, "_recover_terminal_outboxes", crash)
    with pytest.raises(ProcessLost, match="before its first recovery"):
        env.run()
    monkeypatch.setattr(runtime, "_recover_terminal_outboxes", original)
    assert env.stats.run_returns == [2]
    assert not (env.config.state_root / "idle.json").exists()
    return env.journal().pending()[0]


def test_real_run_lost_committed_read_finishes_inprocess_without_new_claim(lab):
    before = operation_count(lab)
    lab.transport.losses = ["committed"]
    assert lab.run() == 0
    original, = lab.transport.lost
    journal = lab.journal()
    with journal.transaction() as db:
        rows = db.execute("SELECT request,response FROM rpc_generations").fetchall()
    saved = [row for row in rows if json.loads(row["request"]) == original]
    assert len(saved) == 1 and saved[0]["response"] is not None
    assert sum(call == original for call in lab.transport.calls) == 2
    assert lab.stats.messages.count("complete") == 1
    assert_finished_once(lab, before)


@pytest.mark.parametrize("expired", [False, True])
def test_restart_public_cli_after_real_run_exit_preserves_original_request(lab, monkeypatch, expired):
    before = operation_count(lab)
    lab.transport.losses = ["committed"]
    original = crash_before_retry(lab, monkeypatch)
    launch = lab.config.state_root / ("launch." + "1" * 32 + ".json")
    original_launch, original_ready = launch.read_bytes(), lab.ready.read_bytes()
    with pytest.raises(FileExistsError):
        lab.run()
    if expired:
        lab.clock[0] += 400

    script = Path(__file__).resolve().parents[1] / "scripts/simulation/shore_synthetic_maid.py"
    spec = spec_from_file_location("shore_maid_recovery_cli", script)
    entry = module_from_spec(spec)
    spec.loader.exec_module(entry)
    original_recover = entry.recover_worker_terminal
    monkeypatch.setattr(entry, "recover_worker_terminal",
        lambda config: original_recover(config, **lab.factories))
    # Exercise the documented public parser + actual recovery implementation.
    assert entry.main(["--runtime-config", str(lab.config.runtime_path), "--recover-terminal"]) == 0
    saved = history(lab.journal(), original["key"])
    assert len(saved) == 1
    assert json.loads(saved[0]["request"]) == original["request"]
    assert saved[0]["response"] is not None
    assert launch.read_bytes() == original_launch and lab.ready.read_bytes() == original_ready
    assert_finished_once(lab, before)


@pytest.mark.parametrize("loss", ["committed", "undelivered"])
def test_repeated_loss_bounded_and_expired_read_remains_recoverable(lab, monkeypatch, loss):
    before = operation_count(lab)
    lab.transport.losses = [loss]
    original = crash_before_retry(lab, monkeypatch)
    result = lab.gate().checked_result("attempt-1")
    manifest = Path(result["manifest_path"])
    original_bytes = manifest.read_bytes()
    archive = Path(lab.gate().delivery("attempt-1")["archive"])
    original_archive = archive.read_bytes()
    lab.transport.losses = [loss] * runtime.TERMINAL_RECOVERY_ATTEMPTS
    before_calls = len(lab.transport.calls)
    assert lab.recover() == 2
    assert len(lab.transport.calls) - before_calls == runtime.TERMINAL_RECOVERY_ATTEMPTS
    assert lab.journal().pending() == [original]
    assert not (lab.config.state_root / "idle.json").exists()
    assert manifest.read_bytes() == original_bytes and archive.read_bytes() == original_archive
    lab.clock[0] += 400
    assert lab.recover() == 0
    saved = history(lab.journal(), original["key"])
    assert len(saved) == (2 if loss == "undelivered" else 1)
    assert json.loads(saved[0]["request"]) == original["request"]
    assert saved[-1]["response"] is not None
    if loss == "undelivered":
        assert {proof["request"]["payload"]["method"] for proof in json.loads(saved[1]["reconciliation"])} == {
            "status", "get_operation"}
    assert manifest.read_bytes() == original_bytes and archive.read_bytes() == original_archive
    assert_finished_once(lab, before)


def test_recovery_reuses_durable_assignment_if_collector_receipt_still_missing(lab, monkeypatch):
    before = operation_count(lab)
    lab.stats.ack = False
    lab.transport.losses = ["committed"]
    crash_before_retry(lab, monkeypatch)
    lab.stats.ack = True
    monkeypatch.setattr(ShoreWorker, "start", lambda *_: pytest.fail("Cannot start terminal work"))
    assert lab.recover() == 0
    assert lab.stats.messages.count("complete") == 2
    assert_finished_once(lab, before)


@pytest.mark.parametrize("state", ["PREPARED", "ACCEPTED", "STARTING"])
def test_public_recovery_refuses_nonterminal_before_client_construction(lab, monkeypatch, state):
    lab.transport.losses = ["committed"]
    crash_before_retry(lab, monkeypatch)
    gate = lab.gate()
    with gate.transaction() as db:
        db.execute("UPDATE inbox SET state=?,result=NULL", (state,))
    monkeypatch.setattr(runtime, "_clients", lambda *_: pytest.fail("Nonterminal cannot open identity"))
    with pytest.raises(ValueError, match="nonterminal or uncertain"):
        runtime.recover_worker_terminal(lab.config)
    assert lab.stats.executions == 1
    assert not (lab.config.state_root / "idle.json").exists()


def test_actual_executor_process_loss_stays_starting_and_cannot_recover(lab, monkeypatch):
    before = operation_count(lab)

    def lose_process(*_args, **_kwargs):
        raise ProcessLost("executor returned before durable result commit")

    monkeypatch.setattr(ShoreWorker, "save_result", lose_process)
    with pytest.raises(ProcessLost, match="before durable result"):
        lab.run()
    record = lab.gate()._record("attempt-1")
    assert record["state"] == "STARTING" and record["result"] is None
    calls = list(lab.transport.calls)
    monkeypatch.setattr(runtime, "_clients", lambda *_: pytest.fail("Unknown execution cannot open identity"))
    with pytest.raises(ValueError, match="nonterminal or uncertain"):
        runtime.recover_worker_terminal(lab.config)
    assert lab.stats.executions == 1 and lab.stats.uploads == 0
    assert lab.transport.calls == calls
    assert operation_count(lab) == before
    assert not (lab.config.state_root / "idle.json").exists()


@pytest.mark.parametrize("artifact", ["result", "archive"])
def test_changed_artifacts_rejected_before_recovery_rpc(lab, monkeypatch, artifact):
    lab.transport.losses = ["committed"]
    original = crash_before_retry(lab, monkeypatch)
    target = Path(lab.gate().result("attempt-1")["manifest_path"] if artifact == "result"
                  else lab.gate().delivery("attempt-1")["archive"])
    target.write_bytes(b"tampered-result")
    before = list(lab.transport.calls)
    with pytest.raises(ValueError):
        lab.recover()
    assert lab.transport.calls == before
    assert lab.journal().pending() == [original]
    assert not (lab.config.state_root / "idle.json").exists()


def test_recovery_after_resolved_before_idle_does_not_reexecute_or_claim(lab, monkeypatch):
    before = operation_count(lab)
    lab.transport.losses = ["committed"]
    persist = runtime._persist_worker_idle
    monkeypatch.setattr(runtime, "_persist_worker_idle", lambda *_: (_ for _ in ()).throw(ProcessLost()))
    with pytest.raises(ProcessLost):
        lab.run()
    monkeypatch.setattr(runtime, "_persist_worker_idle", persist)
    assert lab.gate().is_resolved("attempt-1")
    calls = list(lab.transport.calls)
    assert lab.recover() == 0
    assert lab.transport.calls == calls
    assert_finished_once(lab, before)


@pytest.mark.parametrize("record", ["launch", "ready"])
def test_recovery_requires_original_signed_launch_and_readiness(lab, monkeypatch, record):
    lab.transport.losses = ["committed"]
    original = crash_before_retry(lab, monkeypatch)
    path = lab.ready if record == "ready" else lab.config.state_root / ("launch." + "1" * 32 + ".json")
    value = json.loads(path.read_bytes())
    value["payload"]["runtime_sha256"] = "f" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(runtime, "_clients", lambda *_: pytest.fail("Changed launch cannot open identity"))
    with pytest.raises(ValueError):
        runtime.recover_worker_terminal(lab.config)
    assert lab.journal().pending() == [original]
    assert lab.stats.idles == []


@pytest.mark.parametrize("extra", ["--wake-startup", "--ready-record"])
def test_recovery_cli_refuses_replacement_launch_paths(lab, extra):
    script = Path(__file__).resolve().parents[1] / "scripts/simulation/shore_synthetic_maid.py"
    spec = spec_from_file_location("shore_maid_recovery_cli_boundary", script)
    entry = module_from_spec(spec)
    spec.loader.exec_module(entry)
    with pytest.raises(SystemExit) as error:
        entry.main(["--runtime-config", "NEVER-READ.json", "--recover-terminal", extra, "NEVER-READ.json"])
    assert error.value.code == 2
    assert lab.transport.calls == []
