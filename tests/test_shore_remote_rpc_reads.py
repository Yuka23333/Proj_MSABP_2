"""R2 read recovery: ephemeral authorities/keys only, no remote or CST access."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("shore_control")

from shore_control.authority import digest
from shore_control.client import Client, MemorySigner
from shore_control.crypto import verify_object
from shore_control.permits import DESCRIPTOR_FIELDS

from msabp_opt.simulation.distributed.control_remote_bell import BellRunner
from msabp_opt.simulation.distributed.control_authority import ControlError
from msabp_opt.simulation.distributed.control_remote_worker import (
    IDLE_DOMAIN, MAID_ACTOR, _persist_worker_idle,
)
from msabp_opt.simulation.distributed.control_rpc import SignedRpcJournal, VerifiedRpcError
from msabp_opt.simulation.distributed.control_worker import ShoreWorker

from test_shore_remote_bell import env as _bell_fixture
from test_shore_remote_worker import remote_worker as _maid_fixture

bell_env = _bell_fixture
remote_worker = _maid_fixture


class DropRead:
    """Save the authority result, then simulate a lost get_operation reply."""

    def __init__(self, authority):
        self.authority = authority
        self.drop = False
        self.before_delivery = False
        self.calls = []

    def challenge(self, actor):
        return self.authority.challenge(actor)

    def handle(self, request):
        self.calls.append(request)
        drop = self.drop and request["payload"]["method"] == "get_operation"
        if drop:
            self.drop = False
            if self.before_delivery:
                raise TimeoutError("read persisted but not yet delivered")
        reply = self.authority.handle(request)
        if drop:
            raise TimeoutError("get_operation committed, reply lost")
        return reply


def _history(journal, key):
    with journal.transaction() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM rpc_generations WHERE key=? ORDER BY generation", (key,))]


def _assert_original_read_preserved(journal, pending, transport):
    history = _history(journal, pending["key"])
    assert len(history) == 1
    assert json.loads(history[0]["request"]) == pending["request"]
    assert history[0]["response"] is not None
    # Exact replay verifies the old nonterminal snapshot, not today's status.
    old = journal.proof(pending["key"])
    assert old["response"]["payload"]["result"]["status"] == "ACCEPTED"
    original_calls = [request for request in transport.calls if request == pending["request"]]
    assert len(original_calls) == 2
    assert journal.pending() == []


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("expired", [False, True])
def test_maid_lost_read_then_resolved_and_idle(remote_worker, restart, expired):
    env = remote_worker
    clock = [time.time() + 1]
    env.authority.clock = lambda: clock[0]
    transport = DropRead(env.authority)
    env.worker_client.transport = transport
    cfg = env.config

    def make_worker():
        return ShoreWorker(cfg.state_root / "control", env.worker_client,
            env.keys["authority"].public_key(), binding=cfg["binding"],
            registration=env.bootstrap["workers"][MAID_ACTOR], csv_sha256=cfg["csv_sha256"])

    binding = {name: cfg["binding"][name] for name in ("job_id", "epoch", "host_id", "session_id")}
    envelope = env.controller.call("validate_dispatch", **binding, operation={
        "operation_id": "read-recovery-attempt", "run_id": cfg["run_id"], "worker_id": MAID_ACTOR,
        "attempt_id": "attempt-1", "action": "attempt", "input_sha256": digest({"row": 1}),
        "action_context": None})
    assignment = {"attempt_id": "attempt-1", "control_authority": {
        "descriptor": {name: envelope["payload"][name] for name in DESCRIPTOR_FIELDS}, "permit": envelope}}
    worker = make_worker()
    worker.accept(assignment, run_id=cfg["run_id"], worker_id=MAID_ACTOR,
                  row_sha256=digest({"row": 1}), csv_sha256=cfg["csv_sha256"])
    worker.start("attempt-1")
    result_root = env.root / "attempt-result"
    result_root.mkdir()
    (result_root / "manifest.json").write_text('{"synthetic":true}', encoding="utf-8")
    result = worker.save_result("attempt-1", result_root, outcome="completed")
    worker.report("attempt-1")
    transport.drop = True
    with pytest.raises(TimeoutError, match="reply lost"):
        worker.resolved("attempt-1")
    pending, = worker.rpc.pending()
    session = Client(env.authority, "session:" + binding["session_id"], MemorySigner(env.keys["session"]),
                     env.keys["authority"].public_key(), cfg["binding"]["fleet_id"])
    assert session.call("controller_ack", **result["receipt"])["status"] == "RESOLVED"
    mutation_calls = [r for r in transport.calls if r["payload"]["method"] != "get_operation"]
    if expired:
        clock[0] += 400
    if restart:
        worker = make_worker()
    assert worker.resolved("attempt-1")
    _assert_original_read_preserved(worker.rpc, pending, transport)
    assert [r for r in transport.calls if r["payload"]["method"] != "get_operation"] == mutation_calls
    assert worker.pending_assignments() == []
    sent = []
    collector = SimpleNamespace(request_json=lambda path, body: (
        sent.append((path, body)) or {"status": "RECORDED", "actor_id": MAID_ACTOR}))
    _persist_worker_idle(cfg, worker, env.worker_client, collector)
    payload = verify_object(env.bootstrap["workers"][MAID_ACTOR]["public_key"], IDLE_DOMAIN,
                            sent[0][1]["record"])
    assert sent[0][0] == "/v1/control/idle"
    assert payload["permit_ids"] == [envelope["payload"]["permit_id"]]
    assert payload["outboxes_pending"] == 0


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("expired", [False, True])
def test_bell_lost_read_then_resolved_and_idle(bell_env, restart, expired):
    env = bell_env
    clock = [time.time() + 1]
    env.authority.clock = lambda: clock[0]
    transport = DropRead(env.authority)
    env.clients["bell"].transport = transport
    bell, assignment, launches = env.coordinator(), env.assignment(), []
    bell.accept(assignment)
    bell.launch("wake-attempt-1", lambda *paths: launches.append(paths), ready_path=env.tmp_path / "ready")
    result = bell.record_ready("wake-attempt-1", env.ready(bell))
    bell.report("wake-attempt-1")
    transport.drop = True
    with pytest.raises(TimeoutError, match="reply lost"):
        bell.resolved("wake-attempt-1")
    pending, = bell.rpc.pending()
    env.gate.ack_evidence(assignment["control_authority"]["permit"], result["evidence"], result["root"])
    mutation_calls = [r for r in transport.calls if r["payload"]["method"] != "get_operation"]
    if expired:
        clock[0] += 400
    if restart:
        bell = env.coordinator()
    assert bell.resolved("wake-attempt-1")
    _assert_original_read_preserved(bell.rpc, pending, transport)
    assert len(launches) == 1
    assert [r for r in transport.calls if r["payload"]["method"] != "get_operation"] == mutation_calls
    sent = []
    collector = SimpleNamespace(request_json=lambda path, body: (
        sent.append((path, body)) or {"status": "RECORDED", "actor_id": "bell"}))
    runner = BellRunner(bell, collector, lambda *_: pytest.fail("No relaunch"), ready_root=env.tmp_path / "ready")
    assert runner.report_idle() == {"status": "RECORDED", "actor_id": "bell"}
    assert sent[0][0] == "/v1/control/idle"


def test_fresh_read_recovers_every_matching_request_not_other_identity(bell_env):
    env = bell_env
    bell, assignment = env.coordinator(), env.assignment()
    bell.accept(assignment)
    permit = assignment["control_authority"]["permit"]["payload"]
    params = {"epoch": permit["epoch"], "operation_id": permit["operation_id"]}
    journal = bell.rpc
    for key in ("lost-read-1", "lost-read-2"):
        env.transport.drop_method = "get_operation"
        with pytest.raises(TimeoutError):
            journal.call(env.clients["bell"], "get_operation", params, key=key)
    env.transport.drop_method = "status"
    with pytest.raises(TimeoutError):
        journal.call(env.clients["maid"], "status", {}, key="other-identity")
    before = list(journal.pending())
    result = journal.fresh_read(env.clients["bell"], "get_operation", dict(reversed(list(params.items()))))
    assert result["permit"] == assignment["control_authority"]["permit"]
    assert [p["key"] for p in journal.pending()] == ["other-identity"]
    for pending in before[:2]:
        assert journal.proof(pending["key"])["request"] == pending["request"]
    assert journal.proof("other-identity") is None
    with pytest.raises(ValueError, match="status/get_operation"):
        journal.fresh_read(env.clients["bell"], "accept_permit", {})


def test_fresh_read_retains_historical_unknown_permit_but_reads_current(bell_env):
    env = bell_env
    journal = SignedRpcJournal(env.tmp_path / "separate-rpc.sqlite")
    params = {"epoch": env.binding["epoch"], "operation_id": "wake-1"}
    env.transport.drop_method = "get_operation"
    with pytest.raises(TimeoutError):
        journal.call(env.clients["bell"], "get_operation", params, key="before-issuance")
    original, = journal.pending()
    assignment = env.assignment()
    assert journal.fresh_read(env.clients["bell"], "get_operation", params)["permit"] == assignment["control_authority"]["permit"]
    proof = journal.proof("before-issuance")
    assert proof["request"] == original["request"]
    assert proof["response"]["payload"]["error"]["code"] == "UNKNOWN_PERMIT"
    assert journal.pending() == []


def test_fresh_read_expired_undelivered_request_keeps_nonce_recovery_proofs(bell_env):
    env = bell_env
    clock = [time.time() + 1]
    env.authority.clock = lambda: clock[0]
    env.authority.challenge_ms = 1000
    transport = DropRead(env.authority)
    env.clients["bell"].transport = transport
    bell, assignment = env.coordinator(), env.assignment()
    permit = assignment["control_authority"]["permit"]["payload"]
    params = {"epoch": permit["epoch"], "operation_id": permit["operation_id"]}
    transport.drop = transport.before_delivery = True
    with pytest.raises(TimeoutError, match="not yet delivered"):
        bell.rpc.fresh_read(env.clients["bell"], "get_operation", params)
    original, = bell.rpc.pending()
    clock[0] += 2
    restarted = env.coordinator()
    assert restarted.rpc.fresh_read(env.clients["bell"], "get_operation", params)["permit"] == assignment["control_authority"]["permit"]
    history = _history(restarted.rpc, original["key"])
    assert len(history) == 2
    assert json.loads(history[0]["request"]) == original["request"]
    assert history[1]["response"] is not None
    assert {p["request"]["payload"]["method"] for p in json.loads(history[1]["reconciliation"])} == {"status", "get_operation"}
    assert restarted.rpc.pending() == []


def test_fresh_read_does_not_skip_a_still_uncertain_original_request(bell_env):
    env = bell_env
    transport = DropRead(env.authority)
    env.clients["bell"].transport = transport
    bell, assignment = env.coordinator(), env.assignment()
    permit = assignment["control_authority"]["permit"]["payload"]
    params = {"epoch": permit["epoch"], "operation_id": permit["operation_id"]}
    transport.drop = True
    with pytest.raises(TimeoutError):
        bell.rpc.fresh_read(env.clients["bell"], "get_operation", params)
    original, = bell.rpc.pending()
    transport.drop = True
    with pytest.raises(TimeoutError):
        bell.rpc.fresh_read(env.clients["bell"], "get_operation", params)
    assert bell.rpc.pending() == [original]
    assert transport.calls == [original["request"], original["request"]]
    assert len(_history(bell.rpc, original["key"])) == 1
    assert bell.rpc.proof(original["key"]) is None


def _lose_first_wake_accept_before_delivery(env):
    clock = [time.time() + 1]
    env.authority.clock = lambda: clock[0]
    env.authority.challenge_ms = 1000
    original_handle, lost = env.transport.handle, False

    def handle(request):
        nonlocal lost
        if request["payload"]["method"] == "accept_permit" and not lost:
            lost = True
            raise TimeoutError("accept journaled, not submitted")
        return original_handle(request)

    env.transport.handle = handle
    bell, assignment = env.coordinator(), env.assignment()
    with pytest.raises(TimeoutError, match="not submitted"):
        bell.accept(assignment)
    pending, = bell.rpc.pending()
    return bell, assignment, pending, clock


def test_terminal_not_started_wake_can_settle_expired_accept_as_signed_refusal(bell_env):
    env = bell_env
    bell, assignment, original, clock = _lose_first_wake_accept_before_delivery(env)
    permit = assignment["control_authority"]["permit"]
    env.gate.begin_drain("synthetic drain before first wake acceptance")
    clock[0] += 2
    bell = env.coordinator()

    def recover():
        return bell.rpc.call(env.clients["bell"], "accept_permit", original["params"],
                             key=original["key"], recovery_permit=permit)

    # DRAINING alone is no permission to renew an unaccepted business call.
    with pytest.raises(ControlError, match="drain barrier"):
        recover()
    assert len(_history(bell.rpc, original["key"])) == 1
    result = bell.not_started("wake-attempt-1")
    bell.report("wake-attempt-1")
    with pytest.raises(ControlError, match="receipted work"):
        recover()
    assert len(_history(bell.rpc, original["key"])) == 1
    env.gate.ack_evidence(permit, result["evidence"], result["root"])
    with pytest.raises(VerifiedRpcError) as refusal:
        recover()
    assert refusal.value.code == "WORK_CLOSED"
    history = _history(bell.rpc, original["key"])
    assert len(history) == 2
    assert json.loads(history[0]["request"]) == original["request"]
    terminal = json.loads(history[1]["response"])
    assert terminal["payload"]["ok"] is False
    assert terminal["payload"]["error"]["code"] == "WORK_CLOSED"
    probes = json.loads(history[1]["reconciliation"])
    operation = next(p["response"]["payload"]["result"] for p in probes
                     if p["request"]["payload"]["method"] == "get_operation")
    assert operation["status"] == "RESOLVED"
    assert operation["worker_receipt"] == operation["controller_receipt"]
    assert operation["worker_receipt"]["outcome"] == "not_started"
    assert operation["acceptance"] is None
    assert bell.rpc.pending() == []
    assert bell.resolved("wake-attempt-1")
    assert bell.record("wake-attempt-1")["launch_id"] is None
    assert env.gate.status()["bound_job"]["phase"] == "DRAINING"
    assert env.gate.status()["bound_job"]["epoch"] == permit["payload"]["epoch"]
    collector = SimpleNamespace(request_json=lambda *_: {"status": "RECORDED", "actor_id": "bell"})
    assert BellRunner(bell, collector, lambda *_: pytest.fail("No launch"),
                      ready_root=env.tmp_path / "ready").report_idle()["status"] == "RECORDED"


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_other_terminal_outcomes_do_not_open_accept_refresh_exception(bell_env, outcome):
    env = bell_env
    bell, assignment, original, clock = _lose_first_wake_accept_before_delivery(env)
    permit = assignment["control_authority"]["permit"]
    # Independently establish a closed authority operation, without changing
    # the original pending journal or claiming an external process was run.
    env.clients["bell"].call("accept_permit", **original["params"])
    receipt = {"permit_id": permit["payload"]["permit_id"], "outcome": outcome,
               "durable": True, "evidence_sha256": digest({"test-only-terminal": outcome})}
    env.clients["bell"].call("worker_report", **receipt)
    env.clients["session"].call("controller_ack", **receipt)
    clock[0] += 2
    with pytest.raises(ControlError, match="receipted work"):
        bell.rpc.call(env.clients["bell"], "accept_permit", original["params"],
                      key=original["key"], recovery_permit=permit)
    assert len(_history(bell.rpc, original["key"])) == 1
    assert bell.rpc.pending() == [original]
