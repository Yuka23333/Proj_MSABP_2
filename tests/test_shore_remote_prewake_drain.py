"""Default-lease, no-launch drain recovery; isolated Authority and keys only."""
# ruff: noqa: F811 -- imported pytest fixture is injected by name
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from test_shore_remote_controller import project  # noqa: F401
from msabp_opt.simulation.distributed import control_remote_controller as controller
from msabp_opt.simulation.distributed.control_authority import ControlError
from msabp_opt.simulation.distributed.control_remote_bell import BellRunner
from msabp_opt.simulation.distributed.control_remote_worker import BELL_ACTOR, LOCAL_NODE_ID


def runner(p):
    return BellRunner(p.bell, SimpleNamespace(request_json=lambda path, body:
        p.queue.handle(path, body, actor_id=BELL_ACTOR, node_id=LOCAL_NODE_ID)),
        lambda *_: pytest.fail("Drain must never launch Maid"), ready_root=p.tmp_path / "ready")


def advance(p, seconds):
    old_clock = p.authority.clock
    p.authority.clock = lambda: old_clock() + seconds


def assert_paused(p):
    original = copy.deepcopy(p.gate.binding)
    assert p.queue.idle_pending() == 0
    assert p.gate.finish_drain(worker_outboxes_pending=0)["phase"] == "PAUSED_BOUND"
    assert p.gate.binding == original
    assert p.gate.status()["unresolved_count"] == 0
    audit = p.gate._get("reconciliation")
    assert audit["manifest"]["settled_units"] == 0
    assert audit["manifest"]["pending_units"] == p.points
    assert p.gate.poll_control()["phase"] == "PAUSED_BOUND"
    with pytest.raises(ControlError, match="explicit user close"):
        p.gate.release(disposition="closed", reason="No implicit abandonment")


@pytest.mark.parametrize("expire", [False, True])
def test_no_wake_drain_has_signed_absence_not_fictitious_idle(project, expire):
    p = project
    if expire:
        # The fixture uses Authority's unchanged default, NOT a 300-second lease.
        assert p.authority.lease_ms == 30000
        advance(p, 31)
        with pytest.raises(Exception):
            p.gate.poll_control()
    else:
        p.gate.begin_drain("F removed before first wake")
    assert p.gate.job["phase"] == "DRAINING"
    assert p.poll() == {"assignment": None}
    assert not p.bell.pending()
    assert not (p.queue.root / "idle").exists()
    assert (p.queue.root / "unissued.json").is_file()
    assert_paused(p)
    assert not [r for r in p.transport.requests if r["payload"]["method"] in
                {"validate_dispatch", "accept_permit"}]
    restarted = controller.CollectorWakeQueue(p.queue.root, p.gate, p.runtime)
    assert restarted.idle_pending() == 0


@pytest.mark.parametrize("stage", ["reply_lost", "prepared", "accepted"])
def test_issued_not_started_drains_without_requiring_impossible_maid_idle(project, stage):
    p = project
    if stage == "reply_lost":
        p.transport.drop_method = "validate_dispatch"
        with pytest.raises(TimeoutError):
            p.poll()
    else:
        assignment = p.poll()["assignment"]
        (p.bell.prepare if stage == "prepared" else p.bell.accept)(assignment)
    p.gate.begin_drain("Stop before STARTING")
    recovered = p.poll()["assignment"]
    permit = recovered["control_authority"]["permit"]["payload"]
    bell = runner(p)
    assert bell.step()["state"] == "RESOLVED"
    assert p.bell.record(permit["attempt_id"])["launch_id"] is None
    assert p.bell.checked_result(permit["attempt_id"])["receipt"]["outcome"] == "not_started"
    assert bell.step()["state"] == "IDLE_REPORTED"
    assert not list((p.tmp_path / "maid-state").glob("**/idle.json"))
    assert_paused(p)
    with p.gate.journal.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM control_operations").fetchone()[0] == 1


@pytest.mark.parametrize("submitted", [False, True])
def test_pending_accept_nonce_expiry_keeps_tombstone_and_original_request(project, monkeypatch, submitted):
    p = project
    assignment = p.poll()["assignment"]
    original_handle = p.transport.handle

    def lost(request):
        if request["payload"]["method"] == "accept_permit":
            if submitted:
                original_handle(request)
            raise TimeoutError("Lost accept before/after commit")
        return original_handle(request)

    with monkeypatch.context() as patch:
        patch.setattr(p.transport, "handle", lost)
        with pytest.raises(TimeoutError):
            p.bell.accept(assignment)
    original_pending = p.bell.rpc.pending()[0]
    advance(p, 130)
    p.gate.begin_drain("Expired pre-launch accept")
    bell = runner(p)
    assert bell.step()["state"] == "RESOLVED"
    assert p.bell.rpc.pending() == []
    assert p.bell.rpc.proof(original_pending["key"]) is not None
    with p.bell.rpc.transaction() as db:
        saved = db.execute("SELECT request FROM rpc_generations WHERE key=? AND generation=0",
                           (original_pending["key"],)).fetchone()[0]
    assert json.loads(saved) == original_pending["request"]
    assert bell.step()["state"] == "IDLE_REPORTED"
    assert_paused(p)


def test_starting_without_ready_is_still_unknown_and_cannot_be_closed(project):
    p = project
    permit = p.bell.accept(p.poll()["assignment"])
    p.bell.launch(permit["attempt_id"], lambda *_: None, ready_path=p.tmp_path / "unused")
    p.gate.begin_drain("Unknown process state")
    assert runner(p).step()["state"] == "STARTING"
    assert p.queue.idle_pending() == 2
    with pytest.raises(ControlError):
        p.gate.finish_drain(worker_outboxes_pending=2)
    assert not (p.queue.root / "unissued.json").exists()


def test_local_stop_or_corrupted_absence_never_proves_zero_work(project):
    p = project
    p.gate._set("local_drain", True)
    with pytest.raises(ControlError, match="fence"):
        p.queue.idle_pending()
    p.gate.begin_drain("Real fence now")
    assert p.queue.idle_pending() == 0
    path = p.queue.root / "unissued.json"
    proof = json.loads(path.read_bytes())
    proof["absence"]["response"]["payload"]["request_sha256"] = "f" * 64
    path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ControlError):
        p.queue.idle_pending()


def test_explicit_close_only_after_proved_pause(project, monkeypatch):
    p = project
    root = p.preparation["output_root"]
    controller.immutable_json(controller.Path(root) / "deployment/maid-runtime.json", p.runtime)
    with pytest.raises(ValueError, match="PAUSED_BOUND"):
        controller.close_job(p.preparation, p.gate, reason="Explicit operator cancellation")
    p.gate.begin_drain("Cancel before wake")
    assert_paused(p)
    monkeypatch.setattr(controller, "CollectorWakeQueue", lambda *_: p.queue)
    assert controller.close_job(p.preparation, p.gate, reason="Explicit operator cancellation")["phase"] == "RELEASED"


def test_unsent_dispatch_expired_nonce_remains_fenced_with_durable_proof(project, monkeypatch):
    p = project
    original = p.transport.handle

    def unsent(request):
        if request["payload"]["method"] == "validate_dispatch":
            raise TimeoutError("Never delivered")
        return original(request)

    with monkeypatch.context() as patch:
        patch.setattr(p.transport, "handle", unsent)
        with pytest.raises(TimeoutError):
            p.poll()
    advance(p, 130)
    p.gate.begin_drain("Fence expired unsent dispatch")
    assert p.poll() == {"assignment": None}
    assert_paused(p)
    assert p.gate._get("reconciliation")["cancelled_unissued"]


def test_close_response_durable_but_report_write_lost_replays_original_release(project, monkeypatch):
    p = project
    root = controller.Path(p.preparation["output_root"])
    controller.immutable_json(root / "deployment/maid-runtime.json", p.runtime)
    p.gate.begin_drain("Explicit stop")
    assert_paused(p)
    monkeypatch.setattr(controller, "CollectorWakeQueue", lambda *_: p.queue)
    persist = controller.immutable_json

    def lose_report(path, value):
        if path.name == "explicit-close-report.json":
            raise OSError("Crash after release commit")
        return persist(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "immutable_json", lose_report)
        with pytest.raises(OSError):
            controller.close_job(p.preparation, p.gate, reason="Operator approved cancellation")
    assert p.gate.job["phase"] == "RELEASED"
    requests = len(p.transport.requests)
    report = controller.close_job(p.preparation, p.gate, reason="Retry same close")
    assert report["phase"] == "RELEASED"
    assert len(p.transport.requests) == requests
    assert controller.serve_job(p.preparation, p.gate, {}, None, collector_config="NEVER-READ") == report
