"""Review R1/R4 recovery through the actual synthetic Princess/Maid paths."""
from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("shore_control")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.crypto import public_key_text

from test_shore_project_integration import project  # noqa: F401
from test_shore_project_integration import ProcessLost
from msabp_opt.simulation.distributed.state import PrincessState


def not_started(p):
    assignment = p.request("request_task")["payload"]
    worker = p.worker()
    worker.control_worker.accept(assignment, run_id="run-1", worker_id="maid-a",
        row_sha256=assignment["row_sha256"], csv_sha256=worker.config.csv_sha256)
    worker.control_worker.save_result(assignment["attempt_id"], p.root / assignment["attempt_id"],
                                     outcome="not_started")
    assert worker._run_assignment(assignment) == "not_started"
    return assignment


def resume(p, suffix):
    key = Ed25519PrivateKey.generate()
    p.gate.resume(session_client=p.authority_client("session:" + suffix, key),
                  session_id=suffix, session_public_key=public_key_text(key.public_key()), mode="resume")


@pytest.mark.parametrize("project", [1], indirect=True)
def test_not_started_limit_is_visible_and_grant_does_not_reset_history(project):  # noqa: F811
    p = project
    p.hello()
    limit = p.runtime.max_attempts
    assignments = [not_started(p) for _ in range(limit)]
    reply = p.request("request_task")
    assert reply["type"] == "retry_paused"
    assert reply["payload"]["cases"] == [{"case_id": "0", "attempt_count": limit, "max_attempts": limit}]
    assert p.executions == 0
    _, budget = p.gate.build_checkpoint()
    assert budget["settled_units"] == 0 and budget["pending_units"] == 128

    p.gate.begin_drain("explicit synthetic pause")
    assert p.gate.finish_drain()["phase"] == "PAUSED_BOUND"
    resume(p, "session-b")
    assert p.worker().run() == 2  # A restarted Maid sees a visible pause, not infinite wait.
    state = p.runtime.state
    with PrincessState(state.path, control_authority=p.gate) as reopened:
        assert reopened.control_retry_pauses("run-1", max_attempts=limit) == reply["payload"]["cases"]
        reopened.grant_control_retries("run-1", "0", additional_attempts=2,
                                      grant_id="manual-extension-1", reason="synthetic operator retry")
    state.grant_control_retries("run-1", "0", additional_attempts=2,
                               grant_id="manual-extension-1", reason="synthetic operator retry")
    assignments += [not_started(p) for _ in range(2)]
    assert p.request("request_task")["type"] == "retry_paused"
    assert [a["attempt_number"] for a in assignments] == list(range(1, limit + 3))
    assert len({a["attempt_id"] for a in assignments}) == limit + 2
    assert len({a["control_authority"]["permit"]["payload"]["permit_id"] for a in assignments}) == limit + 2
    assert state.get_task("run-1", "0")["attempt_count"] == limit + 2
    with sqlite3.connect(state.path) as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == limit + 2
    p.gate.begin_drain("synthetic retry allowance exhausted again")
    assert p.gate.finish_drain()["phase"] == "PAUSED_BOUND"
    assert p.gate.build_checkpoint()[1]["settled_units"] == 0


@pytest.mark.parametrize("project", [1], indirect=True)
def test_retry_grants_are_explicit_bounded_and_gated(project):  # noqa: F811
    p = project
    state = p.runtime.state
    for invalid in (0, -1, True, 1.2):
        with pytest.raises(ValueError):
            state.grant_control_retries("run-1", "0", additional_attempts=invalid, grant_id="bad", reason="test")
    state.grant_control_retries("run-1", "0", additional_attempts=1, grant_id="one", reason="operator")
    with pytest.raises(Exception, match="changed content"):
        state.grant_control_retries("run-1", "0", additional_attempts=2, grant_id="one", reason="operator")
    p.gate.begin_drain("synthetic control removed")
    with pytest.raises(Exception, match="drain"):
        state.grant_control_retries("run-1", "0", additional_attempts=1, grant_id="two", reason="operator")
    assert p.gate.build_checkpoint()[1]["pending_units"] == 128


@pytest.mark.parametrize("project", [1], indirect=True)
def test_unsent_report_expiry_restart_collects_real_fake_bytes_during_drain(project, monkeypatch):  # noqa: F811
    p = project
    p.hello()
    assignment = p.request("request_task")["payload"]
    clock = [p.authority.clock()]
    monkeypatch.setattr(p.authority, "clock", lambda: clock[0])
    handle = p.transport.handle
    interrupted = False
    def crash_before_report(request):
        nonlocal interrupted
        if request["payload"]["method"] == "worker_report" and not interrupted:
            interrupted = True
            clock[0] += p.authority.challenge_ms / 1000 + 1
            raise ProcessLost("durable report request never reached authority")
        return handle(request)
    monkeypatch.setattr(p.transport, "handle", crash_before_report)
    with pytest.raises(ProcessLost):
        p.worker()._run_assignment(assignment)
    assert p.executions == 1
    p.gate.begin_drain("synthetic key removal after result but before receipt")
    assert p.worker().run() == 0
    assert p.executions == 1
    assert p.gate.finish_drain()["phase"] == "PAUSED_BOUND"
    assert p.gate.build_checkpoint()[1]["settled_units"] == 1


@pytest.mark.parametrize("project", [1], indirect=True)
@pytest.mark.parametrize("rotate", [False, True])
def test_before_issue_crash_can_drain_and_later_run_new_operation(project, monkeypatch, rotate):  # noqa: F811
    p = project
    p.hello()
    original = p.gate.issue_attempt
    def crash(_intent):
        raise RuntimeError("synthetic crash before facade issuance")
    monkeypatch.setattr(p.gate, "issue_attempt", crash)
    with pytest.raises(Exception, match="synthetic crash"):
        p.request("request_task")
    monkeypatch.setattr(p.gate, "issue_attempt", original)
    with sqlite3.connect(p.runtime.state.path) as db:
        before = db.execute("SELECT operation_id,attempt_id,descriptor_json FROM control_dispatch_intents").fetchone()
    p.gate.begin_drain("fence intent before any permit exists")
    if rotate:
        key = Ed25519PrivateKey.generate()
        p.gate.resume(session_client=p.authority_client("session:recovery", key), session_id="recovery",
                      session_public_key=public_key_text(key.public_key()), mode="recover")
    assert p.runtime.state.reconcile_unissued_intents("run-1") == (before[0],)
    assert p.runtime.state.reconcile_unissued_intents("run-1") == ()
    with sqlite3.connect(p.runtime.state.path) as db:
        row = db.execute("SELECT state,unissued_proof_json,descriptor_json FROM control_dispatch_intents").fetchone()
    assert row[0] == "cancelled_unissued" and json.loads(row[1])
    assert row[2] == before[2]
    assert p.gate.finish_drain()["phase"] == "PAUSED_BOUND"
    assert p.gate.build_checkpoint()[1]["settled_units"] == 0
    resume(p, "resumed")
    assignment = p.request("request_task")["payload"]
    assert assignment["attempt_id"] != before[1]
    assert assignment["control_authority"]["descriptor"]["operation_id"] != before[0]
    assert p.worker()._run_assignment(assignment) == "completed"
    assert p.executions == 1
    p.gate.begin_drain("synthetic completed")
    assert p.gate.finish_drain()["phase"] == "PAUSED_BOUND"
    assert p.gate.build_checkpoint()[1]["settled_units"] == 1
