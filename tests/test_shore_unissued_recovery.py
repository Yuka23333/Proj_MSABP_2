"""Unissued-intent fencing with isolated, in-process synthetic Shore authorities."""
from __future__ import annotations

import copy
import json
import sqlite3

import pytest

from test_control_authority import env, intent  # noqa: F401
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.authority import AuthorityError
from shore_control.crypto import public_key_text

from msabp_opt.simulation.distributed.control_authority import ControlAuthority, ControlError
from msabp_opt.simulation.distributed.state import PrincessState


def controlled_state(environment):
    environment["state"].close()
    state = PrincessState(environment["tmp_path"] / "project.sqlite",
                          control_authority=environment["gate"])
    state.bind_control_run("run-1")
    return state


def rows(environment, query):
    with sqlite3.connect(environment["tmp_path"] / "project.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(query)]


def rotate(environment, *, mode, session_id):
    key = Ed25519PrivateKey.generate()
    client = environment["client"]("session:" + session_id, key)
    return environment["gate"].resume(session_client=client, session_id=session_id,
        session_public_key=public_key_text(key.public_key()), mode=mode)


def crash_before_issue(environment, state, monkeypatch):
    def unavailable(_intent):
        raise TimeoutError("synthetic process loss before first authority request")

    with monkeypatch.context() as patch:
        patch.setattr(environment["gate"], "issue_attempt", unavailable)
        with pytest.raises(TimeoutError):
            state.claim_next("run-1", "maid-a", now=10)
    saved = rows(environment, "SELECT * FROM control_dispatch_intents")[0]
    return {**json.loads(saved["intent_json"]), "descriptor": json.loads(saved["descriptor_json"])}


@pytest.mark.parametrize("rotate_first", [False, True])
def test_before_issue_crash_drain_recover_resume_preserves_audit_and_budget(env, monkeypatch, rotate_first):  # noqa: F811
    gate = env["gate"]
    with controlled_state(env) as state:
        original = crash_before_issue(env, state, monkeypatch)
        descriptor = original["descriptor"]
        dispatch = {k: descriptor[k] for k in ("job_id", "epoch", "host_id", "session_id")}
        dispatch["operation"] = {k: descriptor[k] for k in (
            "operation_id", "run_id", "worker_id", "attempt_id", "action", "input_sha256", "action_context")}
        # Even a prepared old-session signed request must stay fenced after resume.
        late_dispatch = env["controller"].prepare("validate_dispatch", dispatch)
        if rotate_first:
            assert rotate(env, mode="recover", session_id="session-b")["phase"] == "DRAINING"
        else:
            gate.begin_drain("synthetic before-issue crash")
        state.reconcile_unissued_intents("run-1")
        saved = rows(env, "SELECT * FROM control_dispatch_intents")[0]
        proof = json.loads(saved["unissued_proof_json"])
        assert saved["state"] == "cancelled_unissued" and saved["permit_json"] is None
        assert json.loads(saved["descriptor_json"]) == descriptor
        assert gate.verify_unissued_intent(original, proof) == proof
        assert gate.reconcile_unissued_intent(original) == proof
        assert rows(env, "SELECT * FROM attempts") == []
        assert state.get_task("run-1", "0")["attempt_count"] == 0
        assert gate.finish_drain()["phase"] == "PAUSED_BOUND"
        audit = gate._get("reconciliation")
        assert audit["manifest"]["settled_units"] == 0
        assert audit["manifest"]["pending_units"] == 128
        assert audit["evaluations"] == audit["operations"] == []
        assert audit["cancelled_unissued"][0]["proof"] == proof
        assert rotate(env, mode="resume", session_id="session-c")["phase"] == "ACTIVE"
        with pytest.raises(AuthorityError) as error:
            env["controller"].send(late_dispatch)
        assert error.value.code == "FENCED"
        with pytest.raises(ControlError, match="cannot be dispatched"):
            gate.issue_attempt(original)
        fresh = state.claim_next("run-1", "maid-a", now=20)
        assert fresh is not None and fresh.case_id == "0"
        assert fresh.attempt_id != original["attempt_id"]
        assert fresh.control_authority["descriptor"]["operation_id"] != original["operation_id"]
        assert fresh.control_authority["descriptor"]["session_id"] == "session-c"
        assert len(rows(env, "SELECT * FROM control_dispatch_intents")) == 2
        assert gate.status()["unresolved_count"] == 1


def test_cancelled_proof_survives_facade_and_state_restart(env, monkeypatch):  # noqa: F811
    with controlled_state(env) as state:
        original = crash_before_issue(env, state, monkeypatch)
        env["gate"].begin_drain("synthetic restart")
        state.reconcile_unissued_intents("run-1")
    reopened = ControlAuthority(env["tmp_path"] / "project-control.sqlite",
        controller_client=env["controller"], session_client=env["session"],
        worker_registrations=env["registry"])
    with PrincessState(env["tmp_path"] / "project.sqlite", control_authority=reopened) as state:
        state.reconcile_unissued_intents("run-1")
        assert reopened.reconcile_unissued_intent(original)
        assert reopened.finish_drain()["phase"] == "PAUSED_BOUND"


def test_crash_between_facade_cancellation_and_state_commit_replays_same_proof(env, monkeypatch):  # noqa: F811
    gate = env["gate"]
    with controlled_state(env) as state:
        original = crash_before_issue(env, state, monkeypatch)
        gate.begin_drain("synthetic cross-journal cancellation crash")
        reconcile = gate.reconcile_unissued_intent

        def crash_after_facade(intent_data):
            reconcile(intent_data)
            raise TimeoutError("synthetic process loss after facade proof commit")

        with monkeypatch.context() as patch:
            patch.setattr(gate, "reconcile_unissued_intent", crash_after_facade)
            with pytest.raises(TimeoutError, match="facade proof commit"):
                state.reconcile_unissued_intents("run-1")
        assert rows(env, "SELECT state FROM control_dispatch_intents")[0]["state"] == "intent"
        with gate.journal.transaction() as connection:
            saved = dict(connection.execute("SELECT * FROM control_operations").fetchone())
        assert saved["state"] == "cancelled_unissued"
        proof = json.loads(saved["unissued_proof_json"])
        with pytest.raises(ControlError, match="unresolved persistent intent"):
            gate.finish_drain()
    with PrincessState(env["tmp_path"] / "project.sqlite", control_authority=gate) as restarted:
        restarted.reconcile_unissued_intents("run-1")
        assert gate.verify_unissued_intent(original, proof) == proof
        assert json.loads(rows(env, "SELECT unissued_proof_json FROM control_dispatch_intents")[0]["unissued_proof_json"]) == proof
        assert gate.finish_drain()["phase"] == "PAUSED_BOUND"


def test_existing_permit_response_loss_never_becomes_unissued_cancellation(env):  # noqa: F811
    gate = env["gate"]
    with controlled_state(env) as state:
        env["transport"].method = "validate_dispatch"
        with pytest.raises(TimeoutError):
            state.claim_next("run-1", "maid-a", now=10)
        gate.begin_drain("synthetic issued reply loss")
        with pytest.raises(ControlError, match="Existing authority permit"):
            state.reconcile_unissued_intents("run-1")
        saved = rows(env, "SELECT * FROM control_dispatch_intents")[0]
        assert saved["state"] == "intent" and saved["unissued_proof_json"] is None
        with pytest.raises(ControlError, match="unresolved persistent intent"):
            gate.finish_drain()
        recovered = state.claim_next("run-1", "maid-a", now=20)
        assert recovered.attempt_id == saved["attempt_id"]
        assert gate.status()["unresolved_count"] == 1
        with pytest.raises(ControlError):
            gate.finish_drain()


def test_unsigned_unknown_permit_cannot_cancel(env, monkeypatch):  # noqa: F811
    gate = env["gate"]
    with controlled_state(env) as state:
        crash_before_issue(env, state, monkeypatch)
        gate.begin_drain("synthetic unsigned error")
        original_handle = env["transport"].handle

        def unsigned_unknown(request):
            if request["payload"]["method"] == "get_operation":
                raise AuthorityError("UNKNOWN_PERMIT", "unsigned transport is not absence evidence")
            return original_handle(request)

        monkeypatch.setattr(env["transport"], "handle", unsigned_unknown)
        with pytest.raises(AuthorityError) as error:
            state.reconcile_unissued_intents("run-1")
        assert error.value.code == "UNKNOWN_PERMIT"
        saved = rows(env, "SELECT * FROM control_dispatch_intents")[0]
        assert saved["state"] == "intent" and saved["unissued_proof_json"] is None
        with pytest.raises(ControlError, match="unresolved persistent intent"):
            gate.build_checkpoint()


@pytest.mark.parametrize("component", ["fence", "absence"])
def test_tampered_signed_proof_and_state_proof_fail_closed(env, monkeypatch, component):  # noqa: F811
    gate = env["gate"]
    with controlled_state(env) as state:
        original = crash_before_issue(env, state, monkeypatch)
        gate.begin_drain("synthetic proof tamper test")
        state.reconcile_unissued_intents("run-1")
        proof = json.loads(rows(env, "SELECT * FROM control_dispatch_intents")[0]["unissued_proof_json"])
        changed = copy.deepcopy(proof)
        changed[component]["response"]["payload"]["request_sha256"] = "f" * 64
        with pytest.raises(ControlError, match="signed unissued proof"):
            gate.verify_unissued_intent(original, changed)
        with sqlite3.connect(env["tmp_path"] / "project.sqlite") as connection:
            connection.execute("UPDATE control_dispatch_intents SET unissued_proof_json=?",
                               (json.dumps(changed),))
        with pytest.raises(ControlError, match="differs from signed facade proof"):
            gate.build_checkpoint()


def test_local_drain_flag_without_authority_fence_is_not_cancellation(env):  # noqa: F811
    gate = env["gate"]
    prepared = intent()
    prepared["descriptor"] = gate.descriptor_for_attempt(prepared)
    gate._set("local_drain", True)
    with pytest.raises(ControlError, match="authority-proven session fence"):
        gate.reconcile_unissued_intent(prepared)
    assert not [request for request in env["transport"].seen
                if request["payload"]["method"] == "get_operation"]


def test_valid_historical_absence_without_fencing_does_not_cancel(env):  # noqa: F811
    gate = env["gate"]
    prepared = intent()
    prepared["descriptor"] = gate.descriptor_for_attempt(prepared)
    with pytest.raises(AuthorityError):
        gate.get_operation(prepared["operation_id"])
    issued = gate.issue_attempt(prepared)
    gate.begin_drain("synthetic prior absence then permit")
    with pytest.raises(ControlError, match="Existing permit"):
        gate.reconcile_unissued_intent(prepared)
    assert issued["permit"] and gate.status()["unresolved_count"] == 1


def test_unsent_signed_dispatch_is_retained_and_remains_fenced_after_resume(env, monkeypatch):  # noqa: F811
    gate = env["gate"]
    prepared = intent()
    prepared["descriptor"] = gate.descriptor_for_attempt(prepared)
    original_handle = env["transport"].handle

    def before_delivery(request):
        if request["payload"]["method"] == "validate_dispatch":
            raise TimeoutError("synthetic loss before first dispatch delivery")
        return original_handle(request)

    with monkeypatch.context() as patch:
        patch.setattr(env["transport"], "handle", before_delivery)
        with pytest.raises(TimeoutError):
            gate.issue_attempt(prepared)
    pending = gate.journal.pending()[0]
    gate.begin_drain("synthetic unsent durable dispatch")
    proof = gate.reconcile_unissued_intent(prepared)
    assert gate.journal.pending() == [pending]
    assert gate.verify_unissued_intent(prepared, proof) == proof
    assert gate.finish_drain()["phase"] == "PAUSED_BOUND"
    rotate(env, mode="resume", session_id="session-b")
    with pytest.raises(AuthorityError) as error:
        gate.journal.call(env["controller"], pending["method"], pending["params"], key=pending["key"])
    assert error.value.code == "FENCED"
    assert gate.journal.proof(pending["key"])["request"] == pending["request"]
    assert gate.status()["unresolved_count"] == 0
