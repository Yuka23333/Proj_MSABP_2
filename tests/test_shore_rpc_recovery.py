"""R2: isolated, synthetic authority only; no services or external execution."""
from __future__ import annotations

import copy
import json
import sqlite3
import time

import pytest

pytest.importorskip("shore_control")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.authority import Authority, AuthorityError, digest
from shore_control.client import Client, MemorySigner
from shore_control.crypto import canonical_json, public_key_text, sign_object, verify_object

from msabp_opt.simulation.distributed.control_authority import ControlError
from msabp_opt.simulation.distributed.control_rpc import SignedRpcJournal, VerifiedRpcError

from test_control_authority import env as _env_fixture, intent
from test_shore_maid import lab as _maid_fixture

env = _env_fixture
lab = _maid_fixture


class Crash(BaseException):
    pass


@pytest.fixture
def clocked(env):
    clock = [time.time() + 1]
    env["authority"].clock = lambda: clock[0]
    env["authority"].challenge_ms = 1000
    env["clock"] = clock
    # Also makes these tests runnable while the facade import is being updated.
    env["gate"].journal = SignedRpcJournal(env["gate"].journal.path)
    return env


def lose_before_delivery(env, method):
    original = env["transport"].handle
    lost = False

    def handle(request):
        nonlocal lost
        if request["payload"]["method"] == method and not lost:
            lost = True
            raise TimeoutError("request persisted, never delivered")
        return original(request)

    env["transport"].handle = handle
    return original


def prepared(env, method):
    gate = env["gate"]
    if method == "validate_dispatch":
        descriptor = gate.descriptor_for_attempt(intent())
        operation = {k: descriptor[k] for k in (
            "operation_id", "run_id", "worker_id", "attempt_id", "action", "input_sha256", "action_context")}
        return env["controller"], {**gate.binding, "operation": operation}, None
    permit = gate.issue_attempt(intent())["permit"]
    payload = permit["payload"]
    accept = {k: payload[k] for k in (
        "permit_id", "operation_id", "run_id", "attempt_id", "action", "input_sha256", "device_id")}
    if method == "accept_permit":
        return env["worker"], accept, permit
    gate.journal.call(env["worker"], "accept_permit", accept,
                      key="seed-accept", recovery_permit=permit)
    receipt = {"permit_id": payload["permit_id"], "outcome": "completed",
               "durable": True, "evidence_sha256": digest({"synthetic": "durable-result"})}
    if method == "controller_ack":
        gate.journal.call(env["worker"], "worker_report", receipt,
                          key="seed-report", recovery_permit=permit)
    return env["session"] if method == "controller_ack" else env["worker"], receipt, permit


def generations(journal, key):
    with journal.transaction() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM rpc_generations WHERE key=? ORDER BY generation", (key,))]


@pytest.mark.parametrize("method", ["validate_dispatch", "accept_permit", "worker_report", "controller_ack"])
def test_first_delivery_expired_reconciles_same_work_and_keeps_history(clocked, method):
    env = clocked
    client, params, permit = prepared(env, method)
    journal = env["gate"].journal
    lose_before_delivery(env, method)
    with pytest.raises(TimeoutError):
        journal.call(client, method, params, key="target", recovery_permit=permit)
    original = journal.pending()[0]["request"]
    env["clock"][0] += 2
    journal = SignedRpcJournal(journal.path)
    result = journal.call(client, method, params, key="target", recovery_permit=permit)
    history = generations(journal, "target")
    assert len(history) == 2
    assert json.loads(history[0]["request"]) == original
    fresh = json.loads(history[1]["request"])
    assert fresh["payload"]["params"] == original["payload"]["params"]
    assert fresh["payload"]["actor_id"] == original["payload"]["actor_id"]
    assert fresh["payload"]["request_id"] != original["payload"]["request_id"]
    assert fresh["payload"]["nonce"] != original["payload"]["nonce"]
    evidence = json.loads(history[1]["reconciliation"])
    assert {p["request"]["payload"]["method"] for p in evidence} >= {"status", "get_operation"}
    for proof in evidence:
        verify_object(client.public_key, "response", proof["response"])
    assert journal.proof("target")["request"] == fresh
    assert not journal.pending()
    with sqlite3.connect(env["authority"].database) as db:
        assert db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 1
    assert result


@pytest.mark.parametrize("method", ["validate_dispatch", "accept_permit", "worker_report", "controller_ack"])
def test_committed_reply_lost_replays_original_after_expiry(clocked, method):
    env = clocked
    client, params, permit = prepared(env, method)
    journal = env["gate"].journal
    env["transport"].method = method
    with pytest.raises(TimeoutError):
        journal.call(client, method, params, key="committed", recovery_permit=permit)
    original = journal.pending()[0]["request"]
    env["clock"][0] += 400  # lease AND challenge expired: exact replay still wins.
    restarted = SignedRpcJournal(journal.path)
    restarted.call(client, method, params, key="committed", recovery_permit=permit)
    assert len(generations(restarted, "committed")) == 1
    assert restarted.proof("committed")["request"] == original
    with restarted.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM rpc_recovery_probes").fetchone()[0] == 0


@pytest.mark.parametrize("boundary", ["_after_reconcile", "_after_refresh"])
def test_repeated_crashes_across_refresh_boundaries_keep_all_requests(clocked, monkeypatch, boundary):
    env = clocked
    client, params, permit = prepared(env, "worker_report")
    journal = env["gate"].journal
    lose_before_delivery(env, "worker_report")
    with pytest.raises(TimeoutError):
        journal.call(client, "worker_report", params, key="crashing", recovery_permit=permit)
    original = journal.pending()[0]["request"]

    def crash(_key):
        raise Crash("synthetic process loss")

    for _ in range(3):
        env["clock"][0] += 2
        journal = SignedRpcJournal(journal.path)
        monkeypatch.setattr(journal, boundary, crash)
        with pytest.raises(Crash):
            journal.call(client, "worker_report", params, key="crashing")
        assert journal.pending()
    env["clock"][0] += 2
    journal = SignedRpcJournal(journal.path)
    assert journal.call(client, "worker_report", params, key="crashing")["status"] == "AWAITING_PEER_RECEIPT"
    history = generations(journal, "crashing")
    assert json.loads(history[0]["request"]) == original
    assert len(history) == (5 if boundary == "_after_refresh" else 2)
    with journal.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM rpc_recovery_probes WHERE response IS NOT NULL").fetchone()[0] >= 8


@pytest.mark.parametrize("method", ["validate_dispatch", "accept_permit"])
def test_no_new_dispatch_or_acceptance_after_signed_drain(clocked, method):
    env = clocked
    client, params, permit = prepared(env, method)
    journal = env["gate"].journal
    lose_before_delivery(env, method)
    with pytest.raises(TimeoutError):
        journal.call(client, method, params, key="blocked", recovery_permit=permit)
    env["session"].call("begin_drain", **env["gate"].binding, reason="test drain barrier")
    env["clock"][0] += 2
    with pytest.raises(ControlError, match="ACTIVE|drain barrier"):
        journal.call(client, method, params, key="blocked")
    assert len(generations(journal, "blocked")) == 1
    with sqlite3.connect(env["authority"].database) as db:
        assert db.execute("SELECT COUNT(*) FROM operations WHERE acceptance IS NOT NULL").fetchone()[0] == 0


def test_accepted_worker_and_current_collector_can_finish_in_drain(clocked):
    env = clocked
    client, receipt, permit = prepared(env, "worker_report")
    journal = env["gate"].journal
    lose_before_delivery(env, "worker_report")
    with pytest.raises(TimeoutError):
        journal.call(client, "worker_report", receipt, key="report", recovery_permit=permit)
    env["gate"].begin_drain("finish accepted work only")
    env["clock"][0] += 2
    assert journal.call(client, "worker_report", receipt, key="report")["status"] == "AWAITING_PEER_RECEIPT"
    lose_before_delivery(env, "controller_ack")
    with pytest.raises(TimeoutError):
        journal.call(env["session"], "controller_ack", receipt, key="ack", recovery_permit=permit)
    env["clock"][0] += 2
    assert journal.call(env["session"], "controller_ack", receipt, key="ack")["status"] == "RESOLVED"
    assert env["gate"].status()["unresolved_count"] == 0


@pytest.mark.parametrize("method,allowed", [("worker_report", True), ("controller_ack", False)])
def test_session_rotation_preserves_worker_finish_but_fences_old_collector(clocked, method, allowed):
    env = clocked
    client, receipt, permit = prepared(env, method)
    journal = env["gate"].journal
    lose_before_delivery(env, method)
    with pytest.raises(TimeoutError):
        journal.call(client, method, receipt, key="rotated", recovery_permit=permit)
    new_key = Ed25519PrivateKey.generate()
    env["gate"].resume(session_client=env["client"]("session:session-b", new_key),
        session_id="session-b", session_public_key=public_key_text(new_key.public_key()), mode="recover")
    env["clock"][0] += 2
    if allowed:
        assert journal.call(client, method, receipt, key="rotated")["status"] == "AWAITING_PEER_RECEIPT"
        assert len(generations(journal, "rotated")) == 2
    else:
        with pytest.raises(VerifiedRpcError) as error:
            journal.call(client, method, receipt, key="rotated")
        assert error.value.code == "AUTH"
        assert len(generations(journal, "rotated")) == 1
        assert journal.proof("rotated") is None


def test_stale_dispatch_cannot_refresh_without_present_controller_signer(clocked):
    env = clocked
    client, params, _permit = prepared(env, "validate_dispatch")
    journal = env["gate"].journal
    lose_before_delivery(env, "validate_dispatch")
    with pytest.raises(TimeoutError):
        journal.call(client, "validate_dispatch", params, key="no-controller")
    env["clock"][0] += 2
    env["signer"].present = False
    with pytest.raises(FileNotFoundError):
        journal.call(client, "validate_dispatch", params, key="no-controller")
    assert len(generations(journal, "no-controller")) == 1


@pytest.mark.parametrize("attack", ["unsigned_status", "bad_signature", "wrong_binding", "wrong_work"])
def test_unsigned_stale_is_not_permission_and_untrusted_probes_never_refresh(clocked, attack):
    env = clocked
    client, params, permit = prepared(env, "worker_report")
    journal = env["gate"].journal
    handle = env["transport"].handle

    def malicious(request):
        method = request["payload"]["method"]
        if method == "worker_report":
            raise AuthorityError("CHALLENGE_STALE", "unsigned attacker claims stale")
        if method == "status" and attack == "unsigned_status":
            raise AuthorityError("CHALLENGE_STALE", "still unsigned")
        response = handle(request)
        if method == "status" and attack == "bad_signature":
            response = copy.deepcopy(response)
            response["payload"]["result"]["bound_job"]["phase"] = "RELEASED"
        elif method == "status" and attack == "wrong_binding":
            body = copy.deepcopy(response["payload"])
            body["request_id"] = "another-request"
            response = sign_object(env["keys"]["authority"], "response", body)
        elif method == "get_operation" and attack == "wrong_work":
            body = copy.deepcopy(response["payload"])
            body["result"]["permit"]["payload"]["operation_id"] = "another-operation"
            response = sign_object(env["keys"]["authority"], "response", body)
        return response

    env["transport"].handle = malicious
    with pytest.raises((AuthorityError, ValueError, ControlError)):
        journal.call(client, "worker_report", params, key="attack", recovery_permit=permit)
    assert len(generations(journal, "attack")) == 1
    assert journal.proof("attack") is None
    assert journal.pending()[0]["key"] == "attack"


def test_signed_business_refusal_is_durable_and_full_proof_is_exposed(clocked):
    env = clocked
    journal = env["gate"].journal
    with pytest.raises(VerifiedRpcError) as first:
        journal.call(env["session"], "get_operation", {"epoch": 1, "operation_id": "absent"}, key="refusal")
    assert isinstance(first.value, AuthorityError)
    assert first.value.code == "UNKNOWN_PERMIT"
    assert first.value.proof == journal.proof("refusal")
    verify_object(env["session"].public_key, "response", first.value.response)
    before = len(env["transport"].seen)
    with pytest.raises(VerifiedRpcError) as cached:
        SignedRpcJournal(journal.path).call(env["session"], "get_operation", {"epoch": 1, "operation_id": "absent"}, key="refusal")
    assert cached.value.proof == first.value.proof
    assert len(env["transport"].seen) == before
    assert journal.pending() == []


@pytest.mark.parametrize("method", ["status", "get_operation", "renew"])
def test_fresh_recovers_expired_pending_read_or_renew_without_historical_authorization(clocked, method):
    env = clocked
    gate = env["gate"]
    if method == "get_operation":
        gate.issue_attempt(intent())
    params = {} if method == "status" else ({"epoch": 1, "operation_id": "operation-1"} if method == "get_operation" else gate.binding)
    role = "controller" if method == "renew" else "session"
    lose_before_delivery(env, method)
    with pytest.raises(TimeoutError):
        gate._fresh(role, method, params)
    original = gate.journal.pending()[0]["request"]
    env["clock"][0] += 2
    result = gate._fresh(role, method, params)
    assert result
    assert not gate.journal.pending()
    requests = [r for r in env["transport"].seen if r["payload"]["method"] == method]
    assert requests[0] == original
    assert requests[-1]["payload"]["request_id"] != original["payload"]["request_id"]


def test_legacy_unsigned_error_migrates_as_audit_not_cached_authority(clocked, tmp_path):
    env = clocked
    client = env["session"]
    request = client.prepare("status", {})
    path = tmp_path / "legacy.sqlite"
    def encoded(value):
        return canonical_json(value).decode("utf-8")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE rpc_journal(key TEXT PRIMARY KEY,identity TEXT,method TEXT,params TEXT,request TEXT,result TEXT,error TEXT)")
        db.execute("INSERT INTO rpc_journal VALUES(?,?,?,?,?,NULL,?)", (
            "old", encoded({"actor_id": client.actor_id, "authority_id": client.authority_id, "fleet_id": client.fleet_id}),
            "status", "{}", encoded(request), encoded({"code": "RPC", "message": "unverified historical transport error"})))
    journal = SignedRpcJournal(path)
    assert journal.call(client, "status", {}, key="old")["bound_job"]["phase"] == "ACTIVE"
    history = generations(journal, "old")
    assert history[0]["legacy_error"] is not None
    assert json.loads(history[0]["request"]) == request
    assert journal.proof("old")["response"]


def test_changed_params_actor_or_recovery_permit_cannot_reuse_business_key(clocked):
    env = clocked
    client, params, permit = prepared(env, "worker_report")
    journal = env["gate"].journal
    lose_before_delivery(env, "worker_report")
    with pytest.raises(TimeoutError):
        journal.call(client, "worker_report", params, key="immutable", recovery_permit=permit)
    with pytest.raises(ControlError, match="identity or content"):
        journal.call(client, "worker_report", {**params, "outcome": "failed"}, key="immutable")
    with pytest.raises(ControlError, match="identity or content"):
        journal.call(env["session"], "worker_report", params, key="immutable")
    altered = copy.deepcopy(permit["payload"])
    altered["operation_id"] = "different-op"
    with pytest.raises(ControlError, match="cannot change"):
        journal.call(client, "worker_report", params, key="immutable",
                     recovery_permit=sign_object(env["keys"]["authority"], "permit", altered))


def test_stale_first_acquire_recovers_only_same_business_on_signed_empty_fleet(clocked, tmp_path):
    env = clocked
    authority = Authority(tmp_path / "new-authority.sqlite", env["keys"]["authority"],
        fleet_id=env["authority"].fleet_id,
        controllers={"controller": public_key_text(env["keys"]["controller"].public_key())},
        workers=env["registry"], clock=lambda: env["clock"][0], challenge_seconds=1)
    transport = type(env["transport"])(authority)
    client = Client(transport, "controller", MemorySigner(env["keys"]["controller"]),
                    env["keys"]["authority"].public_key(), authority.fleet_id)
    journal = SignedRpcJournal(tmp_path / "new-client.sqlite")
    lose_before_delivery({"transport": transport}, "acquire")
    params = env["gate"]._get("acquire")
    with pytest.raises(TimeoutError):
        journal.call(client, "acquire", params, key="acquire")
    env["clock"][0] += 2
    result = journal.call(client, "acquire", params, key="acquire")
    assert result["job_id"] == params["job_id"]
    assert len(generations(journal, "acquire")) == 2
    with sqlite3.connect(authority.database) as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_stale_session_recovery_reuses_exact_previous_and_new_session(clocked):
    env = clocked
    gate = env["gate"]
    new_key = Ed25519PrivateKey.generate()
    new_client = env["client"]("session:session-b", new_key)
    lose_before_delivery(env, "resume_bound_job")
    kwargs = dict(session_client=new_client, session_id="session-b",
                  session_public_key=public_key_text(new_key.public_key()), mode="recover")
    with pytest.raises(TimeoutError):
        gate.resume(**kwargs)
    env["clock"][0] += 2
    assert gate.resume(**kwargs)["phase"] == "DRAINING"
    assert gate.binding["session_id"] == "session-b"
    history = generations(gate.journal, "resume:session-b")
    assert len(history) == 2
    assert json.loads(history[0]["request"])["payload"]["params"] == json.loads(history[1]["request"])["payload"]["params"]


def test_actual_maid_result_and_both_expired_receipts_recover_once_and_finish_drain(lab):
    # This fixture uses an in-process fake collector (no HTTP listener), actual
    # Maid/ShoreWorker entrypoints and real synthetic result bytes on disk.
    worker = lab.maid()
    original = lab.transport.handle
    crash_method = ["worker_report"]

    def crash_before(request):
        if request["payload"]["method"] == crash_method[0]:
            crash_method[0] = None
            raise Crash("durable request before its first authority delivery")
        return original(request)

    lab.transport.handle = crash_before
    collector_journal = SignedRpcJournal(lab.root / "collector-rpc.sqlite")
    original_call = lab.session.call

    def collector_call(method, **params):
        if method == "controller_ack":
            return collector_journal.call(lab.session, method, params,
                key="collector-ack", recovery_permit=lab.permit)
        return original_call(method, **params)

    lab.session.call = collector_call
    with pytest.raises(Crash):
        worker._run_assignment(lab.assignment)
    assert lab.executions == 1
    assert worker.control_worker.result("attempt-1") is not None
    lab.session.call("begin_drain", **lab.bound, reason="recover completed outboxes")
    lab.now[0] += 16
    crash_method[0] = "controller_ack"
    with pytest.raises(Crash):
        lab.maid()._run_assignment(lab.assignment)
    lab.now[0] += 16
    assert lab.maid()._run_assignment(lab.assignment) == "completed"
    assert lab.executions == 1
    assert len(generations(worker.control_worker.rpc, "report:" + lab.permit["payload"]["permit_id"])) == 2
    assert len(generations(collector_journal, "collector-ack")) == 2
    snapshot = lab.session.call("status")
    assert snapshot["unresolved_count"] == 0
    job = snapshot["bound_job"]
    spec = job["scope"]["job_spec"]
    manifest = {"schema": "shore.job-checkpoint", "schema_version": 1,
        "fleet_id": job["fleet_id"], "job_id": job["job_id"], "epoch": job["epoch"],
        "bound_host_id": job["host_id"], "collector_url": job["scope"]["collector_url"],
        "data_root": job["scope"]["data_root"], "job_spec_sha256": digest(spec),
        "approved_units": 2, "settled_units": 1, "pending_units": 1,
        "budget_unit": spec["budget_unit"], "accounting_policy_sha256": spec["accounting_policy_sha256"],
        "runs": [{"run_id": "run-1", "original_plan_sha256": digest({"plan": 1}),
                  "input_manifest_sha256": lab.config.csv_sha256,
                  "accounting_ledger_sha256": digest({"completed": "attempt-1"})}],
        "budget_ledger_sha256": digest({"settled": 1, "pending": 1}),
        "reconciliation_manifest_sha256": digest(worker.control_worker.result("attempt-1")["receipt"])}
    checkpoint = {"manifest_sha256": digest(manifest), "pending_count": 1,
        "uploads_in_progress": 0, "worker_outboxes_pending": 0, "receipts_pending": 0,
        "local_unresolved_count": 0, "gpu_uncollected": 0}
    paused = lab.session.call("finish_drain", **lab.bound, checkpoint=checkpoint, job_checkpoint=manifest)
    assert paused["phase"] == "PAUSED_BOUND"
