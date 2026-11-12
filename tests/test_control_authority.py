"""Isolated synthetic Shore/actual project-state adapter regressions.

No loopback service, lab database, external solver or real identities are used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("shore_control", reason="explicit optional Shore checkout required")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.authority import Authority, AuthorityError, digest
from shore_control.client import Client, MemorySigner
from shore_control.crypto import canonical_json, public_key_text
from shore_control.permits import DESCRIPTOR_FIELDS

from msabp_opt.simulation.distributed.control_authority import ControlAuthority, ControlError
from msabp_opt.simulation.distributed.state import PrincessState, freeze_csv

from test_shore_project_integration import project  # noqa: F401


class DropReply:
    def __init__(self, authority):
        self.authority = authority
        self.method = None
        self.seen = []

    def challenge(self, actor):
        return self.authority.challenge(actor)

    def handle(self, request):
        self.seen.append(request)
        response = self.authority.handle(request)
        if request["payload"]["method"] == self.method:
            self.method = None
            raise TimeoutError("synthetic response loss after durable authority commit")
        return response


class RemovableSigner(MemorySigner):
    present = True

    def sign(self, domain, payload):
        if not self.present:
            raise FileNotFoundError("synthetic controller removed")
        return super().sign(domain, payload)


@pytest.fixture
def env(tmp_path, request):
    approved_units = getattr(request, "param", 128)
    keys = {name: Ed25519PrivateKey.generate() for name in ("authority", "controller", "session-a", "maid-a")}
    registry = {"maid-a": {"public_key": public_key_text(keys["maid-a"].public_key()),
        "device_id": "synthetic-device", "tailnet_node_id": "synthetic-node", "role": "maid", "allowed_actions": ["attempt"]}}
    authority = Authority(tmp_path / "authority.sqlite", keys["authority"], fleet_id="isolated-test-fleet",
        controllers={"controller": public_key_text(keys["controller"].public_key())}, workers=registry,
        lease_seconds=300)
    transport = DropReply(authority)
    signer = RemovableSigner(keys["controller"])

    def client(actor, key=None, custom_signer=None):
        return Client(transport, actor, custom_signer or MemorySigner(key or keys[actor]),
                      keys["authority"].public_key(), "isolated-test-fleet")

    controller = client("controller", custom_signer=signer)
    session = client("session:session-a", keys["session-a"])
    worker = client("maid-a")
    gate = ControlAuthority(tmp_path / "project-control.sqlite", controller_client=controller,
        session_client=session, worker_clients={"maid-a": worker}, worker_registrations=registry)
    manifest = tmp_path / "full-job.json"
    policy = tmp_path / "policy.json"
    manifest.write_bytes(canonical_json({"job": "synthetic-complete-job", "approved_units": approved_units}))
    policy.write_bytes(canonical_json({"schema": "msabp.budget-policy", "schema_version": 1,
                                     "unit_key": "evaluation_id", "settled_outcomes": ["completed"]}))
    spec = {"schema": "shore.job-spec", "schema_version": 1,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "approved_units": approved_units,
        "budget_unit": "objective_evaluation", "accounting_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}
    gate.acquire(job_id="synthetic-complete-job", host_id="test-host", session_id="session-a",
        session_public_key=public_key_text(keys["session-a"].public_key()), project_id="msabp-test",
        collector_url="http://100.64.0.1:8090", data_root=r"D:\Synthetic\ControlTest", job_spec=spec)
    gate.bind_job_documents(manifest, policy)
    source = tmp_path / "source.csv"
    source.write_text("sample_id,value\n" + "".join(f"{i},{i}\n" for i in range(32)), encoding="utf-8")
    frozen = freeze_csv(source, tmp_path / "run" / "input.csv")
    state = PrincessState(tmp_path / "project.sqlite")
    state.initialize_run("run-1", frozen)
    state.register_worker("run-1", "maid-a")
    with sqlite3.connect(tmp_path / "project.sqlite") as db:
        db.executescript("""CREATE TABLE control_collection(attempt_id TEXT PRIMARY KEY,request_json TEXT,settled INTEGER DEFAULT 0);
            CREATE TABLE control_uploads(upload_id TEXT PRIMARY KEY,attempt_id TEXT,status TEXT);""")
    plan = tmp_path / "original-plan.json"
    plan.write_bytes(canonical_json({"original_plan": "unchanged"}))
    gate.register_run("run-1", digest({"original_plan": "unchanged"}), frozen.sha256)
    gate.bind_run_plan("run-1", plan)
    gate.bind_run_state("run-1", tmp_path / "project.sqlite")
    return dict(locals())


def intent(number=1):
    return {"operation_id": f"operation-{number}", "run_id": "run-1", "worker_id": "maid-a",
            "attempt_id": f"attempt-{number}", "case_id": str(number), "input_sha256": digest({"sample": number})}


def test_dispatch_reply_loss_replays_exact_signed_request_after_restart(env):
    gate, transport = env["gate"], env["transport"]
    transport.method = "validate_dispatch"
    with pytest.raises(TimeoutError):
        gate.issue_attempt(intent())
    pending = gate.journal.pending()
    assert len(pending) == 1
    original = pending[0]["request"]
    assert gate.status()["unresolved_count"] == 1
    restarted = ControlAuthority(env["tmp_path"] / "project-control.sqlite",
        controller_client=env["controller"], session_client=env["session"],
        worker_registrations=env["registry"])
    result = restarted.issue_attempt(intent())
    requests = [r for r in transport.seen if r["payload"]["method"] == "validate_dispatch"]
    assert requests == [original, original]
    assert result["descriptor"]["operation_id"] == "operation-1"
    assert restarted.journal.pending() == []
    with pytest.raises(ControlError, match="intent changed"):
        restarted.issue_attempt({**intent(), "input_sha256": digest({"changed": True})})


def test_fresh_renew_never_reuses_historical_active_and_no_key_fallback(env):
    gate = env["gate"]
    gate.require_new_work()
    gate.require_new_work()
    requests = [r for r in env["transport"].seen if r["payload"]["method"] == "renew"]
    assert len({r["payload"]["request_id"] for r in requests}) == 2
    env["signer"].present = False
    with pytest.raises(FileNotFoundError):
        gate.require_new_work()
    assert gate.status()["bound_job"]["phase"] == "DRAINING"
    gate.begin_drain("synthetic controller removed")
    with pytest.raises(ControlError, match="draining"):
        gate.require_new_work()
    with pytest.raises(ControlError, match="drain"):
        gate.issue_attempt(intent())


def test_rpc_shaped_transport_error_keeps_exact_request_pending(env):
    gate, transport = env["gate"], env["transport"]
    original_handle = transport.handle
    failed = False

    def rpc_lost(request):
        nonlocal failed
        response = original_handle(request)
        if request["payload"]["method"] == "validate_dispatch" and not failed:
            failed = True
            raise AuthorityError("RPC", "synthetic HTTP timeout")
        return response

    transport.handle = rpc_lost
    with pytest.raises(AuthorityError) as exc:
        gate.issue_attempt(intent())
    assert exc.value.code == "RPC"
    request = gate.journal.pending()[0]["request"]
    gate.issue_attempt(intent())
    dispatches = [r for r in transport.seen if r["payload"]["method"] == "validate_dispatch"]
    assert dispatches == [request, request]
    assert gate.journal.pending() == []


def test_replayed_active_renew_response_is_not_current_authorization(env):
    gate, transport = env["gate"], env["transport"]
    transport.method = "renew"
    with pytest.raises(TimeoutError):
        gate._fresh("controller", "renew", gate.binding)
    old_request = gate.journal.pending()[0]["request"]
    env["session"].call("begin_drain", **gate.binding, reason="external drain barrier")
    with pytest.raises(AuthorityError) as exc:
        gate.require_new_work()
    assert exc.value.code == "NOT_ACTIVE"
    renews = [r for r in transport.seen if r["payload"]["method"] == "renew"]
    assert renews[:2] == [old_request, old_request]
    assert renews[2]["payload"]["request_id"] != old_request["payload"]["request_id"]


def test_unissued_persisted_descriptor_during_drain_never_requests_new_permit(env):
    gate = env["gate"]
    prepared = intent()
    prepared["descriptor"] = gate.descriptor_for_attempt(prepared)
    gate.begin_drain("synthetic intent persisted before facade call")
    with pytest.raises(AuthorityError) as exc:
        gate.issue_attempt(prepared)
    assert exc.value.code == "UNKNOWN_PERMIT"
    assert not [r for r in env["transport"].seen if r["payload"]["method"] == "validate_dispatch"]


def test_old_permit_replay_survives_drain_and_new_session(env):
    gate = env["gate"]
    result = gate.issue_attempt(intent())
    gate.begin_drain("synthetic drain")
    new_key = Ed25519PrivateKey.generate()
    new_client = env["client"]("session:session-b", new_key)
    gate.resume(session_client=new_client, session_id="session-b",
                session_public_key=public_key_text(new_key.public_key()), mode="recover")
    assert gate.issue_attempt(intent()) == result
    assert gate.binding["session_id"] == "session-b"
    assert result["descriptor"]["session_id"] == "session-a"
    with pytest.raises(ControlError):
        gate.issue_attempt(intent(2))


def test_zero_input_and_unknown_run_rejected(env):
    with pytest.raises(AuthorityError) as exc:
        env["gate"].issue_attempt({**intent(), "input_sha256": "0" * 64})
    assert exc.value.code == "INVALID"
    assert env["gate"].status()["unresolved_count"] == 0
    with pytest.raises(ControlError, match="registered"):
        env["gate"].issue_attempt({**intent(2), "run_id": "unknown"})


def complete_case(env, index):
    state, gate, worker = env["state"], env["gate"], env["worker"]
    claim = state.claim_next("run-1", "maid-a")
    operation = {"operation_id": "op-case-" + str(index), "run_id": "run-1", "worker_id": "maid-a",
                 "attempt_id": claim.attempt_id, "case_id": claim.case_id, "input_sha256": claim.row_sha256}
    issued = gate.issue_attempt(operation)
    permit = issued["permit"]["payload"]
    gate.journal.call(worker, "accept_permit", {k: permit[k] for k in (
        "permit_id", "operation_id", "run_id", "attempt_id", "action", "input_sha256", "device_id")},
        key="accept:" + claim.attempt_id)
    root = env["tmp_path"] / ("case-" + str(index))
    root.mkdir()
    data = b"synthetic verified result\n"
    (root / "Farfield Source [1].ffs").write_bytes(data)
    evidence = {"schema": "shore.operation-evidence", "schema_version": 1, "kind": "attempt",
        **{k: permit[k] for k in ("fleet_id", "job_id", "epoch", "permit_id", "worker_id", "device_id", "input_sha256")},
        "descriptor_sha256": digest({k: permit[k] for k in DESCRIPTOR_FIELDS}), "outcome": "completed",
        "artifacts": [{"path": "farfield.ffs", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
        "result": {"state": "completed", "execution_id": "execution-" + str(index),
                   "failure_code": None, "termination_record_sha256": None, "tombstone_sha256": None}}
    aliases = {"farfield.ffs": "Farfield Source [1].ffs"}
    gate.journal.call(worker, "worker_report", {"permit_id": permit["permit_id"], "outcome": "completed",
                     "evidence_sha256": digest(evidence), "durable": True}, key="report:" + claim.attempt_id)
    gate.ack_evidence(issued["permit"], evidence, root, aliases)
    state.complete_task("run-1", "maid-a", claim.attempt_id, claim.lease_token, artifact_path=root)
    record = gate.settle_evaluation("evaluation-" + str(index), "run-1", claim.case_id, claim.attempt_id)
    return claim, record


def test_128_budget_32_actual_evaluations_leave_96_future_pending(env):
    for i in range(32):
        complete_case(env, i)
    gate = env["gate"]
    gate.begin_drain("synthetic complete batch")
    result = gate.finish_drain()
    reconciliation = gate._get("reconciliation")
    assert reconciliation["manifest"]["settled_units"] == 32
    assert reconciliation["manifest"]["pending_units"] == 96
    assert len(reconciliation["evaluations"]) == 32
    assert result["phase"] == "PAUSED_BOUND"
    assert gate.status()["bound_job"]["job_id"] == "synthetic-complete-job"
    with pytest.raises(AuthorityError) as exc:
        gate.release(disposition="completed", reason="not actually complete")
    assert exc.value.code == "PENDING"
    with pytest.raises(ControlError, match="explicit user"):
        gate.release(disposition="closed", reason="not authorized")
    assert gate.release(disposition="closed", reason="synthetic explicit close authorization",
                        explicit_close=True)["phase"] == "RELEASED"
    assert gate.status()["bound_job"] is None


def test_settlement_rechecks_bytes_and_forbids_retry_double_count(env):
    claim, record = complete_case(env, 0)
    with pytest.raises(ControlError, match="another evaluation"):
        env["gate"].settle_evaluation("another-id", "run-1", claim.case_id, claim.attempt_id)
    (env["tmp_path"] / "case-0" / "Farfield Source [1].ffs").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        env["gate"].build_checkpoint()


@pytest.mark.parametrize("table,statement", [
    ("uploads", "INSERT INTO control_uploads VALUES('u1','a1','writing')"),
    ("collection", "INSERT INTO control_collection VALUES('a1','{}',0)"),
])
def test_actual_collector_upload_and_completion_journals_block_finish(env, table, statement):
    with sqlite3.connect(env["tmp_path"] / "project.sqlite") as db:
        db.execute(statement)
    with pytest.raises(ControlError, match="Collector"):
        env["gate"].build_checkpoint()


def test_all_registered_runs_need_state_provenance_and_immutable_inputs(env):
    gate = env["gate"]
    gate.register_run("run-2", digest({"plan": 2}), digest({"inputs": 2}))
    with pytest.raises(ControlError, match="provenance"):
        gate.build_checkpoint()
    with pytest.raises(ControlError, match="changed"):
        gate.register_run("run-1", digest({"new_plan": 1}), env["frozen"].sha256)


def test_actual_original_plan_bytes_must_remain_frozen(env):
    env["plan"].write_bytes(b"changed original plan")
    with pytest.raises(ControlError, match="original plan bytes changed"):
        env["gate"].build_checkpoint()


def test_clearing_explicit_inflight_counter_uses_a_new_finish_request(env):
    gate = env["gate"]
    gate.begin_drain("synthetic drain with outbox")
    with pytest.raises(AuthorityError) as exc:
        gate.finish_drain(worker_outboxes_pending=1)
    assert exc.value.code == "UNRESOLVED"
    result = gate.finish_drain(worker_outboxes_pending=0)
    assert result["phase"] == "PAUSED_BOUND"
    assert result["checkpoint"]["pending_count"] == 128


@pytest.mark.parametrize("env", [2], indirect=True)
def test_new_case_budget_is_reserved_before_permit_and_retry_charges_once(env):
    gate, transport = env["gate"], env["transport"]
    transport.method = "validate_dispatch"
    with pytest.raises(TimeoutError):
        gate.issue_attempt(intent(1))
    gate.issue_attempt(intent(2))
    before = len(transport.seen)
    with pytest.raises(ControlError, match="budget is already reserved"):
        gate.issue_attempt(intent(3))
    assert len(transport.seen) == before
    retry = {**intent(1), "operation_id": "retry-operation-1", "attempt_id": "retry-attempt-1"}
    gate.issue_attempt(retry)
    assert gate.status()["unresolved_count"] == 3
    with pytest.raises(ControlError, match="Retry changed"):
        gate.issue_attempt({**intent(3), "case_id": "1"})


def test_attempt_without_project_evaluation_id_is_rejected_before_rpc(env):
    missing = intent()
    del missing["case_id"]
    before = len(env["transport"].seen)
    with pytest.raises(ControlError, match="case_id"):
        env["gate"].issue_attempt(missing)
    assert len(env["transport"].seen) == before


def test_monitor_renews_long_running_work_then_automatically_drains_on_key_loss(env):
    gate = env["gate"]
    gate.issue_attempt(intent())
    assert gate.poll_control()["phase"] == "ACTIVE"
    env["signer"].present = False
    with pytest.raises(FileNotFoundError):
        gate.poll_control()
    assert gate.status()["bound_job"]["phase"] == "DRAINING"
    assert gate.status()["unresolved_count"] == 1
    before = len([r for r in env["transport"].seen if r["payload"]["method"] == "renew"])
    env["signer"].present = True
    assert gate.poll_control()["phase"] == "DRAINING"
    after = len([r for r in env["transport"].seen if r["payload"]["method"] == "renew"])
    assert before == after


@pytest.mark.parametrize("outcome", ["failed", "not_started"])
def test_checkpoint_rechecks_nonbudget_terminal_proof_bytes(project, outcome):  # noqa: F811
    project.hello()
    assignment = project.request("request_task")["payload"]

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic known termination")

    worker = project.worker(executor=fail if outcome == "failed" else None)
    if outcome == "not_started":
        worker.control_worker.accept(assignment, run_id="run-1", worker_id="maid-a",
            row_sha256=assignment["row_sha256"], csv_sha256=worker.config.csv_sha256)
        worker.control_worker.save_result(assignment["attempt_id"], project.root / "never-started-proof",
                                          outcome="not_started")
    assert worker._run_assignment(assignment) == outcome
    project.gate.begin_drain("synthetic terminal proof audit")
    _, checkpoint = project.gate.build_checkpoint()
    assert checkpoint["settled_units"] == 0
    assert checkpoint["pending_units"] == 128
    with project.gate.journal.transaction() as db:
        evidence = db.execute("SELECT root FROM control_evidence").fetchone()
    proof = Path(evidence["root"]) / ("termination.json" if outcome == "failed" else "tombstone.json")
    proof.write_bytes(proof.read_bytes().replace(b'"synthetic":true', b'"synthetic":false'))
    with pytest.raises(ValueError):
        project.gate.build_checkpoint()
    assert project.gate.status()["unresolved_count"] == 0
