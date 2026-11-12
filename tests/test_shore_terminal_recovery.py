"""Producer terminal budgets and explicit worker restart binding contract."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("shore_control")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.crypto import canonical_json, public_key_text

from msabp_opt.simulation.distributed import control_worker
from msabp_opt.simulation.distributed.control_failure import (
    MAX_TERMINAL_RECORD_BYTES, TERMINAL_RECORD_FIELDS, validate_terminal_record,
)
from msabp_opt.simulation.distributed.control_worker import ShoreWorker, compute_evidence
from msabp_opt.simulation.distributed.maid import Maid
from msabp_opt.simulation.distributed.protocol import make_message

from test_shore_maid import lab  # noqa: F401
from test_shore_project_integration import FakeSession, ProcessLost, project  # noqa: F401


def started_gate(fixture):
    gate = fixture.gate()
    gate.accept(fixture.assignment, run_id="run-1", worker_id="maid-a",
                row_sha256=fixture.assignment["row_sha256"], csv_sha256=fixture.config.csv_sha256)
    gate.start("attempt-1")
    return gate


def save_failure(fixture, gate, message):
    return gate.save_result("attempt-1", fixture.root / "failure-proof", outcome="failed",
                            failure_code="RuntimeError", error_message=message)


def read_failure(result, permit):
    data = (Path(result["root"]) / "termination.json").read_bytes()
    assert len(data) <= MAX_TERMINAL_RECORD_BYTES
    record = validate_terminal_record(data, result["evidence"], permit)
    assert set(record) == TERMINAL_RECORD_FIELDS
    assert result["aliases"] == {"termination.json": "termination.json"}
    assert {path.name for path in Path(result["root"]).iterdir()} == {"termination.json"}
    return data, record


@pytest.mark.parametrize("message", ["x" * 65536, "失败🌊" * 16000, '\\"\n\t\x00' * 16000],
                         ids=["ascii", "multibyte", "json-escaping"])
def test_failure_producer_bounds_full_canonical_bytes_and_preserves_original_commitment(lab, message):  # noqa: F811
    gate = started_gate(lab)
    result = save_failure(lab, gate, message)
    data, record = read_failure(result, lab.permit["payload"])
    raw = message.encode("utf-8")
    marker = (f"\n[truncated; original_utf8_bytes={len(raw)}; "
              f"original_sha256={hashlib.sha256(raw).hexdigest()}]")
    summary = record["error_message"]
    assert summary.endswith(marker)
    prefix = summary[:-len(marker)]
    assert message.startswith(prefix)
    assert len(canonical_json({**record, "error_message": prefix + message[len(prefix)] + marker})) > 65536
    assert canonical_json(record) == data
    assert gate.checked_result("attempt-1") == result


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_full_terminal_record_exact_boundary(lab, delta):  # noqa: F811
    gate = started_gate(lab)
    record = {"synthetic": True, "permit_id": lab.permit["payload"]["permit_id"],
              "execution_id": "maid-" + lab.permit["payload"]["permit_id"], "state": "failed",
              "failure_code": "RuntimeError", "error_message": ""}
    message = "x" * (65536 - len(canonical_json(record)) + delta)
    result = save_failure(lab, gate, message)
    data, persisted = read_failure(result, lab.permit["payload"])
    if delta <= 0:
        assert len(data) == 65536 + delta
        assert persisted["error_message"] == message
    else:
        assert len(data) == 65536
        assert "[truncated; original_utf8_bytes=" in persisted["error_message"]


@pytest.mark.parametrize("mutation", ["persisted-bytes", "extra-artifact", "non-string-message"])
def test_producer_rejects_invalid_terminal_before_irreversible_result(lab, monkeypatch, mutation):  # noqa: F811
    gate = started_gate(lab)
    if mutation == "persisted-bytes":
        original = control_worker._persist_json

        def corrupt_persist(path, value):
            original(path, {**value, "error_message": "persisted bytes do not match commitment"})

        monkeypatch.setattr(control_worker, "_persist_json", corrupt_persist)
    elif mutation == "extra-artifact":
        root = lab.root / "failure-proof"
        root.mkdir()
        (root / "unexpected.log").write_text("outside the strict terminal alias contract", encoding="utf-8")
    with pytest.raises(ValueError):
        save_failure(lab, gate, 7 if mutation == "non-string-message" else "known failure")
    assert gate._record("attempt-1")["state"] == "STARTING"
    assert gate._record("attempt-1")["result"] is None


def test_historical_oversized_result_is_never_silently_rewritten(lab):  # noqa: F811
    gate = started_gate(lab)
    permit = lab.permit["payload"]
    root = lab.root / "historical-failure"
    root.mkdir()
    record = {"synthetic": True, "permit_id": permit["permit_id"],
              "execution_id": "maid-" + permit["permit_id"], "state": "failed",
              "failure_code": "RuntimeError", "error_message": "x" * 65536}
    data = canonical_json(record)
    (root / "termination.json").write_bytes(data)
    evidence, aliases, evidence_hash = compute_evidence(permit, root, outcome="failed",
        execution_id=record["execution_id"], failure_code="RuntimeError",
        termination_record_sha256=hashlib.sha256(data).hexdigest())
    result = {"root": str(root), "manifest_path": None, "evidence": evidence, "aliases": aliases,
              "receipt": {"permit_id": permit["permit_id"], "outcome": "failed", "durable": True,
                          "evidence_sha256": evidence_hash}}
    with gate.transaction() as db:
        db.execute("UPDATE inbox SET state='RESULT',result=? WHERE attempt_id='attempt-1'",
                   (canonical_json(result).decode(),))
    before = gate._record("attempt-1")
    with pytest.raises(ValueError, match="manual reconciliation"):
        gate.checked_result("attempt-1")
    assert gate._record("attempt-1") == before
    assert (root / "termination.json").read_bytes() == data
    with pytest.raises(ValueError, match="cannot rewrite"):
        save_failure(lab, gate, "must not overwrite historical evidence")


@pytest.mark.parametrize("project", [{"count": 1, "settled_outcomes": ["completed", "failed"]}], indirect=True)
@pytest.mark.parametrize("message", ["x" * 65536, "失败🌊" * 16000, '\\"\n\t\x00' * 16000],
                         ids=["ascii", "multibyte", "json-escaping"])
def test_real_maid_long_failure_restart_dual_receipts_checkpoint_and_drain(project, message):  # noqa: F811
    project.hello()
    assignment = project.request("request_task")["payload"]
    worker = project.worker()

    def failed(*_args, **_kwargs):
        project.executions += 1
        raise RuntimeError(message)

    def lost_before_report(_attempt):
        raise ProcessLost("durable RESULT committed; worker report not yet sent")

    worker._run_csv_row = failed
    worker.control_worker.report = lost_before_report
    with pytest.raises(ProcessLost):
        worker._run_assignment(assignment)
    attempt = assignment["attempt_id"]
    result = worker.control_worker.result(attempt)
    read_failure(result, assignment["control_authority"]["permit"]["payload"])
    assert worker.control_worker._record(attempt)["state"] == "RESULT"
    assert project.executions == 1
    assert project.worker(executor=failed).run() == 2
    assert project.executions == 1
    status = project.gate.get_operation(assignment["control_authority"]["descriptor"]["operation_id"])
    assert status["status"] == "RESOLVED"
    assert status["worker_receipt"]["evidence_sha256"] == result["receipt"]["evidence_sha256"]
    assert status["controller_receipt"]["evidence_sha256"] == result["receipt"]["evidence_sha256"]
    assert project.worker().control_worker.pending_assignments() == []
    project.gate.begin_drain("large known failed execution was reconciled after worker restart")
    assert project.gate.finish_drain()["phase"] == "PAUSED_BOUND"
    _, manifest = project.gate.build_checkpoint()
    assert manifest["settled_units"] == 1 and manifest["pending_units"] == 127


def test_not_started_message_has_same_full_record_budget_without_extra_artifacts(lab):  # noqa: F811
    gate = lab.gate()
    gate.accept(lab.assignment, run_id="run-1", worker_id="maid-a",
        row_sha256=lab.assignment["row_sha256"], csv_sha256=lab.config.csv_sha256)
    result = gate.save_result("attempt-1", lab.root / "never-started", outcome="not_started",
        error_message="尚未启动\n" * 20000)
    data = (Path(result["root"]) / "tombstone.json").read_bytes()
    record = validate_terminal_record(data, result["evidence"], lab.permit["payload"])
    assert len(data) <= MAX_TERMINAL_RECORD_BYTES
    assert "[truncated; original_utf8_bytes=" in record["error_message"]
    assert result["aliases"] == {} and result["evidence"]["artifacts"] == []
    assert result["tombstone"] == record
    assert gate.checked_result("attempt-1") == result


@pytest.mark.parametrize("project", [{"count": 1, "settled_outcomes": ["completed", "failed"]}], indirect=True)
def test_rotated_collector_requires_original_worker_binding_for_restarted_outbox(project):  # noqa: F811
    project.hello()
    assignment = project.request("request_task")["payload"]

    def failed(*_args, **_kwargs):
        project.executions += 1
        raise RuntimeError("known returned failure before worker restart")

    worker = project.worker(executor=failed)

    def lost_before_report(_attempt):
        raise ProcessLost("restart after RESULT, before first worker receipt")

    worker.control_worker.report = lost_before_report
    with pytest.raises(ProcessLost):
        worker._run_assignment(assignment)
    original = worker.control_worker
    project.gate.begin_drain("rotate collector while original worker outbox is pending")
    replacement_key = Ed25519PrivateKey.generate()
    project.gate.resume(session_client=project.authority_client("session:session-b", replacement_key),
        session_id="session-b", session_public_key=public_key_text(replacement_key.public_key()), mode="recover")
    with pytest.raises(ValueError, match="original.*binding.*RESOLVED"):
        project.worker(executor=failed)
    restored = ShoreWorker(original.root, original.client, original.authority_public_key,
        binding=original.binding, registration=original.registration, csv_sha256=original.csv_sha256)
    restarted = Maid(worker.config, client=project.client, artifact_client=project.client,
        session=FakeSession(), control_worker=restored, run_csv_row=failed)
    assert restarted.run() == 2
    assert project.executions == 1
    acknowledgments = [request for request in project.transport.requests
                       if request["payload"]["method"] == "controller_ack"]
    assert acknowledgments[-1]["payload"]["actor_id"] == "session:session-b"
    assert restored.pending_assignments() == []
    assert project.worker().control_worker.pending_assignments() == []
    _, manifest = project.gate.build_checkpoint()
    assert manifest["settled_units"] == 1 and manifest["pending_units"] == 127


@pytest.mark.parametrize("state", ["PREPARED", "ACCEPTED", "STARTING", "RESULT"])
def test_worker_startup_rejects_changed_binding_until_old_work_resolved(lab, state):  # noqa: F811
    gate = lab.gate()
    gate.accept(lab.assignment, run_id="run-1", worker_id="maid-a",
                row_sha256=lab.assignment["row_sha256"], csv_sha256=lab.config.csv_sha256)
    if state == "PREPARED":
        with gate.transaction() as db:
            db.execute("UPDATE inbox SET state='PREPARED',acceptance=NULL")
    elif state in {"STARTING", "RESULT"}:
        gate.start("attempt-1")
    if state == "RESULT":
        save_failure(lab, gate, "known failure")
    before = gate._record("attempt-1")
    with pytest.raises(ValueError, match="original.*binding.*RESOLVED"):
        ShoreWorker(gate.root, gate.client, gate.authority_public_key,
            binding={**gate.binding, "session_id": "new-session"},
            registration=gate.registration, csv_sha256=gate.csv_sha256)
    assert gate._record("attempt-1") == before
    assert lab.gate()._record("attempt-1") == before


def test_resolved_old_binding_allows_startup_but_does_not_relax_new_assignment_validation(lab):  # noqa: F811
    old = lab.gate()
    assert lab.maid(old)._run_assignment(lab.assignment) == "completed"
    new = ShoreWorker(old.root, old.client, old.authority_public_key,
        binding={**old.binding, "session_id": "new-session"},
        registration=old.registration, csv_sha256=old.csv_sha256)
    assert new.pending_assignments() == []
    with pytest.raises(ValueError, match="independently expected work"):
        new.accept(lab.assignment, run_id="run-1", worker_id="maid-a",
            row_sha256=lab.assignment["row_sha256"], csv_sha256=lab.config.csv_sha256)
    assert lab.executions == 1


def test_maid_retry_paused_response_exits_without_stop_or_repoll(lab, monkeypatch, capsys):  # noqa: F811
    original = lab.collector.send

    def paused(message):
        if message["type"] == "request_task":
            lab.collector.messages.append("request_task")
            return make_message("retry_paused", run_id=message["run_id"], worker_id=message["worker_id"],
                protocol_version=2, payload={"reason": "retry allowance exhausted",
                    "cases": [{"case_id": "case-1", "attempt_count": 3, "max_attempts": 3}]})
        return original(message)

    monkeypatch.setattr(lab.collector, "send", paused)
    assert lab.maid().run() == 2
    assert lab.executions == 0
    assert lab.collector.messages == ["hello", "request_task"]
    assert "retry allowance exhausted" in capsys.readouterr().out
