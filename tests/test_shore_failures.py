"""Known terminal byte proofs, HTTP recovery, and no refund of unknown work."""
from __future__ import annotations

import base64
import copy
from pathlib import Path

import pytest

pytest.importorskip("shore_control")
from shore_control.crypto import canonical_json

from msabp_opt.simulation.distributed.control_failure import (
    MAX_TERMINAL_RECORD_BYTES, terminal_payload, validate_terminal_record,
)
from msabp_opt.simulation.distributed.control_worker import ExecutionUncertain
from msabp_opt.simulation.distributed.http_api import ApiError

from test_shore_maid import lab  # noqa: F401
from test_shore_project_integration import ProcessLost, project  # noqa: F401


def prepared_failure(fixture):
    gate = fixture.gate()
    gate.accept(fixture.assignment, run_id="run-1", worker_id="maid-a",
                row_sha256=fixture.assignment["row_sha256"], csv_sha256=fixture.config.csv_sha256)
    gate.start("attempt-1")
    result = gate.save_result("attempt-1", fixture.root / "failure-proof", outcome="failed",
                             failure_code="SyntheticFailure", error_message="known returned failure")
    return gate, result


def test_actual_worker_terminal_record_bytes_checked_before_report(lab):  # noqa: F811
    gate, result = prepared_failure(lab)
    payload = terminal_payload(lab.assignment, result, lab.permit["payload"])
    data = base64.b64decode(payload["terminal_record_b64"], validate=True)
    assert validate_terminal_record(data, result["evidence"], lab.permit["payload"])["state"] == "failed"
    path = Path(result["root"]) / "termination.json"
    path.write_bytes(data.replace(b"known returned failure", b"forged returned failure"))
    with pytest.raises(ValueError):
        gate.report("attempt-1")
    assert lab.executions == 0


@pytest.mark.parametrize("mutation", ["bytes", "extra_field", "noncanonical", "oversized", "missing_artifact"])
def test_terminal_record_validator_rejects_unbacked_or_malformed_proof(lab, mutation):  # noqa: F811
    _, result = prepared_failure(lab)
    data = (Path(result["root"]) / "termination.json").read_bytes()
    evidence = copy.deepcopy(result["evidence"])
    if mutation == "bytes":
        data = data.replace(b"known", b"false")
    elif mutation == "extra_field":
        import json

        data = canonical_json({**json.loads(data), "invented": True})
    elif mutation == "noncanonical":
        data += b"\n"
    elif mutation == "oversized":
        data = b"x" * (MAX_TERMINAL_RECORD_BYTES + 1)
    else:
        evidence["artifacts"] = []
    with pytest.raises(ValueError):
        validate_terminal_record(data, evidence, lab.permit["payload"])


def test_starting_cannot_be_relabelled_not_started(lab):  # noqa: F811
    gate = lab.gate()
    gate.accept(lab.assignment, run_id="run-1", worker_id="maid-a",
                row_sha256=lab.assignment["row_sha256"], csv_sha256=lab.config.csv_sha256)
    gate.start("attempt-1")
    with pytest.raises(ValueError, match="uncertain"):
        gate.save_result("attempt-1", lab.root / "false-tombstone", outcome="not_started")
    with pytest.raises(ExecutionUncertain):
        lab.maid().run()
    assert lab.executions == 0


def test_prepared_without_acceptance_explicit_tombstone_resolves_during_drain(project):  # noqa: F811
    project.hello()
    assignment = project.request("request_task")["payload"]
    project.gate.begin_drain("synthetic removal before acceptance")
    worker = project.worker()
    with pytest.raises(ValueError):
        worker.control_worker.accept(assignment, run_id="run-1", worker_id="maid-a",
            row_sha256=assignment["row_sha256"], csv_sha256=worker.config.csv_sha256)
    assert worker.control_worker._record(assignment["attempt_id"])["acceptance"] is None
    worker.control_worker.save_result(assignment["attempt_id"], project.root / "never-started-proof",
                                      outcome="not_started")
    assert worker._run_assignment(assignment) == "not_started"
    assert project.executions == 0
    assert project.gate.status()["unresolved_count"] == 0
    assert project.runtime.state.get_task("run-1", "0")["status"] == "pending"
    _, manifest = project.gate.build_checkpoint()
    assert manifest["settled_units"] == 0 and manifest["pending_units"] == 128
    with pytest.raises(ApiError):
        project.request("request_task")


def test_http_terminal_rejects_changed_proof_before_ack_or_state_transition(project):  # noqa: F811
    project.hello()
    assignment = project.request("request_task")["payload"]
    worker, attempt = project.worker(), assignment["attempt_id"]
    permit = worker.control_worker.accept(assignment, run_id="run-1", worker_id="maid-a",
        row_sha256=assignment["row_sha256"], csv_sha256=worker.config.csv_sha256)
    worker.control_worker.start(attempt)
    result = worker.control_worker.save_result(attempt, project.root / "known-failure-proof", outcome="failed",
        failure_code="SyntheticFailure", error_message="true record")
    worker.control_worker.report(attempt)
    payload = terminal_payload(assignment, result, permit)
    data = base64.b64decode(payload["terminal_record_b64"]).replace(b"true record", b"fake record")
    payload["terminal_record_b64"] = base64.b64encode(data).decode()
    with pytest.raises(ApiError):
        project.request("control_terminal", payload)
    assert project.runtime.state.get_task("run-1", "0")["status"] == "running"
    status = project.gate.get_operation(permit["operation_id"])
    assert status["controller_receipt"] is None
    assert project.gate.status()["unresolved_count"] == 1


@pytest.mark.parametrize("project", [{"count": 1, "settled_outcomes": ["completed", "failed"]}], indirect=True)
def test_terminal_http_ack_loss_restarts_only_delivery_not_execution(project):  # noqa: F811
    project.hello()
    assignment = project.request("request_task")["payload"]
    class LoseTerminalReply:
        lost = False

        def send(self, message):
            response = project.client.send(message)
            if message["type"] == "control_terminal" and not self.lost:
                self.lost = True
                raise ProcessLost("terminal committed before HTTP response reached Maid")
            return response

    def failed(*_args, **_kwargs):
        project.executions += 1
        raise RuntimeError("synthetic known failure")

    with pytest.raises(ProcessLost):
        project.worker(http_client=LoseTerminalReply(), executor=failed)._run_assignment(assignment)
    assert project.executions == 1
    assert project.worker(executor=failed).run() == 2
    assert project.executions == 1
    assert project.gate.status()["unresolved_count"] == 0
    _, manifest = project.gate.build_checkpoint()
    assert manifest["settled_units"] == 1 and manifest["pending_units"] == 127


@pytest.mark.parametrize("outcome", ["completed", "failed", "not_started"])
def test_late_worker_report_reconciles_cached_collector_receipt(project, outcome):  # noqa: F811
    from shore_control.authority import AuthorityError

    project.hello()
    assignment = project.request("request_task")["payload"]
    def failed(*_args, **_kwargs):
        project.executions += 1
        raise RuntimeError("known synthetic failure")
    worker = project.worker(executor=failed if outcome == "failed" else None)
    if outcome == "not_started":
        worker.control_worker.accept(assignment, run_id="run-1", worker_id="maid-a",
            row_sha256=assignment["row_sha256"], csv_sha256=worker.config.csv_sha256)
        worker.control_worker.save_result(assignment["attempt_id"], project.root / "late-tombstone",
                                          outcome="not_started")
    def report_unavailable(_attempt):
        raise AuthorityError("RPC", "synthetic worker report transport unavailable")
    worker.control_worker.report = report_unavailable
    with pytest.raises(ApiError):
        worker._run_assignment(assignment)
    execution_count = project.executions
    assert project.gate.status()["unresolved_count"] == 1
    assert project.worker(executor=failed if outcome == "failed" else None)._run_assignment(assignment) == outcome
    assert project.executions == execution_count
    assert project.gate.status()["unresolved_count"] == 0
