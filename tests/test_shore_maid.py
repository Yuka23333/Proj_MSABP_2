"""Real Maid entry with synthetic executor and temporary signed Shore authority."""
from __future__ import annotations

import copy
import base64
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("shore_control")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.authority import Authority, digest
from shore_control.client import Client, MemorySigner
from shore_control.crypto import public_key_text
from shore_control.evidence import verify_artifacts
from shore_control.permits import DESCRIPTOR_FIELDS

from msabp_opt.simulation.distributed.control_worker import ExecutionUncertain, ShoreWorker
from msabp_opt.simulation.distributed.maid import (
    Maid, MaidConfigError, MaidRuntimeConfig, OwnedCstSession, canonical_row_sha256, doctor,
)
from msabp_opt.simulation.distributed.protocol import make_message


class Crash(BaseException):
    """A process loss, intentionally not a simulation failure."""


class FakeSession:
    def open(self):
        return self

    def close(self):
        pass


class DroppingTransport:
    def __init__(self, authority):
        self.authority = authority
        self.drop = None
        self.requests = []

    def challenge(self, actor):
        return self.authority.challenge(actor)

    def handle(self, request):
        self.requests.append(request)
        result = self.authority.handle(request)
        if request["payload"]["method"] == self.drop:
            self.drop = None
            raise TimeoutError("synthetic authority reply lost")
        return result


class Collector:
    def __init__(self, fixture):
        self.fixture = fixture
        self.upload_count = 0
        self.complete_count = 0
        self.ack = True
        self.drop_complete = False
        self.drop_upload = False
        self.messages = []

    def upload_artifact(self, attempt_id, worker_id, lease_token, archive):
        self.upload_count += 1
        target = self.fixture.root / "collector"
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)
        if self.drop_upload:
            self.drop_upload = False
            raise Crash("upload bytes committed before reply")
        return {"sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest()}

    def send(self, message):
        self.messages.append(message["type"])
        assert message["protocol_version"] == 2
        reply_payload = {}
        if message["type"] == "hello":
            kind = "welcome"
        elif message["type"] == "request_task":
            kind = "stop"
        elif message["type"] == "heartbeat":
            kind = "heartbeat_ack"
        elif message["type"] == "complete":
            self.complete_count += 1
            control = message["payload"]["control_authority"]
            checked = verify_artifacts(control["evidence"], self.fixture.permit["payload"],
                                       self.fixture.root / "collector", control["aliases"])
            assert checked == control["evidence_sha256"]
            if self.ack:
                self.fixture.session.call("controller_ack", permit_id=control["permit_id"],
                    outcome=control["outcome"], evidence_sha256=checked, durable=True)
            if self.drop_complete:
                self.drop_complete = False
                raise Crash("collector persisted completion before response")
            kind = "completed_ack"
        elif message["type"] == "control_terminal":
            from msabp_opt.simulation.distributed.control_failure import validate_terminal_record

            payload = message["payload"]
            control = payload["control_authority"]
            data = base64.b64decode(payload["terminal_record_b64"], validate=True)
            validate_terminal_record(data, control["evidence"], self.fixture.permit["payload"])
            root = self.fixture.root / "collector-terminal"
            root.mkdir(exist_ok=True)
            filename = "termination.json" if control["outcome"] == "failed" else "tombstone.json"
            (root / filename).write_bytes(data)
            checked = verify_artifacts(control["evidence"], self.fixture.permit["payload"], root, control["aliases"])
            assert checked == control["evidence_sha256"]
            if self.ack:
                self.fixture.session.call("controller_ack", permit_id=control["permit_id"],
                    outcome=control["outcome"], evidence_sha256=checked, durable=True)
            kind = "terminal_ack"
            reply_payload = {"outcome": control["outcome"]}
        else:
            raise AssertionError(message["type"])
        return make_message(kind, run_id=message["run_id"], worker_id=message["worker_id"],
                            protocol_version=2, payload=reply_payload)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.delenv("MSABP_CONTROL_MODE", raising=False)
    keys = {name: Ed25519PrivateKey.generate() for name in ("authority", "controller", "maid-a", "session")}
    now = [1000.0]
    registration = {"public_key": public_key_text(keys["maid-a"].public_key()),
                    "device_id": "device-a", "tailnet_node_id": "node-a",
                    "role": "maid", "allowed_actions": ["attempt"]}
    authority = Authority(tmp_path / "authority.sqlite", keys["authority"], fleet_id="test-maid-fleet",
        controllers={"controller": public_key_text(keys["controller"].public_key())},
        workers={"maid-a": registration}, clock=lambda: now[0])
    transport = DroppingTransport(authority)
    def client(actor, key):
        return Client(transport, actor, MemorySigner(keys[key]), keys["authority"].public_key(), "test-maid-fleet")
    controller = client("controller", "controller")
    session = client("session:session", "session")
    worker_client = client("maid-a", "maid-a")
    job_spec = {"schema": "shore.job-spec", "schema_version": 1,
                "manifest_sha256": digest({"test": "maid"}), "approved_units": 2,
                "budget_unit": "doe_sample", "accounting_policy_sha256": digest({"settle": "success"})}
    job = controller.call("acquire", job_id="job-1", host_id="host-1", session_id="session",
        session_public_key=public_key_text(keys["session"].public_key()), project_id="msabp",
        collector_url="http://100.64.0.1:8765", data_root=str(tmp_path), job_spec=job_spec)
    bound = {key: job[key] for key in ("job_id", "epoch", "host_id", "session_id")}
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("sample_id,value\ncase-1,3\n", encoding="utf-8")
    csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    controller.call("register_run", **bound, run_id="run-1", original_plan_sha256=digest({"plan": 1}),
                    input_manifest_sha256=csv_hash)
    row_hash = canonical_row_sha256(("sample_id", "value"), {"sample_id": "case-1", "value": "3"})
    permit = controller.call("validate_dispatch", **bound, operation={
        "operation_id": "operation-1", "run_id": "run-1", "worker_id": "maid-a",
        "attempt_id": "attempt-1", "action": "attempt", "input_sha256": row_hash,
        "action_context": None})
    descriptor = {key: permit["payload"][key] for key in DESCRIPTOR_FIELDS}
    binding = {**bound, **job["scope"], "fleet_id": "test-maid-fleet"}
    config = MaidRuntimeConfig(run_id="run-1", worker_id="maid-a", princess_url="http://127.0.0.1:8765",
        api_token="synthetic", csv_path=csv_path, csv_sha256=csv_hash,
        project_path=tmp_path / "NEVER-OPEN-CST.cst", output_root=tmp_path / "output",
        dry_run=False, protocol_mode="shore-fake-v1", heartbeat_seconds=60, poll_seconds=0.001)
    fixture = SimpleNamespace(root=tmp_path, now=now, authority=authority, transport=transport,
        keys=keys, controller=controller, session=session, worker_client=worker_client,
        permit=permit, config=config, executions=0, bound=bound,
        assignment={"case_id": "case-1", "row_index": 0, "attempt_id": "attempt-1",
                    "lease_token": "lease-1", "row_sha256": row_hash,
                    "control_authority": {"descriptor": descriptor, "permit": permit}})
    def gate():
        return ShoreWorker(tmp_path / "worker-state", worker_client, keys["authority"].public_key(),
            binding=binding, registration=registration, csv_sha256=csv_hash)
    fixture.gate = gate
    def executor(_row, **kwargs):
        fixture.executions += 1
        directory = kwargs["output_root"] / "case-1"
        directory.mkdir(parents=True)
        manifest = directory / "manifest.json"
        manifest.write_text('{"status":"completed","synthetic":true}', encoding="utf-8")
        (directory / "Farfield Source [1].ffs").write_bytes(b"synthetic farfield")
        return SimpleNamespace(case_directory=directory, manifest_path=manifest)
    fixture.executor = executor
    fixture.collector = Collector(fixture)
    fixture.maid = lambda gate_instance=None, executor_fn=None: Maid(config,
        client=fixture.collector, session=FakeSession(), control_worker=gate_instance or gate(),
        run_csv_row=executor_fn or executor)
    return fixture


def test_real_entry_signed_gate_ffs_alias_and_cleanup(lab):
    worker = lab.maid()
    assert worker._run_assignment(lab.assignment) == "completed"
    assert lab.executions == 1
    assert not (lab.config.output_root / "attempts" / "attempt-1").exists()
    result = worker.control_worker.result("attempt-1")
    assert result["aliases"]["farfield.ffs"] == "Farfield Source [1].ffs"
    assert worker._run_assignment(lab.assignment) == "completed"
    assert lab.executions == 1


@pytest.mark.parametrize("change", ["missing", "signature", "worker", "device", "run", "input", "csv"])
def test_invalid_permission_never_invokes_executor(lab, change):
    assignment = copy.deepcopy(lab.assignment)
    worker = lab.maid()
    if change == "missing":
        assignment.pop("control_authority")
    elif change == "signature":
        assignment["control_authority"]["permit"]["signature"] = "bad"
    elif change == "csv":
        worker.config = replace(worker.config, csv_sha256="a" * 64)
    else:
        field = {"worker": "worker_id", "device": "device_id", "run": "run_id", "input": "input_sha256"}[change]
        assignment["control_authority"]["descriptor"][field] = "different"
    with pytest.raises(ValueError):
        worker._run_assignment(assignment)
    assert lab.executions == 0

def test_explicit_mode_refuses_real_executor_and_missing_gate(lab):
    with pytest.raises(MaidConfigError, match="fake executor/session"):
        Maid(lab.config, client=lab.collector)
    with pytest.raises(MaidConfigError, match="fake executor/session"):
        Maid(lab.config, client=lab.collector, control_worker=lab.gate(), session=FakeSession())
    legacy = replace(lab.config, protocol_mode="legacy")
    worker = Maid(legacy, client=lab.collector, session=FakeSession(), run_csv_row=lab.executor)
    with pytest.raises(MaidConfigError, match="Legacy Maid"):
        worker._run_assignment(lab.assignment)
    assert lab.executions == 0


def test_accept_reply_loss_replays_exact_signed_request(lab):
    lab.transport.drop = "accept_permit"
    with pytest.raises(TimeoutError):
        lab.maid()._run_assignment(lab.assignment)
    assert lab.executions == 0
    assert lab.maid()._run_assignment(lab.assignment) == "completed"
    requests = [request for request in lab.transport.requests if request["payload"]["method"] == "accept_permit"]
    assert len(requests) == 2 and requests[0] == requests[1]
    assert lab.executions == 1


def test_start_intent_crash_never_reexecutes(lab):
    gate = lab.gate()
    worker = lab.maid(gate)
    original_start = gate.start
    def crash_after_intent(attempt):
        original_start(attempt)
        raise Crash("after durable start intent, before executor")
    gate.start = crash_after_intent
    with pytest.raises(Crash):
        worker._run_assignment(lab.assignment)
    with pytest.raises(ExecutionUncertain):
        lab.maid()._run_assignment(lab.assignment)
    assert lab.executions == 0


def test_executor_return_without_result_commit_never_reexecutes(lab):
    gate = lab.gate()
    def crash_commit(*_args, **_kwargs):
        raise Crash("executor returned before durable result commit")
    gate.save_result = crash_commit
    with pytest.raises(Crash):
        lab.maid(gate)._run_assignment(lab.assignment)
    with pytest.raises(ExecutionUncertain):
        lab.maid()._run_assignment(lab.assignment)
    assert lab.executions == 1


@pytest.mark.parametrize("window", ["upload", "completion"])
def test_upload_and_completion_reply_loss_resume_without_execution(lab, window):
    lab.collector.drop_upload = window == "upload"
    lab.collector.drop_complete = window == "completion"
    with pytest.raises(Crash):
        lab.maid()._run_assignment(lab.assignment)
    assert lab.executions == 1
    assert lab.maid()._run_assignment(lab.assignment) == "completed"
    assert lab.executions == 1
    assert lab.collector.upload_count == (2 if window == "upload" else 1)


def test_old_completed_ack_does_not_delete_pending_bytes(lab):
    lab.collector.ack = False
    assert lab.maid()._run_assignment(lab.assignment) == "completed_pending"
    assert (lab.config.output_root / "outbox" / "attempt-1.zip").exists()
    assert (lab.config.output_root / "attempts" / "attempt-1").exists()
    lab.collector.ack = True
    assert lab.maid()._run_assignment(lab.assignment) == "completed"
    assert lab.executions == 1


def test_accepted_work_can_finish_after_drain_and_lease_expiry(lab):
    gate = lab.gate()
    gate.accept(lab.assignment, run_id="run-1", worker_id="maid-a",
                row_sha256=lab.assignment["row_sha256"], csv_sha256=lab.config.csv_sha256)
    lab.session.call("begin_drain", **lab.bound, reason="synthetic key removal")
    lab.now[0] += 300
    assert lab.maid()._run_assignment(lab.assignment) == "completed"
    assert lab.executions == 1


def test_fake_executor_failure_is_durable_and_never_reports_success(lab):
    def failed(*_args, **_kwargs):
        lab.executions += 1
        raise RuntimeError("synthetic failure")
    assert lab.maid(executor_fn=failed)._run_assignment(lab.assignment) == "failed"
    assert lab.maid(executor_fn=failed)._run_assignment(lab.assignment) == "failed"
    assert lab.executions == 1
    assert lab.collector.complete_count == lab.collector.upload_count == 0
    result = lab.gate().result("attempt-1")
    assert result["receipt"]["outcome"] == "failed"
    assert result["evidence"]["result"]["termination_record_sha256"]


def test_changed_bytes_rejected_before_reupload(lab):
    lab.collector.ack = False
    assert lab.maid()._run_assignment(lab.assignment) == "completed_pending"
    result = lab.gate().result("attempt-1")
    (Path(result["root"]) / "Farfield Source [1].ffs").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        lab.maid()._run_assignment(lab.assignment)
    assert lab.executions == 1 and lab.collector.upload_count == 1


def test_worker_report_reply_loss_replays_same_request(lab):
    lab.transport.drop = "worker_report"
    assert lab.maid()._run_assignment(lab.assignment) == "completed"
    requests = [request for request in lab.transport.requests if request["payload"]["method"] == "worker_report"]
    assert len(requests) == 2 and requests[0] == requests[1]
    assert lab.executions == 1


def test_drain_before_first_accept_denies_execution(lab):
    lab.session.call("begin_drain", **lab.bound, reason="before worker accepts")
    with pytest.raises(ValueError):
        lab.maid()._run_assignment(lab.assignment)
    assert lab.executions == 0
    record = lab.gate()._record("attempt-1")
    assert record["state"] == "PREPARED" and record["acceptance"] is None


def test_not_started_tombstone_permanently_prevents_execution(lab):
    gate = lab.gate()
    gate.accept(lab.assignment, run_id="run-1", worker_id="maid-a",
                row_sha256=lab.assignment["row_sha256"], csv_sha256=lab.config.csv_sha256)
    result = gate.save_result("attempt-1", lab.root / "never-started", outcome="not_started")
    assert result["evidence"]["artifacts"] == []
    assert result["evidence"]["result"]["execution_id"] is None
    assert result["evidence"]["result"]["tombstone_sha256"] == digest(result["tombstone"])
    with pytest.raises(ExecutionUncertain):
        gate.start("attempt-1")
    assert lab.maid()._run_assignment(lab.assignment) == "not_started"
    assert lab.executions == 0


def test_late_worker_result_uses_recovered_collector_session(lab):
    gate = lab.gate()
    gate.accept(lab.assignment, run_id="run-1", worker_id="maid-a",
                row_sha256=lab.assignment["row_sha256"], csv_sha256=lab.config.csv_sha256)
    lab.session.call("begin_drain", **lab.bound, reason="recover collector")
    recovered_key = Ed25519PrivateKey.generate()
    lab.controller.call("resume_bound_job", job_id=lab.bound["job_id"], epoch=lab.bound["epoch"],
        host_id=lab.bound["host_id"], previous_session_id="session", session_id="recovered",
        session_public_key=public_key_text(recovered_key.public_key()), mode="recover")
    lab.session = Client(lab.transport, "session:recovered", MemorySigner(recovered_key),
                         lab.keys["authority"].public_key(), "test-maid-fleet")
    assert lab.maid()._run_assignment(lab.assignment) == "completed"
    assert lab.executions == 1


def test_control_environment_rejects_legacy_mode(lab, monkeypatch):
    monkeypatch.setenv("MSABP_CONTROL_MODE", "shore-fake-v1")
    with pytest.raises(MaidConfigError, match="legacy"):
        Maid(replace(lab.config, protocol_mode="legacy"), client=lab.collector,
             session=FakeSession(), run_csv_row=lab.executor)
    assert lab.executions == 0


def test_unavailable_worker_rpc_still_uploads_accepted_work_and_keeps_outbox(lab):
    from shore_control.authority import AuthorityError

    gate = lab.gate()
    def unavailable(_attempt):
        raise AuthorityError("RPC", "worker transport unavailable")
    gate.report = unavailable
    assert lab.maid(gate)._run_assignment(lab.assignment) == "completed_pending"
    assert lab.collector.upload_count == 1
    assert (lab.config.output_root / "outbox" / "attempt-1.zip").exists()
    assert lab.maid()._run_assignment(lab.assignment) == "completed"
    assert lab.executions == 1


def test_control_mode_native_session_and_doctor_refuse_before_cst_import(lab, monkeypatch):
    import builtins

    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "cst" or name.startswith("cst."):
            raise AssertionError("native CST import attempted")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setenv("MSABP_CONTROL_MODE", "shore-fake-v1")
    with pytest.raises(RuntimeError, match="CST session open"):
        OwnedCstSession(lab.config.project_path).open()
    with pytest.raises(RuntimeError, match="CST doctor import"):
        doctor(lab.config)
    report = doctor(replace(lab.config, dry_run=True))
    assert "cst.interface" not in report["modules"]


def test_run_startup_replays_outbox_before_collector_stop(lab):
    assignment = {**lab.assignment, "lease_expires_at": 1001.25}
    lab.collector.drop_complete = True
    with pytest.raises(Crash):
        lab.maid()._run_assignment(assignment)
    lab.collector.messages.clear()
    assert lab.maid().run() == 0
    assert lab.collector.messages == ["hello", "complete", "request_task"]
    assert lab.executions == 1 and lab.collector.upload_count == 1
    assert lab.gate().pending_assignments() == []


def test_run_startup_uncertain_execution_refuses_new_poll(lab):
    gate = lab.gate()
    gate.accept(lab.assignment, run_id="run-1", worker_id="maid-a",
                row_sha256=lab.assignment["row_sha256"], csv_sha256=lab.config.csv_sha256)
    gate.start("attempt-1")
    with pytest.raises(ExecutionUncertain):
        lab.maid().run()
    assert lab.collector.messages == ["hello"]
    assert lab.executions == 0
