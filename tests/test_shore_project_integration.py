"""Actual runtime/HTTP/coordinator/Maid paths with only synthetic execution.

The Shore authority and all private test keys are generated in isolated pytest
temporary directories/in memory. This never uses the deployed lab fleet.
"""
from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("shore_control", reason="explicit optional Shore checkout required")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.authority import Authority, AuthorityError
from shore_control.client import Client, MemorySigner
from shore_control.crypto import canonical_json, public_key_text

from msabp_opt.simulation.distributed import case_runner
from msabp_opt.simulation.distributed.config import DeviceConfig, DeviceRegistry, LaunchMode
from msabp_opt.simulation.distributed.control_authority import ControlAuthority, ControlError
from msabp_opt.simulation.distributed.control_worker import ShoreWorker
from msabp_opt.simulation.distributed.http_api import ARTIFACT_PATH_PREFIX, ApiError, PrincessClient
from msabp_opt.simulation.distributed.maid import Maid, MaidRuntimeConfig
from msabp_opt.simulation.distributed.protocol import make_message
from msabp_opt.simulation.distributed.runtime import PrincessRuntime, prepare_run
from msabp_opt.simulation.distributed.transport import launch_maid


class ProcessLost(BaseException):
    """A crash outside the normal returned simulation-failure path."""


class RemovableSigner(MemorySigner):
    present = True

    def sign(self, domain, payload):
        if not self.present:
            raise FileNotFoundError("synthetic controller signer removed")
        return super().sign(domain, payload)


class AuthorityTransport:
    def __init__(self, authority):
        self.authority = authority
        self.drop_method = None
        self.requests = []

    def challenge(self, actor):
        return self.authority.challenge(actor)

    def handle(self, request):
        self.requests.append(request)
        response = self.authority.handle(request)
        if request["payload"]["method"] == self.drop_method:
            self.drop_method = None
            raise TimeoutError("synthetic authority response lost after commit")
        return response


class FakeSession:
    def open(self):
        return self

    def close(self):
        pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def project(tmp_path, monkeypatch, request):
    monkeypatch.setenv("MSABP_CONTROL_MODE", "shore-fake-v1")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cst" or name.startswith("cst."):
            raise AssertionError("Project integration must never import CST")
        return original_import(name, *args, **kwargs)

    def forbidden_process(*args, **kwargs):
        raise AssertionError("Project integration must never start SSH/GPU/local processes")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "Popen", forbidden_process)
    monkeypatch.setattr(subprocess, "run", forbidden_process)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    device = DeviceConfig(id="maid-a", enabled=True, launch_mode=LaunchMode.LOCAL,
                          ssh_target=None, repo_root=str(tmp_path), python_path=sys.executable)
    registry = DeviceRegistry(bind_host="127.0.0.1", advertise_url=url, port=port, devices=(device,))
    source = tmp_path / "samples.csv"
    options = getattr(request, "param", 2)
    count = options.get("count", 2) if isinstance(options, dict) else options
    settled_outcomes = options.get("settled_outcomes", ["completed"]) if isinstance(options, dict) else ["completed"]
    source.write_text("sample_id,value\n" + "".join(f"{i},{i + 1}\n" for i in range(count)), encoding="utf-8")
    preparation = prepare_run(source_csv=source, run_id="run-1", registry=registry,
                              devices=(device,), repository_root=tmp_path)
    frozen_before = sha(preparation.paths.worklist_csv)
    keys = {name: Ed25519PrivateKey.generate() for name in (
        "authority", "controller", "controller-b", "session-a", "maid-a", "maid-b")}
    registrations = {name: {
        "public_key": public_key_text(keys[name].public_key()), "device_id": "device-" + name,
        "tailnet_node_id": "node-" + name, "role": "maid", "allowed_actions": ["attempt"],
    } for name in ("maid-a", "maid-b")}
    authority = Authority(tmp_path / "authority.sqlite", keys["authority"], fleet_id="project-test-fleet",
        controllers={name: public_key_text(keys[name].public_key()) for name in ("controller", "controller-b")},
        workers=registrations, lease_seconds=300)
    transport = AuthorityTransport(authority)
    signer = RemovableSigner(keys["controller"])

    def authority_client(actor, key=None, actor_signer=None):
        return Client(transport, actor, actor_signer or MemorySigner(key or keys[actor]),
                      keys["authority"].public_key(), "project-test-fleet")

    controller = authority_client("controller", actor_signer=signer)
    session = authority_client("session:session-a", keys["session-a"])
    workers = {name: authority_client(name) for name in registrations}
    gate = ControlAuthority(tmp_path / "control.sqlite", controller_client=controller,
        session_client=session, worker_clients=workers, worker_registrations=registrations)
    manifest, policy = tmp_path / "complete-job.json", tmp_path / "accounting.json"
    manifest.write_bytes(canonical_json({"job": "synthetic-full-job", "approved_units": 128}))
    policy.write_bytes(canonical_json({"schema": "msabp.budget-policy", "schema_version": 1,
        "unit_key": "evaluation_id", "settled_outcomes": settled_outcomes}))
    spec = {"schema": "shore.job-spec", "schema_version": 1, "manifest_sha256": sha(manifest),
        "approved_units": 128, "budget_unit": "objective_evaluation",
        "accounting_policy_sha256": sha(policy)}
    gate.acquire(job_id="synthetic-full-job", host_id="synthetic-host", session_id="session-a",
        session_public_key=public_key_text(keys["session-a"].public_key()), project_id="msabp-test",
        # Shore pins a Tailnet-shaped synthetic endpoint. Only the project's
        # injected HTTP test transport below routes it to loopback; no Tailnet
        # address is connected and no deployed actor/authority is accessed.
        collector_url=f"http://100.64.0.254:{port}",
        data_root=str(preparation.paths.results_root), job_spec=spec)
    gate.bind_job_documents(manifest, policy)
    gate.register_run("run-1", sha(preparation.paths.runtime_json), frozen_before)
    gate.bind_run_plan("run-1", preparation.paths.runtime_json)
    runtime = PrincessRuntime(preparation=preparation, registry=registry, devices=(device,),
        project_template=tmp_path / "NEVER-OPEN.cst", control_authority=gate,
        heartbeat_seconds=1, lease_seconds=5, poll_seconds=0.001,
        push_file=forbidden_process, doctor=forbidden_process, launcher=forbidden_process)
    runtime.start_server()
    client = PrincessClient(url, preparation.api_token, timeout=3)
    fixture = SimpleNamespace(root=tmp_path, gate=gate, runtime=runtime, client=client,
        keys=keys, signer=signer, transport=transport, registrations=registrations,
        authority=authority, preparation=preparation, executions=0, uploads=0,
        authority_client=authority_client, workers=workers, device=device)

    def execute(_row, **kwargs):
        fixture.executions += 1
        case_id = str(kwargs["case_id"])
        directory = Path(kwargs["output_root"]) / ("case_" + case_id.zfill(4))
        directory.mkdir(parents=True)
        contents = {
            "s11": (case_runner.S11_FILENAME, b"1.0 -12.0\n2.0 -8.0\n"),
            "rad_eff": (case_runner.RAD_EFF_FILENAME, b"1.0 -2.0\n2.0 -1.5\n"),
            "tot_eff": (case_runner.TOT_EFF_FILENAME, b"1.0 -3.0\n2.0 -2.5\n"),
            "farfield_source": (case_runner.FARFIELD_SOURCE_FILENAME, b"synthetic-farfield\n"),
        }
        artifacts = {}
        for kind, (filename, data) in contents.items():
            (directory / filename).write_bytes(data)
            artifacts[kind] = {"path": filename, "size_bytes": len(data),
                               "sha256": hashlib.sha256(data).hexdigest()}
        result_manifest = {"schema_version": case_runner.MANIFEST_SCHEMA_VERSION,
            "case_id": case_id, "status": "completed", "dry_run": False,
            "parameters": {}, "geometry": {}, "artifacts": artifacts}
        manifest_path = directory / case_runner.MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(result_manifest, sort_keys=True), encoding="utf-8")
        return SimpleNamespace(case_directory=directory, manifest_path=manifest_path)

    def worker(worker_id="maid-a", http_client=None, executor=None):
        config = MaidRuntimeConfig(run_id="run-1", worker_id=worker_id, princess_url=url,
            api_token=preparation.api_token, csv_path=preparation.paths.worklist_csv,
            csv_sha256=frozen_before, project_path=tmp_path / "NEVER-OPEN.cst",
            output_root=tmp_path / (worker_id + "-output"), dry_run=False,
            protocol_mode="shore-fake-v1", heartbeat_seconds=1, poll_seconds=0.001,
            artifact_commit_deadline_seconds=3)
        worker_gate = ShoreWorker(tmp_path / (worker_id + "-journal"), workers[worker_id],
            keys["authority"].public_key(),
            binding={**gate.binding, **gate.job["scope"], "fleet_id": "project-test-fleet"},
            registration=registrations[worker_id], csv_sha256=frozen_before)
        return Maid(config, client=http_client or client, artifact_client=http_client or client,
                    session=FakeSession(), control_worker=worker_gate, run_csv_row=executor or execute)

    def request(kind, payload=None, worker_id="maid-a", version=2):
        return client.send(make_message(kind, run_id="run-1", worker_id=worker_id,
            payload=payload or {}, protocol_version=version))

    fixture.worker, fixture.request = worker, request
    fixture.hello = lambda worker_id="maid-a": request("hello", {"csv_sha256": frozen_before}, worker_id)
    try:
        yield fixture
    finally:
        runtime.close()
        assert sha(preparation.paths.worklist_csv) == frozen_before
        assert not (tmp_path / "NEVER-OPEN.cst").exists()


def test_runtime_http_real_maid_success_and_full_budget_remains(project):
    assert project.client.health() == {"status": "ok"}
    assert project.worker().run() == 0
    assert project.executions == 2
    assert project.runtime.state.progress("run-1").completed == 2
    assert project.gate.status()["unresolved_count"] == 0
    checkpoint, manifest = project.gate.build_checkpoint()
    assert checkpoint["pending_count"] == manifest["pending_units"] == 126
    assert manifest["settled_units"] == 2
    result = project.preparation.paths.results_root / "case_0000"
    assert (result / "Farfield Source [1].ffs").read_bytes() == b"synthetic-farfield\n"
    assert not (result / "farfield.ffs").exists()


def test_both_coordinator_claim_paths_and_legacy_v1_rejection(project, monkeypatch):
    with pytest.raises(ApiError, match="v2/legacy"):
        project.request("hello", {"csv_sha256": project.preparation.worklist.worklist.sha256}, version=1)
    project.hello()
    seen = []
    original = project.runtime.state.claim_next

    def observed(*args, **kwargs):
        seen.append(kwargs["exclude_case_ids"])
        return original(*args, **kwargs)

    monkeypatch.setattr(project.runtime.state, "claim_next", observed)
    assignment = project.request("request_task", {"exclude_case_ids": ["0", "1"]})
    assert seen == [("0", "1"), ()]
    assert assignment["payload"]["control_authority"]["permit"]
    assert project.executions == 0
    project.hello("maid-b")
    project.signer.present = False
    with pytest.raises(ApiError, match="signer removed"):
        project.request("request_task", worker_id="maid-b")
    assert project.runtime.state.progress("run-1").running == 1


def test_drain_during_real_http_part_upload_retains_attempt_and_finishes(project):
    project.hello()
    assignment = project.request("request_task")["payload"]
    arrived, proceed = threading.Event(), threading.Event()
    original = project.runtime._server.artifact_handler

    def staged(*args, **kwargs):
        arrived.set()
        assert proceed.wait(10), "test failed to release synthetic upload"
        return original(*args, **kwargs)

    project.runtime._server.artifact_handler = staged
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(project.worker()._run_assignment, assignment)
        try:
            assert arrived.wait(5)
            assert list(project.preparation.paths.upload_root.glob("*.part"))
            project.signer.present = False
            project.gate.begin_drain("synthetic removable signer removed mid-upload")
            with pytest.raises(ControlError, match="upload"):
                project.gate.build_checkpoint()
            assert project.runtime.state.release_expired_leases("run-1", now=10**12) == ()
            assert project.runtime.state.get_task("run-1", "0")["attempt_id"] == assignment["attempt_id"]
            project.hello("maid-b")
            with pytest.raises(ApiError, match="draining"):
                project.request("request_task", worker_id="maid-b")
        finally:
            proceed.set()
        assert task.result(timeout=10) == "completed"
    assert project.executions == 1
    assert project.gate.status()["unresolved_count"] == 0
    assert project.runtime.state.progress("run-1").pending == 1


def test_dispatch_rpc_reply_loss_through_http_recovers_one_intent(project):
    project.hello()
    project.transport.drop_method = "validate_dispatch"
    with pytest.raises(ApiError, match="response lost"):
        project.request("request_task")
    requests = [r for r in project.transport.requests if r["payload"]["method"] == "validate_dispatch"]
    assert len(requests) == 1
    project.runtime.close()
    project.runtime.start_server()
    assignment = project.request("request_task")["payload"]
    repeated = [r for r in project.transport.requests if r["payload"]["method"] == "validate_dispatch"]
    assert repeated == [requests[0], requests[0]]
    assert assignment["attempt_id"] == requests[0]["payload"]["params"]["operation"]["attempt_id"]
    assert project.worker()._run_assignment(assignment) == "completed"
    assert project.executions == 1


@pytest.mark.parametrize("window", ["local_commit_before_ack", "completed_reply_lost"])
@pytest.mark.parametrize("project", [1], indirect=True)
def test_collector_crash_restarts_without_second_execution(project, monkeypatch, window):
    original_ack = project.gate.ack_evidence
    if window == "local_commit_before_ack":
        def fail_ack(*args, **kwargs):
            raise RuntimeError("synthetic crash before controller acknowledgment")
        monkeypatch.setattr(project.gate, "ack_evidence", fail_ack)

    class CrashAfterComplete:
        def send(self, message):
            if message["type"] != "complete":
                return project.client.send(message)
            try:
                project.client.send(message)
            except ApiError:
                if window != "local_commit_before_ack":
                    raise
            raise ProcessLost(window)

        def upload_artifact(self, *args):
            return project.client.upload_artifact(*args)

    with pytest.raises(ProcessLost):
        project.worker(http_client=CrashAfterComplete()).run()
    assert project.executions == 1
    assert project.runtime.state.get_task("run-1", "0")["status"] == "completed"
    with sqlite3.connect(project.preparation.paths.database) as db:
        assert db.execute("SELECT COUNT(*) FROM control_collection").fetchone()[0] == 1
    monkeypatch.setattr(project.gate, "ack_evidence", original_ack)
    project.runtime.close()
    project.runtime.start_server()
    assert project.worker().run() == 0
    assert project.executions == 1
    assert project.gate.status()["unresolved_count"] == 0


def test_old_session_accepted_worker_late_upload_confirmed_by_new_session(project):
    project.hello()
    assignment = project.request("request_task")["payload"]
    worker = project.worker()
    worker.control_worker.accept(assignment, run_id="run-1", worker_id="maid-a",
        row_sha256=assignment["row_sha256"], csv_sha256=worker.config.csv_sha256)
    project.gate.begin_drain("synthetic controller recovery")
    replacement_key = Ed25519PrivateKey.generate()
    replacement = project.authority_client("session:session-b", replacement_key)
    project.gate.resume(session_client=replacement, session_id="session-b",
        session_public_key=public_key_text(replacement_key.public_key()), mode="recover")
    assert assignment["control_authority"]["descriptor"]["session_id"] == "session-a"
    assert project.gate.binding["session_id"] == "session-b"
    project.runtime.close()
    project.runtime.start_server()
    assert worker._run_assignment(assignment) == "completed"
    acknowledgments = [r for r in project.transport.requests
                       if r["payload"]["method"] == "controller_ack"]
    assert acknowledgments[-1]["payload"]["actor_id"] == "session:session-b"
    assert project.executions == 1


def test_drain_wins_before_acceptance_executor_never_starts(project):
    project.hello()
    assignment = project.request("request_task")["payload"]
    worker = project.worker()
    project.gate.begin_drain("synthetic drain precedes worker admission")
    with pytest.raises(Exception, match="[Dd]rain|ACTIVE|active"):
        worker._run_assignment(assignment)
    assert project.executions == 0
    assert project.gate.status()["unresolved_count"] == 1
    assert project.runtime.state.release_expired_leases("run-1", now=10**12) == ()


@pytest.mark.parametrize("mode", list(LaunchMode))
def test_runtime_launch_and_all_transport_modes_refuse_external_execution(project, mode):
    device = replace(project.device, launch_mode=mode, ssh_target="synthetic-only",
                     bell_host="synthetic-only", scheduled_task_name="synthetic-only")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("No process or device wake may be reached")

    with pytest.raises(RuntimeError, match="Shore fake-worker lab refuses"):
        launch_maid(device, runner=forbidden, popen_factory=forbidden, bell_client_factory=forbidden)
    with pytest.raises(RuntimeError, match="controlled|Shore|fake"):
        project.runtime.deploy_device(device)
    project.runtime._schedule_restarts(10**12)
    project.runtime._schedule_recoveries(10**12, project.runtime.state.progress("run-1").as_dict())
    assert calls == []
    assert project.runtime._restart_due == project.runtime._recovery_due == {}


def test_controlled_runtime_cannot_reopen_as_legacy_when_mode_env_removed(project, monkeypatch):
    project.runtime.close()
    monkeypatch.delenv("MSABP_CONTROL_MODE")
    with pytest.raises(RuntimeError, match="[Cc]ontrol|Shore"):
        PrincessRuntime(preparation=project.preparation, registry=project.runtime.registry,
            devices=(project.device,), project_template=project.root / "NEVER-OPEN.cst")


def test_dry_run_manifest_cannot_settle_a_completed_shore_evaluation(project):
    def dry_result(_row, **kwargs):
        project.executions += 1
        directory = Path(kwargs["output_root"]) / "case_0000"
        directory.mkdir(parents=True)
        manifest = directory / case_runner.MANIFEST_FILENAME
        manifest.write_text(json.dumps({"schema_version": case_runner.MANIFEST_SCHEMA_VERSION,
            "case_id": kwargs["case_id"], "status": "dry_run", "dry_run": True,
            "parameters": {}, "geometry": {}, "artifacts": {}}), encoding="utf-8")
        return SimpleNamespace(case_directory=directory, manifest_path=manifest)

    project.hello()
    assignment = project.request("request_task")["payload"]
    with pytest.raises(ApiError, match="completed|dry.run|dry-run"):
        project.worker(executor=dry_result)._run_assignment(assignment)
    assert project.runtime.state.progress("run-1").completed == 0
    assert project.gate.status()["unresolved_count"] == 1
    with project.gate.journal.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM control_evaluations").fetchone()[0] == 0


@pytest.mark.parametrize("project", [
    {"count": 1, "settled_outcomes": ["completed"]},
    {"count": 1, "settled_outcomes": ["completed", "failed"]},
], indirect=True)
def test_known_executor_failure_dual_receipts_and_explicit_budget_policy(project):
    def failed(_row, **_kwargs):
        project.executions += 1
        raise RuntimeError("synthetic returned execution failure")

    project.hello()
    assignment = project.request("request_task")["payload"]
    worker = project.worker(executor=failed)
    assert worker._run_assignment(assignment) == "failed"
    assert project.runtime.state.get_task("run-1", "0")["status"] == "failed"
    assert project.gate.status()["unresolved_count"] == 0
    assert project.executions == 1
    assert worker._run_assignment(assignment) == "failed"
    assert project.executions == 1
    _, manifest = project.gate.build_checkpoint()
    settled = int("failed" in project.gate._policy()["settled_outcomes"])
    assert manifest["settled_units"] == settled
    assert manifest["pending_units"] == 128 - settled


def test_known_not_started_tombstone_resolves_without_execution_or_budget_unit(project):
    project.hello()
    assignment = project.request("request_task")["payload"]
    worker = project.worker()
    worker.control_worker.accept(assignment, run_id="run-1", worker_id="maid-a",
        row_sha256=assignment["row_sha256"], csv_sha256=worker.config.csv_sha256)
    worker.control_worker.save_result(assignment["attempt_id"], project.root / "not-started",
                                      outcome="not_started")
    project.gate.begin_drain("synthetic known cancellation before executor start")
    assert worker._run_assignment(assignment) == "not_started"
    assert project.executions == 0
    assert project.runtime.state.get_task("run-1", "0")["status"] == "pending"
    assert project.gate.status()["unresolved_count"] == 0
    _, manifest = project.gate.build_checkpoint()
    assert manifest["settled_units"] == 0 and manifest["pending_units"] == 128


@pytest.mark.parametrize("project", [32], indirect=True)
def test_32_of_128_full_job_stays_exclusive_until_explicit_closed_release(project):
    assert project.worker().run() == 0
    assert project.executions == 32
    challenger = project.authority_client("controller-b")
    replacement_key = Ed25519PrivateKey.generate()
    contender = dict(job_id="next-complete-job", host_id="another-synthetic-host", session_id="next-session",
        session_public_key=public_key_text(replacement_key.public_key()),
        **project.gate.job["scope"])
    with pytest.raises(AuthorityError):
        challenger.call("acquire", **contender)
    project.gate.begin_drain("synthetic batch completed, full budget remains")
    paused = project.gate.finish_drain()
    assert paused["phase"] == "PAUSED_BOUND"
    _, manifest = project.gate.build_checkpoint()
    assert manifest["settled_units"] == 32 and manifest["pending_units"] == 96
    with pytest.raises(AuthorityError):
        challenger.call("acquire", **contender)
    with pytest.raises(AuthorityError):
        project.gate.release(disposition="completed", reason="synthetic wrong full-job completion")
    with pytest.raises(ControlError, match="explicit"):
        project.gate.release(disposition="closed", reason="synthetic missing explicit close")
    previous_epoch = project.gate.binding["epoch"]
    project.gate.release(disposition="closed", reason="synthetic explicitly abandon 96 units", explicit_close=True)
    acquired = challenger.call("acquire", **contender)
    assert acquired["epoch"] > previous_epoch and acquired["job_id"] == "next-complete-job"


def test_interrupted_upload_with_failed_part_cleanup_remains_durably_unfinished(project, monkeypatch):
    project.hello()
    assignment = project.request("request_task")["payload"]
    original_unlink = Path.unlink
    cleanup_attempted = threading.Event()

    def unavailable_cleanup(path, *args, **kwargs):
        if path.parent == project.preparation.paths.upload_root and path.suffix == ".part":
            cleanup_attempted.set()
            raise PermissionError("synthetic locked partial upload")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unavailable_cleanup)
    with socket.create_connection(project.runtime._server.server_address, timeout=3) as stream:
        header = (f"POST {ARTIFACT_PATH_PREFIX}{assignment['attempt_id']} HTTP/1.1\r\n"
            f"Host: 127.0.0.1\r\nAuthorization: Bearer {project.preparation.api_token}\r\n"
            f"X-MSABP-Worker-ID: maid-a\r\nX-MSABP-Lease-Token: {assignment['lease_token']}\r\n"
            "Content-Type: application/zip\r\nContent-Length: 100\r\nConnection: close\r\n\r\n")
        stream.sendall(header.encode("ascii") + b"partial")
        stream.shutdown(socket.SHUT_WR)
        response = stream.recv(8192)
        assert b"500" in response.split(b"\r\n", 1)[0]
    assert cleanup_attempted.wait(3)
    assert list(project.preparation.paths.upload_root.glob("*.part"))
    with sqlite3.connect(project.preparation.paths.database) as db:
        assert db.execute("SELECT status FROM control_uploads").fetchone()[0] == "writing"
    project.gate.begin_drain("synthetic interrupted stream remains on disk")
    with pytest.raises(ControlError, match="upload"):
        project.gate.build_checkpoint()
