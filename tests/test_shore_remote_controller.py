"""Collector wake orchestration with generated keys and in-process authority only."""
from __future__ import annotations

import builtins
import copy
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import platform
import socket
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("shore_control")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.authority import Authority, digest
from shore_control.client import Client, MemorySigner
from shore_control.crypto import canonical_json, fingerprint, public_key_text

from msabp_opt.simulation.distributed import control_remote_controller as controller
from msabp_opt.simulation.distributed import control_remote_http
from msabp_opt.simulation.distributed.control_authority import ControlAuthority, ControlError
from msabp_opt.simulation.distributed.control_remote_bell import BellRunner, READY_DOMAIN, WakeCoordinator
from msabp_opt.simulation.distributed.control_remote_worker import (
    BELL_ACTOR, DEVICE_ID, LOCAL_NODE_ID, MAID_ACTOR, MODE, PROJECT_SOURCE_FILES,
    RemoteSyntheticMaidConfig, SCHEMA, SHORE_SOURCE_FILES, _persist_worker_idle,
)
from msabp_opt.simulation.distributed.control_synthetic import SyntheticSession, synthetic_case
from msabp_opt.simulation.distributed.control_worker import ShoreWorker
from msabp_opt.simulation.distributed.maid import Maid, MaidRuntimeConfig
from msabp_opt.simulation.distributed.protocol import make_message
from msabp_opt.simulation.distributed.state import PrincessState


class LocalAuthorityTransport:
    def __init__(self, authority):
        self.authority, self.requests, self.drop_method = authority, [], None

    def challenge(self, actor):
        return self.authority.challenge(actor)

    def handle(self, request):
        self.requests.append(copy.deepcopy(request))
        response = self.authority.handle(request)
        if request["payload"]["method"] == self.drop_method:
            self.drop_method = None
            raise TimeoutError("Synthetic response lost after commit")
        return response


@pytest.fixture
def project(tmp_path, monkeypatch, request):
    monkeypatch.setenv("MSABP_CONTROL_MODE", MODE)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Controller tests must never use sockets or processes")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(platform, "node", lambda: LOCAL_NODE_ID)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cst" or name.startswith("cst."):
            raise AssertionError("CST is outside offline controller tests")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    keys = {actor: Ed25519PrivateKey.generate() for actor in (
        "authority", controller.CONTROLLER_ACTOR, "session:session", BELL_ACTOR, MAID_ACTOR)}
    registrations = {actor: {"public_key": public_key_text(keys[actor].public_key()),
        "device_id": DEVICE_ID, "tailnet_node_id": LOCAL_NODE_ID, "role": role,
        "allowed_actions": [action]} for actor, role, action in (
            (BELL_ACTOR, "bell", "wake"), (MAID_ACTOR, "maid", "attempt"))}
    fleet = "shore-princess-" + "1234" * 6
    bootstrap = {"schema": "shore.remote-bootstrap", "schema_version": 1, "mode": MODE,
        "fleet_id": fleet, "authority_id": fingerprint(keys["authority"].public_key()),
        "authority_public_key": public_key_text(keys["authority"].public_key()),
        "endpoint": "https://100.64.0.1:18766", "tls_sha256": "a" * 64,
        "controllers": {controller.CONTROLLER_ACTOR: public_key_text(keys[controller.CONTROLLER_ACTOR].public_key())},
        "workers": registrations, "controller_nodes": [controller.HOST_NODE, LOCAL_NODE_ID],
        "network": {"tailnet_name": "offline@example.test", "magic_dns_suffix": "example.ts.net",
            "local_node_id": controller.HOST_NODE, "local_address": "100.64.0.1", "nodes": {
                controller.HOST_NODE: {"address": "100.64.0.1", "dns_name": "host.example.ts.net."},
                LOCAL_NODE_ID: {"address": "100.64.0.2", "dns_name": "worker.example.ts.net."}}},
        "collector": {"endpoint": "https://100.64.0.1:18767", "tls_sha256": "b" * 64}}
    source = tmp_path / "samples.csv"
    points = getattr(request, "param", 1)
    source.write_bytes(("sample_id,value\n" + "".join(f"case-{i},{i}\n" for i in range(points))).encode())
    preparation = {"output_root": str(tmp_path / "project"), "run_id": "run", "points": points,
                   "paths": {"samples_csv": str(source)}}
    prepared = controller.prepare_project_run(preparation, bootstrap)
    authority = Authority(tmp_path / "authority.sqlite", keys["authority"], fleet_id=fleet,
                          controllers=bootstrap["controllers"], workers=registrations)
    transport = LocalAuthorityTransport(authority)
    clients = {actor: Client(transport, actor, MemorySigner(key), keys["authority"].public_key(), fleet)
               for actor, key in keys.items() if actor != "authority"}
    gate = ControlAuthority(tmp_path / "gate.sqlite", controller_client=clients[controller.CONTROLLER_ACTOR],
        session_client=clients["session:session"], worker_registrations=registrations)
    manifest, policy = tmp_path / "manifest.json", tmp_path / "policy.json"
    manifest.write_bytes(canonical_json({"job": "synthetic-test", "approved_units": points}))
    policy.write_bytes(canonical_json({"schema": "msabp.budget-policy", "schema_version": 1,
                                       "unit_key": "evaluation_id", "settled_outcomes": ["completed"]}))
    spec = {"schema": "shore.job-spec", "schema_version": 1, "approved_units": points,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "budget_unit": "objective_evaluation", "accounting_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}
    gate.acquire(job_id="job", host_id=controller.HOST_NODE, session_id="session",
        session_public_key=public_key_text(keys["session:session"].public_key()), project_id="MSABP_2",
        collector_url=bootstrap["collector"]["endpoint"], data_root=preparation["output_root"], job_spec=spec)
    gate.bind_job_documents(manifest, policy)
    gate.register_run("run", hashlib.sha256(prepared.paths.runtime_json.read_bytes()).hexdigest(), prepared.worklist.worklist.sha256)
    gate.bind_run_plan("run", prepared.paths.runtime_json)
    runtime = {"schema": SCHEMA, "schema_version": 1, "mode": MODE, "run_id": "run",
        "worker_id": MAID_ACTOR, "local_node_id": LOCAL_NODE_ID,
        "bootstrap_path": str(tmp_path / "public" / "bootstrap.json"), "bootstrap_sha256": "c" * 64,
        "source_root": str(tmp_path / "source"), "source_hashes": {name: "a" * 64 for name in PROJECT_SOURCE_FILES},
        "shore_source_root": str(tmp_path / "shore-source"),
        "shore_source_hashes": {name: "b" * 64 for name in SHORE_SOURCE_FILES},
        "worker_config_path": str(tmp_path / "identity" / "worker-config.json"),
        "csv_path": str(source), "csv_sha256": prepared.worklist.worklist.sha256, "api_token": prepared.api_token,
        "binding": {**gate.binding, **gate.job["scope"], "fleet_id": fleet},
        "state_root": str(tmp_path / "maid-state"), "output_root": str(tmp_path / "maid-output")}
    queue = controller.CollectorWakeQueue(tmp_path / "collector-wake", gate, runtime)
    state = PrincessState(prepared.paths.database, control_authority=gate)
    state.bind_control_run("run")
    coordinator = controller.WakeGatedCoordinator(state, wake_queue=queue, run_id="run",
        csv_sha256=prepared.worklist.worklist.sha256, results_dir=prepared.paths.results_root,
        incoming_dir=prepared.paths.incoming_root, max_attempts=1)
    gate.bind_run_state("run", prepared.paths.database)
    bell = WakeCoordinator(tmp_path / "bell-state", clients[BELL_ACTOR], keys["authority"].public_key(),
        binding=runtime["binding"], registration=registrations[BELL_ACTOR], target_actor_id=MAID_ACTOR,
        target_registration=registrations[MAID_ACTOR], runtime=runtime, run_id="run")

    def send(kind, **payload):
        return coordinator.handle_message(make_message(kind, run_id="run", worker_id=MAID_ACTOR,
                                                        protocol_version=2, payload=payload))

    def poll():
        return queue.handle("/v1/wake/poll", {}, actor_id=BELL_ACTOR, node_id=LOCAL_NODE_ID)

    def complete(body=None):
        return queue.handle("/v1/wake/complete", body or bell.completion_payload(queue.intent["attempt_id"]),
                            actor_id=BELL_ACTOR, node_id=LOCAL_NODE_ID)

    def started():
        assignment = poll()["assignment"]
        permit = bell.accept(assignment)
        bell.launch(permit["attempt_id"], lambda *_: None, ready_path=tmp_path / "unused-ready")
        record = bell.record(permit["attempt_id"])
        ready = clients[MAID_ACTOR].signer.sign(READY_DOMAIN, {"schema": READY_DOMAIN, "schema_version": 1,
            "launch_id": record["launch_id"], "permit_id": permit["permit_id"],
            "runtime_sha256": digest(runtime), "actor_id": MAID_ACTOR,
            **{k: permit[k] for k in ("device_id", "fleet_id", "job_id", "epoch", "run_id")},
            "pid": 123, "process_started_utc": "2026-10-16T12:00:00+00:00"})
        ready_path = tmp_path / "ready.json"
        ready_path.write_bytes(canonical_json(ready))
        bell.record_ready(permit["attempt_id"], ready_path)
        return permit

    fixture = SimpleNamespace(**locals())
    try:
        yield fixture
    finally:
        state.close()


def test_wake_requires_matching_dual_receipts_before_first_task(project):
    p = project
    assert p.send("hello", csv_sha256=p.runtime["csv_sha256"])["type"] == "welcome"
    assert p.send("request_task")["type"] == "wait"
    permit = p.started()
    assert not p.queue.ready()
    assert p.complete()["status"] == "AWAITING_PEER_RECEIPT"
    assert not p.queue.ready()
    assert p.send("request_task")["type"] == "wait"
    assert p.state.progress("run").running == 0
    assert p.bell.report(permit["attempt_id"])["status"] == "RESOLVED"
    assert p.complete()["status"] == "RESOLVED"
    assert p.queue.ready()
    assignment = p.send("request_task")
    assert assignment["type"] == "assignment"
    assert assignment["payload"]["control_authority"]["descriptor"]["worker_id"] == MAID_ACTOR
    assert p.state.progress("run").running == 1
    assert p.poll() == {"assignment": None}
    restarted = controller.CollectorWakeQueue(p.tmp_path / "collector-wake", p.gate, p.runtime)
    assert restarted.ready()


def test_repeated_poll_and_lost_dispatch_response_use_same_wake_operation(project):
    p = project
    p.transport.drop_method = "validate_dispatch"
    with pytest.raises(TimeoutError):
        p.poll()
    original = next(r for r in p.transport.requests if r["payload"]["method"] == "validate_dispatch")
    assignment = p.poll()["assignment"]
    assert p.poll()["assignment"] == assignment
    calls = [r for r in p.transport.requests if r["payload"]["method"] == "validate_dispatch"]
    assert calls == [original, original]
    with p.gate.journal.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM control_operations").fetchone()[0] == 1


def test_unverified_resolved_summary_cannot_replace_authority_receipts(project):
    p = project
    permit = p.started()
    body = p.bell.completion_payload(permit["attempt_id"])
    result = p.bell.checked_result(permit["attempt_id"])
    controller.immutable_json(p.queue.root / "resolved.json", {"completion": body,
        "evidence_root": result["root"], "reply": {"permit_id": permit["permit_id"], "status": "RESOLVED"}})
    try:
        ready = p.queue.ready()
    except (ValueError, ControlError):
        ready = False
    assert ready is False
    assert p.state.progress("run").running == 0


def test_not_started_dual_receipt_never_opens_maid_gate_or_reissues_wake(project):
    p = project
    permit = p.bell.accept(p.poll()["assignment"])
    p.bell.not_started(permit["attempt_id"])
    p.bell.report(permit["attempt_id"])
    assert p.complete()["status"] == "RESOLVED"
    assert not p.queue.ready()
    assert p.send("request_task")["type"] == "wait"
    assert p.poll() == {"assignment": None}


def test_drain_retains_late_wake_collection_without_new_dispatch(project):
    p = project
    permit = p.started()
    p.gate.begin_drain("offline simulated F removal")
    # Existing permits remain retrievable for reconciliation, not new work.
    assert p.poll()["assignment"]["control_authority"]["permit"]["payload"] == permit
    p.bell.report(permit["attempt_id"])
    assert p.complete()["status"] == "RESOLVED"
    assert p.queue.ready()
    p.send("hello", csv_sha256=p.runtime["csv_sha256"])
    assert p.send("request_task")["type"] == "stop"
    assert p.state.progress("run").running == 0


@pytest.mark.parametrize("actor,node", [(MAID_ACTOR, LOCAL_NODE_ID), (BELL_ACTOR, controller.HOST_NODE)])
def test_wake_callback_rejects_wrong_role_or_actual_source(project, actor, node):
    with pytest.raises(ValueError, match="actor/source|Only Bell"):
        project.queue.handle("/v1/wake/poll", {}, actor_id=actor, node_id=node)


def test_ready_tamper_rejected_even_after_resolution(project):
    p = project
    permit = p.started()
    p.bell.report(permit["attempt_id"])
    p.complete()
    saved = json.loads((p.queue.root / "resolved.json").read_bytes())
    path = Path(saved["evidence_root"]) / "ready.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        p.queue.ready()


def test_server_start_creates_collector_tables_before_binding_snapshot(project, monkeypatch):
    p = project
    p.state.close()
    # Model a fresh startup, not a collector already initialized by the fixture.
    with sqlite3.connect(p.prepared.paths.database) as db:
        db.execute("DROP TABLE control_collection")
        db.execute("DROP TABLE control_uploads")
    root = Path(p.preparation["output_root"])
    controller.immutable_json(root / "deployment" / "maid-runtime.json", p.runtime)
    material_path = p.tmp_path / "collector-local.json"
    controller.immutable_json(material_path, {"schema": "shore.collector-local", "schema_version": 1,
        "cert_path": str(p.tmp_path / "synthetic-certificate-not-read.pem"),
        "key_path": str(p.tmp_path / "synthetic-key-not-read.pem"), **p.bootstrap["collector"]})
    calls = []
    server = SimpleNamespace(serve_forever=lambda: None, shutdown=lambda: None, server_close=lambda: None)

    def factory(coordinator, **kwargs):
        calls.append((coordinator, kwargs))
        assert isinstance(coordinator, controller.WakeGatedCoordinator)
        p.gate._run_snapshot("run")
        return server

    def stop_after_start():
        raise KeyboardInterrupt

    monkeypatch.setattr(control_remote_http, "create_remote_server", factory)
    monkeypatch.setattr(p.gate, "poll_control", stop_after_start)
    with pytest.raises(KeyboardInterrupt):
        controller.serve_job(p.preparation, p.gate, p.bootstrap, SimpleNamespace(), collector_config=material_path)
    assert len(calls) == 1
    assert p.gate.job["phase"] == "DRAINING"


def idle_record(project, actor, permit_ids, *, signer=None, **changes):
    p = project
    payload = {"schema": controller.IDLE_DOMAIN, "schema_version": 1, "mode": MODE,
        "actor_id": actor, "fleet_id": p.fleet, "job_id": p.gate.binding["job_id"],
        "epoch": p.gate.binding["epoch"], "run_id": "run",
        "runtime_sha256": p.queue.runtime.runtime_sha256, "outboxes_pending": 0,
        "permit_ids": permit_ids, **changes}
    return {"record": (signer or p.clients[actor].signer).sign(controller.IDLE_DOMAIN, payload)}


def test_idle_requires_each_actor_real_permits_and_no_pending_work(project):
    p = project
    permit = p.started()
    p.bell.report(permit["attempt_id"])
    p.complete()
    assert p.queue.idle_pending() == 2
    good = idle_record(p, BELL_ACTOR, [permit["permit_id"]])
    assert p.queue.record_idle(good, actor_id=BELL_ACTOR) == {"status": "RECORDED", "actor_id": BELL_ACTOR}
    assert p.queue.idle_pending() == 1
    with pytest.raises(ValueError, match="omits|invents"):
        p.queue.record_idle(idle_record(p, BELL_ACTOR, []), actor_id=BELL_ACTOR)
    with pytest.raises(ValueError, match="signature"):
        p.queue.record_idle(idle_record(p, MAID_ACTOR, [], signer=p.clients[BELL_ACTOR].signer), actor_id=MAID_ACTOR)
    with pytest.raises(ValueError, match="new work"):
        p.queue.record_idle(idle_record(p, MAID_ACTOR, []), actor_id=MAID_ACTOR)
    with pytest.raises(ValueError, match="frozen actor"):
        p.queue.record_idle(idle_record(p, BELL_ACTOR, [permit["permit_id"]], outboxes_pending=1), actor_id=BELL_ACTOR)
    p.send("hello", csv_sha256=p.runtime["csv_sha256"])
    attempt = p.send("request_task")["payload"]["control_authority"]["permit"]["payload"]
    p.gate.begin_drain("synthetic controlled stop")
    with pytest.raises(ValueError, match="unresolved"):
        p.queue.record_idle(idle_record(p, MAID_ACTOR, [attempt["permit_id"]]), actor_id=MAID_ACTOR)
    assert p.queue.idle_pending() == 1


class DirectSignedClient:
    """Real auth policy and artifact handlers; no network/socket/daemon exists."""
    def __init__(self, project, policy, actor):
        self.project, self.policy, self.actor = project, policy, actor
        network = project.bootstrap["network"]
        verifier = SimpleNamespace(network={**network, "local_node_id": LOCAL_NODE_ID,
            "local_address": network["nodes"][LOCAL_NODE_ID]["address"]})
        self.signing_client = control_remote_http.RemotePrincessClient(project.bootstrap, actor,
            project.clients[actor].signer, verifier, run_id="run")

    def authenticate(self, path, raw, *, attempt="", lease=""):
        content_type = "application/zip" if attempt else "application/json"
        headers = Message()
        headers[control_remote_http.AUTH_HEADER] = self.signing_client._auth(
            path, hashlib.sha256(raw).hexdigest(), len(raw), content_type,
            attempt_id=attempt, lease_token=lease)
        auth = self.policy.authenticate(headers, node_id=LOCAL_NODE_ID, path=path,
                                         length=len(raw), content_type=content_type)
        self.policy.consume(auth)
        return auth

    def send(self, message):
        auth = self.authenticate(control_remote_http.MESSAGE_PATH, control_remote_http._json_bytes(message))
        return self.policy.message(auth, message)

    def upload_artifact(self, attempt_id, worker_id, lease_token, archive_path):
        assert worker_id == self.actor
        raw = Path(archive_path).read_bytes()
        assert len(raw) < control_remote_http.MAX_WIRE_BYTES
        auth = self.authenticate(control_remote_http.ARTIFACT_PATH_PREFIX + attempt_id, raw,
                                 attempt=attempt_id, lease=lease_token)
        handler = object.__new__(control_remote_http.RemoteCollectorHandler)
        handler.rfile = io.BytesIO(raw)
        # This direct handler fixture bypasses do_POST's validated body counter.
        handler._body_remaining = len(raw)
        return handler._artifact(self.policy, auth, len(raw))

    def request_json(self, path, body):
        self.authenticate(path, control_remote_http._json_bytes(body))
        return self.project.queue.handle(path, body, actor_id=self.actor, node_id=LOCAL_NODE_ID)


@pytest.mark.parametrize("project", [1, 4], indirect=True)
def test_one_and_four_point_synthetic_full_settlement_with_actual_idle(project):
    p = project
    wake_permit = p.started()
    p.bell.report(wake_permit["attempt_id"])
    assert p.complete()["status"] == "RESOLVED"
    assert p.bell.resolved(wake_permit["attempt_id"])
    policy = control_remote_http._CollectorPolicy(p.coordinator, p.bootstrap,
        SimpleNamespace(network=p.bootstrap["network"]), p.prepared.paths.upload_root, p.queue.handle)
    client = DirectSignedClient(p, policy, MAID_ACTOR)
    config = RemoteSyntheticMaidConfig.from_mapping(p.runtime)
    worker = ShoreWorker(config.state_root / "control", p.clients[MAID_ACTOR], p.keys["authority"].public_key(),
        binding=p.runtime["binding"], registration=p.registrations[MAID_ACTOR], csv_sha256=p.runtime["csv_sha256"])
    maid_config = MaidRuntimeConfig(run_id="run", worker_id=MAID_ACTOR,
        princess_url=p.bootstrap["collector"]["endpoint"], api_token=p.prepared.api_token,
        csv_path=p.prepared.paths.worklist_csv, csv_sha256=p.runtime["csv_sha256"],
        project_path=p.tmp_path / "NEVER-OPEN.cst", output_root=Path(p.runtime["output_root"]),
        dry_run=True, protocol_mode=MODE, poll_seconds=0.001, heartbeat_seconds=1)
    maid = Maid(maid_config, client=client, artifact_client=client, control_worker=worker,
                session=SyntheticSession(), run_csv_row=synthetic_case)
    assert maid.run() == 0
    assert p.state.progress("run").completed == p.points
    assert not worker.pending_assignments()
    assert p.queue.idle_pending() == 2
    bell_runner = BellRunner(p.bell, DirectSignedClient(p, policy, BELL_ACTOR),
        lambda *_: pytest.fail("Completed Bell must never relaunch"), ready_root=config.state_root)
    assert bell_runner.report_idle() == {"status": "RECORDED", "actor_id": BELL_ACTOR}
    assert p.queue.idle_pending() == 1
    _persist_worker_idle(config, worker, p.clients[MAID_ACTOR], client)
    assert p.queue.idle_pending() == 0
    p.gate.begin_drain("All isolated synthetic evaluations settled")
    assert p.gate.finish_drain(worker_outboxes_pending=p.queue.idle_pending())["phase"] == "PAUSED_BOUND"
    assert p.gate.release(disposition="completed", reason="Offline synthetic fixture complete")["phase"] == "RELEASED"
    manifests = list(p.prepared.paths.results_root.rglob("manifest.json"))
    assert len(manifests) == p.points
    for path in manifests:
        result = json.loads(path.read_text())
        assert result["synthetic"] is True and result["physical_result"] is False
    assert not (p.tmp_path / "NEVER-OPEN.cst").exists()
