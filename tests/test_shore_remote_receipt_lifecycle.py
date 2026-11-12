"""Real collector shutdown and receipt-only restart with temporary loopback TLS.

The original serve_job loop drains, releases and closes its own listener. A
separate public recovery call must acknowledge the persisted idle after the
unchanged three-second grace, without authority traffic or additional work.
All identities, certificates, authority state and results are test-local.
"""
# ruff: noqa: F811 -- imported pytest fixtures are injected by name
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import http.client
from importlib.util import module_from_spec, spec_from_file_location
import ipaddress
import json
from pathlib import Path
import shutil
import socket
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("shore_control")
pytest.importorskip("cryptography")

from shore_control import tailnet_rpc
from shore_control.rpc import REQUEST_TIMEOUT

from test_shore_remote_controller import project  # noqa: F401
from test_shore_remote_tls_delay import (
    LoopbackCollectorClient, LoopbackVerifier, _certificate, _loopback_address,
    _REAL_CONNECT, _TAILNET_ADDRESS, completed_project,  # noqa: F401
)
from msabp_opt.simulation.distributed import control_remote_controller as controller
from msabp_opt.simulation.distributed import control_remote_http as remote_http
from msabp_opt.simulation.distributed.control_remote_bell import BellRunner
from msabp_opt.simulation.distributed.control_remote_worker import (
    BELL_ACTOR, LOCAL_NODE_ID, MAID_ACTOR, _persist_worker_idle,
)
from msabp_opt.simulation.distributed.protocol import make_message


def _state_snapshot(p):
    """Exclude HTTP replay nonces; include every task/attempt and authority row."""
    with sqlite3.connect(p.authority.database) as db:
        authority_operations = db.execute("SELECT * FROM operations ORDER BY operation_id").fetchall()
    with p.gate.journal.transaction() as db:
        project_operations = [tuple(row) for row in
            db.execute("SELECT * FROM control_operations ORDER BY operation_id").fetchall()]
    results = {str(path.relative_to(p.prepared.paths.results_root)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in p.prepared.paths.results_root.rglob("*") if path.is_file()}
    return (p.gate._run_snapshot("run"), authority_operations, project_operations, results)


def _no_authority_or_new_work(p, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Receipt recovery must remain offline and cannot create or execute work")

    monkeypatch.setattr(p.transport, "challenge", forbidden)
    monkeypatch.setattr(p.transport, "handle", forbidden)
    for name in ("rpc", "acquire", "poll_control", "issue_operation", "require_new_work",
                 "begin_drain", "finish_drain", "release"):
        monkeypatch.setattr(p.gate, name, forbidden)


@contextmanager
def _public_call(p, monkeypatch, function, *args, **kwargs):
    """Run the public blocking entry without bypassing its lifecycle."""
    outcome = SimpleNamespace(result=None, error=None)

    def invoke():
        try:
            outcome.result = function(*args, **kwargs)
        except BaseException as error:
            outcome.error = error

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    outcome.thread = thread
    try:
        yield outcome
    finally:
        thread.join(timeout=10)
        if thread.is_alive():
            # Failure cleanup only: exercise serve_job's existing Ctrl-C path
            # so a failed assertion cannot strand a listening test server.
            def interrupted():
                raise KeyboardInterrupt
            monkeypatch.setattr(p.gate, "poll_control", interrupted)
            thread.join(timeout=4)
        assert not thread.is_alive(), "Public collector entry did not close its listener"
        if outcome.error is not None:
            raise outcome.error


@pytest.fixture
def lifecycle(completed_project, monkeypatch):
    p = completed_project
    root = Path(p.preparation["output_root"])
    controller.immutable_json(root / "deployment" / "maid-runtime.json", p.runtime)
    # The shared fixture has a deliberately separate offline queue. Install
    # that already-completed fixture once BEFORE any listener is created.
    shutil.copytree(p.queue.root, root / "wake")
    cert, key, pin = _certificate(p.tmp_path)
    p.bootstrap["collector"]["tls_sha256"] = pin
    material = p.tmp_path / "collector-local.json"
    controller.immutable_json(material, {"schema": "shore.collector-local", "schema_version": 1,
        "cert_path": str(cert), "key_path": str(key), **p.bootstrap["collector"]})
    verifier = LoopbackVerifier(LOCAL_NODE_ID)
    verifier.network = p.bootstrap["network"]
    servers, allowed_ports = [], set()
    changed = threading.Condition()
    dropped = threading.Event()
    observed = SimpleNamespace(dropped_at=None)
    original_factory = tailnet_rpc.create_tls_server

    class LostFinalIdle(remote_http.RemoteCollectorHandler):
        def _reply(self, status, value):
            if (status == 200 and value == {"status": "RECORDED", "actor_id": MAID_ACTOR}
                    and not dropped.is_set()):
                observed.dropped_at = time.monotonic()
                dropped.set()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            return super()._reply(status, value)

    def create_loopback(handler, host, port, cert_path, key_path, source, nodes):
        assert host == "100.64.0.1" and port == 18767
        assert source is verifier and nodes == {LOCAL_NODE_ID}
        # Leave production certificate validation, policy construction,
        # authentication, serving, draining and shutdown completely intact.
        actual_handler = LostFinalIdle if not servers else handler
        server = original_factory(actual_handler, "127.0.0.1", 0,
                                  cert_path, key_path, LoopbackVerifier(LOCAL_NODE_ID), nodes)
        assert server.server_port != 18765
        with changed:
            allowed_ports.add(server.server_port)
            servers.append(server)
            changed.notify_all()
        return server

    def connect(sock, address):
        _loopback_address(address[0])
        assert address[1] in allowed_ports and address[1] != 18765
        return _REAL_CONNECT(sock, address)

    monkeypatch.setattr(tailnet_rpc, "create_tls_server", create_loopback)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(tailnet_rpc, "tailnet_address", lambda address:
        _loopback_address(address) if ipaddress.ip_address(address).is_loopback
        else _TAILNET_ADDRESS(address))

    def wait_server(index):
        with changed:
            assert changed.wait_for(lambda: len(servers) > index, timeout=5), "Public entry did not bind TLS"
            return servers[index]

    def client(server, actor=MAID_ACTOR):
        return LoopbackCollectorClient(p, None, actor,
            f"https://127.0.0.1:{server.server_port}", pin)

    return SimpleNamespace(project=p, root=root, material=material, verifier=verifier,
        servers=servers, wait_server=wait_server, client=client, dropped=dropped, observed=observed)


def _release_via_real_loop(lifecycle, monkeypatch):
    env, p = lifecycle, lifecycle.project
    with _public_call(p, monkeypatch, controller.serve_job, p.preparation, p.gate,
            p.bootstrap, env.verifier, collector_config=env.material) as call:
        server = env.wait_server(0)
        runner = BellRunner(p.bell, env.client(server, BELL_ACTOR),
            lambda *_: pytest.fail("Completed Bell cannot relaunch"), ready_root=p.maid_config.state_root)
        assert runner.report_idle() == {"status": "RECORDED", "actor_id": BELL_ACTOR}
        with pytest.raises((OSError, http.client.HTTPException)):
            _persist_worker_idle(p.maid_config, p.worker, p.clients[MAID_ACTOR], env.client(server))
        assert env.dropped.wait(timeout=1)
        worker_idle = (p.maid_config.state_root / "idle.json").read_bytes()
        collector_idle = env.root / "wake" / "idle" / (MAID_ACTOR + ".json")
        assert json.loads(collector_idle.read_bytes()) == {"record": json.loads(worker_idle)}
        call.thread.join(timeout=10)
        assert not call.thread.is_alive() and call.error is None
        assert call.result["phase"] == p.gate.job["phase"] == "RELEASED"
        assert time.monotonic() - env.observed.dropped_at >= 3.0
        assert server.socket.fileno() == -1
        with pytest.raises((OSError, http.client.HTTPException)):
            env.client(server).request_json(remote_http.IDLE_PATH, {"record": json.loads(worker_idle)})
    return worker_idle


def _post(client, path, body, *, content_type="application/json", attempt="", lease=""):
    """Send signed headers before a delayed body to exercise early TLS denial.

    A handler that replies/closes without draining a validated bounded body can
    discard its own 403/404 when the client's later TLS record arrives. Keep
    these separate writes explicit; coalescing them hides that transport race.
    """
    raw = body if isinstance(body, bytes) else remote_http._json_bytes(body)
    auth = client.signing_client._auth(path, hashlib.sha256(raw).hexdigest(), len(raw),
        content_type, attempt_id=attempt, lease_token=lease)
    connection = tailnet_rpc.PinnedHTTPSConnection(client.endpoint, client.pin, client.verifier,
        controller.HOST_NODE, timeout=REQUEST_TIMEOUT)
    response = None
    try:
        connection._http_vsn, connection._http_vsn_str = 10, "HTTP/1.0"
        connection.putrequest("POST", path, skip_host=True, skip_accept_encoding=True)
        for name, value in {
            "Host": client.endpoint.removeprefix("https://"), "Content-Type": content_type,
            "Content-Length": str(len(raw)), "Connection": "close", remote_http.AUTH_HEADER: auth,
        }.items():
            connection.putheader(name, value)
        connection.endheaders()
        time.sleep(0.025)
        connection.send(raw)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        if response is not None:
            response.close()
        connection.close()


@pytest.mark.parametrize("project", [4], indirect=True)
def test_public_receipt_restart_after_actual_release_and_lost_final_idle(lifecycle, monkeypatch):
    env, p = lifecycle, lifecycle.project
    idle = _release_via_real_loop(env, monkeypatch)
    collector_idle = (env.root / "wake" / "idle" / (MAID_ACTOR + ".json")).read_bytes()
    before, requests = _state_snapshot(p), copy.deepcopy(p.transport.requests)
    original_binding = copy.deepcopy(p.gate.binding)
    _no_authority_or_new_work(p, monkeypatch)
    with _public_call(p, monkeypatch, controller.serve_receipts, p.preparation, p.gate,
            p.bootstrap, env.verifier, collector_config=env.material, seconds=4) as call:
        server = env.wait_server(1)
        assert server is not env.servers[0] and env.servers[0].socket.fileno() == -1
        assert env.client(server).request_json(remote_http.IDLE_PATH, json.loads(collector_idle)) == {
            "status": "RECORDED", "actor_id": MAID_ACTOR}
        bell_idle = (env.root / "wake" / "idle" / (BELL_ACTOR + ".json")).read_bytes()
        assert env.client(server, BELL_ACTOR).request_json(remote_http.IDLE_PATH,
            json.loads(bell_idle)) == {"status": "RECORDED", "actor_id": BELL_ACTOR}
        for path, actor, body in (
            ("/v1/wake/poll", BELL_ACTOR, {}),
            ("/v1/wake/complete", BELL_ACTOR, {}),
            (remote_http.MESSAGE_PATH, MAID_ACTOR, make_message("request_task", run_id="run",
                worker_id=MAID_ACTOR, protocol_version=2, payload={})),
        ):
            assert _post(env.client(server, actor), path, body)[0] == 403
        assert _post(env.client(server), "/v1/unknown", {})[0] == 404
        assert _post(env.client(server), remote_http.ARTIFACT_PATH_PREFIX + "forbidden",
            b"forbidden", content_type="application/zip", attempt="forbidden", lease="never")[0] == 403
        changed_idle = json.loads(collector_idle)
        changed_idle["record"]["payload"]["permit_ids"] = []
        changed_idle["record"] = p.clients[MAID_ACTOR].signer.sign(
            controller.IDLE_DOMAIN, changed_idle["record"]["payload"])
        assert 400 <= _post(env.client(server), remote_http.IDLE_PATH, changed_idle)[0] < 500
    assert call.error is None and server.socket.fileno() == -1
    assert p.gate.binding == original_binding
    assert (p.maid_config.state_root / "idle.json").read_bytes() == idle
    assert (env.root / "wake" / "idle" / (MAID_ACTOR + ".json")).read_bytes() == collector_idle
    assert _state_snapshot(p) == before and p.transport.requests == requests
    assert len(before[1]) == len(before[2]) == 5
    assert p.state.progress("run").completed == 4


@pytest.mark.parametrize("project", [4], indirect=True)
@pytest.mark.parametrize("damage", ["authority_signature", "missing_proof", "artifact"])
def test_receipt_restart_rejects_damaged_terminal_evidence_without_refetch(lifecycle, monkeypatch, damage):
    env, p = lifecycle, lifecycle.project
    idle = _release_via_real_loop(env, monkeypatch)
    collector_idle = (env.root / "wake" / "idle" / (MAID_ACTOR + ".json")).read_bytes()
    before, requests = _state_snapshot(p), copy.deepcopy(p.transport.requests)
    with p.gate.journal.transaction() as db:
        descriptors = [json.loads(row[0]) for row in db.execute("SELECT descriptor FROM control_operations")]
    descriptor = next(row for row in descriptors if row["worker_id"] == MAID_ACTOR)
    proof_path = env.root / "wake" / "terminal" / (
        hashlib.sha256(descriptor["operation_id"].encode()).hexdigest() + ".json")
    if damage == "missing_proof":
        proof_path.unlink()
    elif damage == "authority_signature":
        proof = json.loads(proof_path.read_bytes())
        proof["response"]["payload"]["result"]["status"] = "AWAITING_PEER_RECEIPT"
        proof_path.write_bytes(remote_http._json_bytes(proof))
    else:
        ready_path = p.tmp_path / "ready.json"
        # The resolved wake carries the copied, durable evidence path, not the
        # originating device's ready file.
        resolved = json.loads((env.root / "wake" / "resolved.json").read_bytes())
        ready_path = Path(resolved["evidence_root"]) / ready_path.name
        ready_path.write_bytes(ready_path.read_bytes() + b" ")
    _no_authority_or_new_work(p, monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        controller.serve_receipts(p.preparation, p.gate, p.bootstrap, env.verifier,
                                  collector_config=env.material, seconds=1)
    assert len(env.servers) == 1, "Reject changed proof before opening a recovery listener"
    assert (p.maid_config.state_root / "idle.json").read_bytes() == idle
    assert (env.root / "wake" / "idle" / (MAID_ACTOR + ".json")).read_bytes() == collector_idle
    assert _state_snapshot(p) == before and p.transport.requests == requests


@pytest.mark.parametrize("project", [4], indirect=True)
def test_original_signed_paused_job_can_reopen_receipts_without_releasing(lifecycle, monkeypatch):
    env, p = lifecycle, lifecycle.project
    p.gate.begin_drain("Temporary isolated fixture paused after completed evaluations")
    queue = controller.CollectorWakeQueue(env.root / "wake", p.gate, p.runtime)

    def direct_client(actor):
        return SimpleNamespace(request_json=lambda path, body:
            queue.handle(path, body, actor_id=actor, node_id=LOCAL_NODE_ID))

    runner = BellRunner(p.bell, direct_client(BELL_ACTOR),
        lambda *_: pytest.fail("Paused Bell cannot relaunch"), ready_root=p.maid_config.state_root)
    assert runner.report_idle()["status"] == "RECORDED"
    _persist_worker_idle(p.maid_config, p.worker, p.clients[MAID_ACTOR], direct_client(MAID_ACTOR))
    assert queue.idle_pending() == 0
    assert p.gate.finish_drain(worker_outboxes_pending=0)["phase"] == "PAUSED_BOUND"
    saved_job = copy.deepcopy(p.gate.job)
    idle_path = env.root / "wake" / "idle" / (MAID_ACTOR + ".json")
    idle = idle_path.read_bytes()
    before, requests = _state_snapshot(p), copy.deepcopy(p.transport.requests)
    env.dropped.set()  # This first listener is receipt-only, not a lossy active serve.
    _no_authority_or_new_work(p, monkeypatch)
    with _public_call(p, monkeypatch, controller.serve_receipts, p.preparation, p.gate,
            p.bootstrap, env.verifier, collector_config=env.material, seconds=1) as call:
        server = env.wait_server(0)
        assert env.client(server).request_json(remote_http.IDLE_PATH, json.loads(idle)) == {
            "status": "RECORDED", "actor_id": MAID_ACTOR}
    assert call.result["phase"] == "PAUSED_BOUND" and server.socket.fileno() == -1
    assert p.gate.job == saved_job and idle_path.read_bytes() == idle
    assert _state_snapshot(p) == before and p.transport.requests == requests


@pytest.mark.parametrize("phase", ["ACTIVE", "DRAINING", "PAUSED_BOUND", "RELEASED"])
def test_receipts_require_original_signed_terminal_job_not_local_phase(project, monkeypatch, phase):
    p = project
    # Alter only the local summary. No signed finish or release ever happened.
    p.gate._set("job", {**p.gate.job, "phase": phase})
    requests = copy.deepcopy(p.transport.requests)
    _no_authority_or_new_work(p, monkeypatch)
    with pytest.raises(ValueError, match="PAUSED_BOUND|signed terminal"):
        controller.serve_receipts(p.preparation, p.gate, p.bootstrap, None,
                                  collector_config="NEVER-READ.json", seconds=1)
    assert p.transport.requests == requests


@pytest.mark.parametrize("seconds", [0, -1, 3601, 1.0, True])
def test_receipt_window_is_explicitly_bounded_before_any_state_access(seconds):
    with pytest.raises(ValueError, match="integer seconds"):
        controller.serve_receipts({}, None, {}, None,
                                  collector_config="NEVER-READ.json", seconds=seconds)


def test_receipt_cli_requires_live_approval_before_opening_any_files(monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "scripts/simulation/shore_synthetic_princess.py"
    spec = spec_from_file_location("receipt_lifecycle_princess_cli", path)
    entry = module_from_spec(spec)
    spec.loader.exec_module(entry)
    with pytest.raises(SystemExit) as error:
        entry.main(["serve-receipts", "--preparation", "NEVER-READ.json", "--seconds", "1"])
    assert error.value.code == 2
    assert "prior review" in capsys.readouterr().err
