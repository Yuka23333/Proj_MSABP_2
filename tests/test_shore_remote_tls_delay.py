"""Real loopback TLS/deadline coverage with throwaway identities and databases.

Only address/source resolution is synthetic. Both HTTPS listeners, TLS pins,
the unmodified five-second deadline, signed RPC and Authority are real. The
four-point job is completed in-process before enabling transport delays, so
this suite never reaches the deployed authority, keys, remote worker or CST.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import ipaddress
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("shore_control")
pytest.importorskip("cryptography")

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from shore_control import tailnet_rpc
from shore_control.rpc import REQUEST_TIMEOUT

from test_shore_remote_controller import (
    DirectSignedClient, project,  # noqa: F401 - imported pytest fixture
)
from msabp_opt.simulation.distributed import control_remote_controller as controller
from msabp_opt.simulation.distributed import control_remote_http as remote_http
from msabp_opt.simulation.distributed.control_remote_bell import BellRunner
from msabp_opt.simulation.distributed.control_remote_worker import (
    BELL_ACTOR, LOCAL_NODE_ID, MAID_ACTOR, MODE, RemoteSyntheticMaidConfig,
    _persist_worker_idle, _read_idle_permits, _send_idle,
)
from msabp_opt.simulation.distributed.control_synthetic import SyntheticSession, synthetic_case
from msabp_opt.simulation.distributed.control_worker import ShoreWorker
from msabp_opt.simulation.distributed.maid import Maid, MaidRuntimeConfig


_REAL_CONNECT = socket.socket.connect
_TAILNET_ADDRESS = tailnet_rpc.tailnet_address


def _loopback_address(address):
    if not ipaddress.ip_address(address).is_loopback:
        raise AssertionError("TLS integration must only use literal loopback addresses")
    return address


class LoopbackVerifier:
    """Test-only node resolver; never executes Tailscale or trusts HTTP headers."""

    def __init__(self, source_node):
        self.source_node = source_node
        self.network = {"local_address": "127.0.0.1", "local_node_id": controller.HOST_NODE,
            "nodes": {node: {"address": "127.0.0.1"}
                      for node in (controller.HOST_NODE, LOCAL_NODE_ID)}}

    def verify_local(self):
        return {}

    def verify_peer(self, address, port, expected_node_id=None):
        _loopback_address(address)
        assert 0 < port <= 65535 and port != 18765
        return expected_node_id or self.source_node


def _certificate(root):
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "msabp-loopback-only")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
        .sign(key, hashes.SHA256()))
    cert_path, key_path = root / "loopback-cert.pem", root / "loopback-key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return cert_path, key_path, hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


@contextmanager
def _running(server):
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


class DelayedAuthority:
    """Delay each real HTTPS RPC leg without changing its signed semantics."""

    def __init__(self, authority):
        self.authority = authority
        self.delay = 0.0
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.authority, name)

    def challenge(self, actor):
        self.calls.append(("challenge", actor))
        time.sleep(self.delay)
        return self.authority.challenge(actor)

    def handle(self, request):
        self.calls.append(("request", request["payload"]["method"]))
        time.sleep(self.delay)
        return self.authority.handle(request)


class LoopbackCollectorClient(DirectSignedClient):
    """Actual pinned TLS carrying the production actor-signed idle envelope."""

    def __init__(self, p, policy, actor, endpoint, pin):
        super().__init__(p, policy, actor)
        self.endpoint, self.pin = endpoint, pin
        self.verifier = LoopbackVerifier(LOCAL_NODE_ID)

    def request_json(self, path, body):
        raw = remote_http._json_bytes(body)
        auth = self.signing_client._auth(path, hashlib.sha256(raw).hexdigest(),
                                         len(raw), "application/json")
        connection = tailnet_rpc.PinnedHTTPSConnection(self.endpoint, self.pin,
            self.verifier, controller.HOST_NODE, timeout=REQUEST_TIMEOUT)
        response = None
        try:
            connection._http_vsn, connection._http_vsn_str = 10, "HTTP/1.0"
            connection.request("POST", path, body=raw, headers={
                "Host": self.endpoint.removeprefix("https://"),
                "Content-Type": "application/json", "Content-Length": str(len(raw)),
                "Connection": "close", remote_http.AUTH_HEADER: auth})
            response = connection.getresponse()
            value = json.loads(response.read())
            assert response.version == 10 and response.status == 200, value
            return value
        finally:
            if response is not None:
                response.close()
            connection.close()


@pytest.fixture
def completed_project(project):  # noqa: F811 - use the imported pytest fixture
    p = project
    assert p.points == 4
    wake = p.started()
    p.bell.report(wake["attempt_id"])
    assert p.complete()["status"] == "RESOLVED"
    assert p.bell.resolved(wake["attempt_id"])
    policy = remote_http._CollectorPolicy(p.coordinator, p.bootstrap,
        SimpleNamespace(network=p.bootstrap["network"]), p.prepared.paths.upload_root, p.queue.handle)
    direct = DirectSignedClient(p, policy, MAID_ACTOR)
    config = RemoteSyntheticMaidConfig.from_mapping(p.runtime)
    worker = ShoreWorker(config.state_root / "control", p.clients[MAID_ACTOR],
        p.keys["authority"].public_key(), binding=p.runtime["binding"],
        registration=p.registrations[MAID_ACTOR], csv_sha256=p.runtime["csv_sha256"])
    maid_config = MaidRuntimeConfig(run_id="run", worker_id=MAID_ACTOR,
        princess_url=p.bootstrap["collector"]["endpoint"], api_token=p.prepared.api_token,
        csv_path=p.prepared.paths.worklist_csv, csv_sha256=p.runtime["csv_sha256"],
        project_path=p.tmp_path / "NEVER-OPEN.cst", output_root=Path(p.runtime["output_root"]),
        dry_run=True, protocol_mode=MODE, poll_seconds=0.001, heartbeat_seconds=1)
    maid = Maid(maid_config, client=direct, artifact_client=direct, control_worker=worker,
                session=SyntheticSession(), run_csv_row=synthetic_case)
    assert maid.run() == 0
    assert p.state.progress("run").completed == 4
    assert not worker.pending_assignments() and not worker.rpc.pending()
    p.tls_policy, p.maid_config, p.worker = policy, config, worker
    return p


@contextmanager
def _tls_pair(p, monkeypatch, *, handler=remote_http.RemoteCollectorHandler):
    allowed_ports = set()

    def connect(sock, address):
        _loopback_address(address[0])
        assert address[1] in allowed_ports and address[1] != 18765
        return _REAL_CONNECT(sock, address)

    # The shared setup forbids every socket. Open only these two temporary
    # listeners; subprocess/CST guards remain in force throughout the test.
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(tailnet_rpc, "tailnet_address", lambda address:
        _loopback_address(address) if ipaddress.ip_address(address).is_loopback
        else _TAILNET_ADDRESS(address))
    cert, key, pin = _certificate(p.tmp_path)
    delayed = DelayedAuthority(p.authority)
    authority_verifier = LoopbackVerifier(controller.HOST_NODE)
    authority_server = tailnet_rpc.create_server(delayed, "127.0.0.1", 0,
        cert, key, authority_verifier, {controller.HOST_NODE})
    collector_server = tailnet_rpc.create_tls_server(handler, "127.0.0.1", 0,
        cert, key, LoopbackVerifier(LOCAL_NODE_ID), {LOCAL_NODE_ID})
    collector_server.remote_policy = p.tls_policy
    allowed_ports.update((authority_server.server_port, collector_server.server_port))
    transport = tailnet_rpc.TailnetRpcTransport(
        f"https://127.0.0.1:{authority_server.server_port}", pin,
        authority_verifier, controller.HOST_NODE, timeout=REQUEST_TIMEOUT)
    old_transport = p.gate.session.transport
    p.gate.session.transport = transport
    clients = {actor: LoopbackCollectorClient(p, p.tls_policy, actor,
        f"https://127.0.0.1:{collector_server.server_port}", pin)
        for actor in (BELL_ACTOR, MAID_ACTOR)}
    try:
        with _running(authority_server), _running(collector_server):
            yield SimpleNamespace(authority=delayed, clients=clients, server=collector_server)
    finally:
        p.gate.session.transport = old_transport


def _bell_idle(p, tls):
    runner = BellRunner(p.bell, tls.clients[BELL_ACTOR],
        lambda *_: pytest.fail("Settled Bell must not relaunch"), ready_root=p.maid_config.state_root)
    assert runner.report_idle() == {"status": "RECORDED", "actor_id": BELL_ACTOR}


@pytest.mark.parametrize("project", [4], indirect=True)
def test_four_point_final_idle_succeeds_with_delayed_authority_tls(completed_project, monkeypatch):
    p = completed_project
    assert REQUEST_TIMEOUT == 5.0  # Never extend either connection deadline.
    with _tls_pair(p, monkeypatch) as tls:
        _bell_idle(p, tls)
        tls.authority.calls.clear()
        tls.authority.delay = 0.15
        started = time.monotonic()
        _persist_worker_idle(p.maid_config, p.worker, p.clients[MAID_ACTOR], tls.clients[MAID_ACTOR])
        assert time.monotonic() - started < REQUEST_TIMEOUT
        assert len(tls.authority.calls) == 8
        assert len(_read_idle_permits(p.maid_config)) == 4
        tls.authority.delay = 0
        assert p.queue.idle_pending() == 0
        assert p.state.progress("run").completed == 4


@pytest.mark.parametrize("project", [4], indirect=True)
@pytest.mark.parametrize("loss", ["deadline", "after_commit"])
def test_four_point_idle_deadline_loss_replays_original_proof_after_collector_restart(
        completed_project, monkeypatch, loss):
    p = completed_project
    assert REQUEST_TIMEOUT == 5.0
    completion = threading.Event()

    class ObservedHandler(remote_http.RemoteCollectorHandler):
        def _reply(self, status, value):
            maid_idle = value == {"status": "RECORDED", "actor_id": MAID_ACTOR}
            try:
                if maid_idle and loss == "after_commit" and not completion.is_set():
                    # The handler has committed all proof and idle files, but
                    # no response headers/body may reach this first caller.
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                super()._reply(status, value)
            finally:
                if maid_idle:
                    completion.set()

    with _tls_pair(p, monkeypatch, handler=ObservedHandler) as tls:
        _bell_idle(p, tls)
        tls.authority.calls.clear()
        # 0.70 x eight HTTPS legs exceeds the total 5 s deadline; the other
        # case loses a committed reply even with a healthy fast connection.
        tls.authority.delay = 0.70 if loss == "deadline" else 0.05
        prior_binding = dict(p.gate.binding)
        started = time.monotonic()
        with pytest.raises((OSError, http.client.HTTPException)):
            _persist_worker_idle(p.maid_config, p.worker, p.clients[MAID_ACTOR], tls.clients[MAID_ACTOR])
        elapsed = time.monotonic() - started
        if loss == "deadline":
            assert 4.4 < elapsed < 6.5
        else:
            assert elapsed < REQUEST_TIMEOUT
        idle_path = p.maid_config.state_root / "idle.json"
        original_idle = idle_path.read_bytes()
        # Socket closure is not an acknowledgement. Wait only for the server
        # thread's attempted reply before restarting its durable queue object.
        assert completion.wait(timeout=4), "Timed-out idle handler did not finish bounded validation"
        assert len(tls.authority.calls) == 8
        assert not p.worker.pending_assignments() and not p.worker.rpc.pending()
        assert p.gate.binding == prior_binding
        assert len(_read_idle_permits(p.maid_config)) == 4
        restarted = controller.CollectorWakeQueue(p.queue.root, p.gate, p.runtime)
        tls.server.remote_policy.wake_handler = restarted.handle
        original_calls = list(tls.authority.calls)
        started = time.monotonic()
        _send_idle(tls.clients[MAID_ACTOR], json.loads(original_idle))
        assert time.monotonic() - started < REQUEST_TIMEOUT
        assert tls.authority.calls == original_calls, "Verified terminal proof replay must not repeat slow RPC"
        assert idle_path.read_bytes() == original_idle
        assert restarted.idle_pending() == 0
        assert tls.authority.calls == original_calls
        assert p.state.progress("run").completed == 4
        with p.gate.journal.transaction() as db:
            assert db.execute("SELECT COUNT(*) FROM control_operations").fetchone()[0] == 5
        # A local terminal filename/summary alone is never trusted on replay.
        # Corrupt the signed proof and verify it fails closed without refetching
        # or inventing new authority work to replace the existing journal.
        with p.gate.journal.transaction() as db:
            descriptors = [json.loads(row[0]) for row in
                db.execute("SELECT descriptor FROM control_operations").fetchall()]
        descriptor = next(row for row in descriptors if row["worker_id"] == MAID_ACTOR)
        proof_path = restarted.root / "terminal" / (
            hashlib.sha256(descriptor["operation_id"].encode()).hexdigest() + ".json")
        proof = json.loads(proof_path.read_bytes())
        proof["response"]["payload"]["result"]["status"] = "AWAITING_PEER_RECEIPT"
        proof_path.write_bytes(remote_http._json_bytes(proof))
        with pytest.raises(ValueError):
            restarted.idle_pending()
        assert tls.authority.calls == original_calls
