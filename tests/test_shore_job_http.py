"""Offline job HTTP regression: generated test keys, no real identity/RPC/launch."""
from __future__ import annotations

import base64
import builtins
import copy
from datetime import datetime, timedelta, timezone
from email.message import Message
import hashlib
import io
import json
import socket
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("shore_control")
from shore_control.client import MemorySigner
from shore_control.crypto import canonical_json, sign_object, verify_object
from shore_control import tailnet_rpc

from test_shore_job_material import materials  # noqa: F401
from test_shore_remote_worker import remote_worker  # noqa: F401

from msabp_opt.simulation.distributed import control_job_auth as auth
from msabp_opt.simulation.distributed import control_job_http as job_http
from msabp_opt.simulation.distributed import control_remote_http as remote
from msabp_opt.simulation.distributed.control_remote_worker import BELL_ACTOR, LOCAL_NODE_ID, MAID_ACTOR
from msabp_opt.simulation.distributed.http_api import ApiError

_REAL_SOCKET = socket.socket
_REAL_CONNECT = socket.socket.connect


@pytest.fixture
def lab(materials, tmp_path):  # noqa: F811
    case = materials.make()
    saved = {"assignment": copy.deepcopy(case.assignment), "proof": None, "reads": 0}

    def read_assignment():
        saved["reads"] += 1
        return saved["assignment"]

    nonce_db = tmp_path / "job-http.sqlite"
    auth.initialize_job_nonce_db(nonce_db)
    service_args = {**case.kwargs, "read_assignment": read_assignment, "csv_data": case.csv,
                   "nonce_db": nonce_db, "read_release_proof": lambda: saved["proof"]}
    service = job_http.JobMaterialService(**service_args)
    bootstrap = json.loads(materials.bootstrap_data)
    network = bootstrap["network"]
    worker_network = {**network, "local_node_id": LOCAL_NODE_ID,
                      "local_address": network["nodes"][LOCAL_NODE_ID]["address"]}
    verifier = SimpleNamespace(network=worker_network)
    client_args = {**case.kwargs, "signer": MemorySigner(case.keys[BELL_ACTOR]), "verifier": verifier}
    client = job_http.JobMaterialClient(**client_args)
    context = {**{k: case.binding[k] for k in auth.EXPECTED_JOB_FIELDS}, "actor_id": BELL_ACTOR,
               "device_id": bootstrap["workers"][BELL_ACTOR]["device_id"], "node_id": LOCAL_NODE_ID}
    policy = SimpleNamespace(receipt_only=False,
        authenticate=lambda *_a, **_kw: pytest.fail("Job routes must not use run authentication"))
    server = SimpleNamespace(remote_policy=policy, job_service=service,
                             server_address=("100.64.0.1", 18767))
    return SimpleNamespace(**locals())


def wire(lab, *, body=b"{}", path=auth.OFFER_PATH, node=LOCAL_NODE_ID,
         headers=(), signed=None, version="HTTP/1.0", method="POST", before_body_read=None):
    if signed is None:
        signed = auth.sign_job_request(lab.client_args["signer"], lab.context, "POST",
                                       path if path in auth.JOB_PATHS else auth.OFFER_PATH, body)
    header = base64.b64encode(canonical_json(signed)).decode()
    fields = [("Host", "100.64.0.1:18767"), ("Content-Type", "application/json"),
              ("Content-Length", str(len(body))), (auth.AUTH_HEADER, header), *headers]
    raw = (f"{method} {path} {version}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in fields)
           + "\r\n").encode() + body

    class Reader(io.BytesIO):
        def read(self, size=-1):
            if before_body_read is not None:
                before_body_read()
            return super().read(size)

    class Socket:
        client_node_id = node
        output = b""

        def makefile(self, *_args):
            self.reader = Reader(raw)
            return self.reader

        def sendall(self, data):
            self.output += bytes(data)

    sock = Socket()
    remote.RemoteCollectorHandler(sock, ("127.0.0.1", 10000), lab.server)
    head, result = sock.output.split(b"\r\n\r\n", 1)
    status = int(head.split(b" ")[1])
    assert head.startswith(b"HTTP/1.0") and b"Connection: close" in head
    return status, result


def offer(lab):
    status, raw = wire(lab)
    assert status == 200, raw
    return json.loads(raw)


def selector(value):
    return canonical_json({k: value[k] for k in job_http.MATERIAL_REQUEST_FIELDS})


class Response:
    def __init__(self, body=b"{}", *, status=200, version=10, extra_headers=()):
        self.body, self.status, self.version, self.closed = body, status, version, False
        self.headers = Message()
        for key, value in [("Content-Type", "application/json"), ("Content-Length", str(len(body))),
                           ("Connection", "close"), *extra_headers]:
            self.headers[key] = value

    def read(self, _size):
        return self.body

    def close(self):
        self.closed = True


def client_connection(lab, monkeypatch, response=None):
    connections = []

    class Connection:
        def __init__(self, *args):
            self.args, self.closed = args, False
            connections.append(self)

        def request(self, method, path, **kwargs):
            self.method, self.path, self.kwargs = method, path, kwargs

        def getresponse(self):
            if response is not None:
                self.response = response
            else:
                signed = json.loads(base64.b64decode(self.kwargs["headers"][auth.AUTH_HEADER]))
                status, raw = wire(lab, path=self.path, body=self.kwargs["body"], signed=signed)
                self.response = Response(raw, status=status)
            return self.response

        def close(self):
            self.closed = True

    monkeypatch.setattr(tailnet_rpc, "PinnedHTTPSConnection", Connection)
    return connections


def release_proof(lab):
    counters = {name: 0 for name in ("pending_count", "uploads_in_progress", "worker_outboxes_pending",
                "receipts_pending", "local_unresolved_count", "gpu_uncollected")}
    checkpoint = {**counters, "manifest_sha256": "1" * 64, "ledger_sha256": "2" * 64}
    checkpoint["checkpoint_sha256"] = hashlib.sha256(canonical_json(checkpoint)).hexdigest()
    params = {**{k: lab.case.binding[k] for k in ("job_id", "epoch", "host_id", "session_id")},
              "checkpoint_sha256": checkpoint["checkpoint_sha256"],
              "disposition": "completed", "reason": "offline signed fixture"}
    request = lab.case.controller.prepare("release_job", params)
    result = {**lab.case.job, "phase": "RELEASED", "lease_until_ms": 0,
              "checkpoint": checkpoint, "disposition": "completed"}
    payload = request["payload"]
    response = sign_object(lab.case.keys["authority"], "response", {
        "version": 1, "authority_id": lab.bootstrap["authority_id"],
        "request_id": payload["request_id"], "actor_id": payload["actor_id"],
        "request_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "method": "release_job", "ok": True, "result": result})
    return {"request": request, "response": response}


@pytest.mark.parametrize("points", [1, 4])
def test_original_one_four_point_offer_material_roundtrip(lab, materials, points):  # noqa: F811
    if points == 1:
        case = materials.make(1)
        lab.case = case
        lab.saved["assignment"] = case.assignment
        lab.context.update({k: case.binding[k] for k in auth.EXPECTED_JOB_FIELDS})
        lab.server.job_service = job_http.JobMaterialService(**{
            **lab.service_args, **case.kwargs, "csv_data": case.csv})
    current = offer(lab)
    assert set(current) == job_http.OFFER_FIELDS
    assert current["terminal_proof"] is None
    assert current["permit_id"] == lab.case.permit["payload"]["permit_id"]
    assert current["permit_sha256"] == hashlib.sha256(canonical_json(lab.case.permit)).hexdigest()
    status, raw = wire(lab, path=auth.MATERIAL_PATH, body=selector(current))
    assert status == 200 and raw == lab.case.raw
    assert current["material_sha256"] == hashlib.sha256(raw).hexdigest()
    assert current["material_bytes"] == len(raw)


def test_unissued_does_not_create_assignment_permit_or_call_authority(lab, monkeypatch):
    lab.saved["assignment"] = None
    monkeypatch.setattr(lab.case.authority, "handle", lambda *_a: pytest.fail("No authority on HTTP read"))
    assert wire(lab)[0] == 404
    assert lab.saved["assignment"] is None
    lab.saved["assignment"] = copy.deepcopy(lab.case.assignment)
    assert offer(lab)["permit_id"] == lab.case.permit["payload"]["permit_id"]


def test_drain_response_loss_restart_reuses_exact_original_bytes(lab, monkeypatch):
    current = offer(lab)
    bound = {k: lab.case.binding[k] for k in ("job_id", "epoch", "host_id", "session_id")}
    lab.case.controller.call("begin_drain", **bound, reason="test F missing")
    monkeypatch.setattr(lab.case.authority, "handle", lambda *_a: pytest.fail("No authority on HTTP read"))
    signed = auth.sign_job_request(lab.client_args["signer"], lab.context,
                                  "POST", auth.MATERIAL_PATH, selector(current))
    assert wire(lab, path=auth.MATERIAL_PATH, body=selector(current), signed=signed)[1] == lab.case.raw
    lab.server.job_service = job_http.JobMaterialService(**lab.service_args)
    assert wire(lab, path=auth.MATERIAL_PATH, body=selector(current), signed=signed)[0] == 409
    assert wire(lab, path=auth.MATERIAL_PATH, body=selector(current))[1] == lab.case.raw


@pytest.mark.parametrize("change", ["missing", "token", "permit", "binding"])
def test_original_material_never_changes_after_first_offer(lab, change):
    current = offer(lab)
    if change == "missing":
        lab.saved["assignment"] = None
    elif change == "token":
        lab.saved["assignment"]["runtime"]["api_token"] = "changed"
    elif change == "permit":
        lab.saved["assignment"]["control_authority"]["permit"]["payload"]["permit_id"] = "changed"
    else:
        lab.saved["assignment"]["runtime"]["binding"]["session_id"] = "new-session"
    assert wire(lab, path=auth.MATERIAL_PATH, body=selector(current))[0] == 503


@pytest.mark.parametrize("key", sorted(job_http.MATERIAL_REQUEST_FIELDS))
def test_selector_must_name_exact_offer(lab, key):
    current = offer(lab)
    current[key] = "a" * 64
    assert wire(lab, path=auth.MATERIAL_PATH, body=selector(current))[0] == 409


@pytest.mark.parametrize("node", [None, "n8BsWasrMb11CNTRL", "forged"])
def test_actual_socket_source_cannot_be_overridden(lab, node):
    assert wire(lab, node=node, headers=[("X-Client-Node-ID", LOCAL_NODE_ID)])[0] == 403
    assert lab.saved["reads"] == 0


def test_maid_run_auth_and_unauthenticated_requests_never_read_assignment(lab):
    signed = auth.sign_job_request(MemorySigner(lab.case.keys[MAID_ACTOR]),
        {**lab.context, "actor_id": MAID_ACTOR}, "POST", auth.OFFER_PATH, b"{}")
    assert wire(lab, signed=signed)[0] == 403
    payload = signed["payload"]
    signed = sign_object(lab.case.keys[BELL_ACTOR], remote.AUTH_DOMAIN, payload)
    assert wire(lab, signed=signed)[0] == 403
    assert lab.saved["reads"] == 0


@pytest.mark.parametrize("change", ["disabled", "receipt-only", "explicit-recovery"])
def test_opt_in_and_receipt_only_boundaries(lab, change):
    if change == "disabled":
        del lab.server.job_service
    else:
        lab.server.remote_policy.receipt_only = True
        lab.server.allow_job_reads = change == "explicit-recovery"
    assert wire(lab)[0] == {"disabled": 404, "receipt-only": 403, "explicit-recovery": 200}[change]


@pytest.mark.parametrize("path", ["/v1/wake/poll", auth.OFFER_PATH])
def test_legacy_run_routes_and_disabled_job_routes_do_not_import_new_job_modules(lab, monkeypatch, path):
    real_import = builtins.__import__

    def legacy_import(name, *args, **kwargs):
        if name.endswith(("control_job_auth", "control_job_http")):
            raise AssertionError("Legacy release does not contain new job modules")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", legacy_import)
    del lab.server.job_service
    lab.policy.authenticate = lambda *_a, **_k: {
        "body_sha256": hashlib.sha256(b"{}").hexdigest(), "actor_id": BELL_ACTOR}
    lab.policy.consume = lambda _auth: None
    lab.policy.wake_handler = lambda *_a, **_k: {"assignment": None}
    assert wire(lab, path=path)[0] == (200 if path == "/v1/wake/poll" else 404)


@pytest.mark.parametrize("body", [b'{"extra":1}', b'{"x":1,"x":2}', b'[]', b'{"x":NaN}', b' {}'])
def test_strict_offer_body_schema(lab, body):
    assert wire(lab, body=body)[0] == 400
    assert lab.saved["reads"] == 0


@pytest.mark.parametrize("header", [("Content-Length", "2"), ("Transfer-Encoding", "chunked"),
    ("Content-Encoding", "gzip"), ("Content-Type", "application/json"), ("Host", "100.64.0.1:18767"),
    ("Origin", "https://example.test"), ("Expect", "100-continue"), (auth.AUTH_HEADER, ""),
    (auth.AUTH_HEADER, "A" * 12000)])
def test_strict_request_framing_and_auth_header(lab, header):
    assert wire(lab, headers=[header])[0] in {400, 403}
    assert lab.saved["reads"] == 0


@pytest.mark.parametrize("path", [auth.OFFER_PATH + "?x=1", auth.OFFER_PATH + "/", "/v1/job/%6ffer"])
def test_exact_fixed_path(lab, path):
    assert wire(lab, path=path)[0] == 404
    assert lab.saved["reads"] == 0


def test_job_request_size_http_version_and_expiry_before_business(lab, monkeypatch):
    assert wire(lab, version="HTTP/1.1")[0] == 400
    # Sign a bounded body, then send a larger unsigned body: framing must reject
    # before reading/parsing it, but the inherited rejection drain stays bounded.
    signed = auth.sign_job_request(lab.client_args["signer"], lab.context, "POST", auth.OFFER_PATH, b"{}")
    assert wire(lab, body=b" " * (auth.MAX_REQUEST_BYTES + 1), signed=signed)[0] == 413
    now = signed["payload"]["issued_at_ms"]
    monkeypatch.setattr(auth, "_wall_clock_ms", lambda: now)
    assert wire(lab, signed=signed, before_body_read=lambda: monkeypatch.setattr(
        auth, "_wall_clock_ms", lambda: now + auth.MAX_AGE_MS + 1))[0] == 403
    assert lab.saved["reads"] == 0


def test_pinned_client_roundtrip_exact_bytes_fresh_nonces_and_no_token(lab, monkeypatch):
    connections = client_connection(lab, monkeypatch)
    current = lab.client.offer()
    assert lab.client.material(current) == lab.case.raw
    assert lab.client.material(current) == lab.case.raw
    requests = []
    for connection in connections:
        assert connection.closed and connection.response.closed
        assert connection.args[:2] == (lab.bootstrap["collector"]["endpoint"], "b" * 64)
        assert connection.args[3] == lab.bootstrap["network"]["local_node_id"]
        assert connection._http_vsn_str == "HTTP/1.0"
        headers = connection.kwargs["headers"]
        assert "Authorization" not in headers and remote.AUTH_HEADER not in headers
        envelope = json.loads(base64.b64decode(headers[auth.AUTH_HEADER]))
        payload = verify_object(lab.bootstrap["workers"][BELL_ACTOR]["public_key"], auth.AUTH_DOMAIN, envelope)
        assert "run_id" not in payload
        requests.append(payload["request_id"])
    assert len(set(requests)) == 3


@pytest.mark.parametrize("kind", ["redirect", "version", "duplicate-length", "gzip", "chunked",
    "not-closed", "duplicate-json", "whitespace", "oversize", "truncated", "schema", "float-epoch"])
def test_client_rejects_malformed_bounded_responses_and_closes(lab, monkeypatch, kind):
    current = offer(lab)
    response = Response(canonical_json(current))
    if kind == "redirect":
        response.status = 302
    elif kind == "version":
        response.version = 11
    elif kind in {"duplicate-length", "gzip", "chunked"}:
        header = {"duplicate-length": ("Content-Length", "2"), "gzip": ("Content-Encoding", "gzip"),
                  "chunked": ("Transfer-Encoding", "chunked")}[kind]
        response.headers[header[0]] = header[1]
    elif kind == "not-closed":
        response.headers.replace_header("Connection", "keep-alive")
    elif kind == "truncated":
        response.headers.replace_header("Content-Length", str(len(response.body) + 1))
    else:
        body = {"duplicate-json": b'{"x":1,"x":2}', "whitespace": b" " + response.body,
                "oversize": b" " * (job_http.MAX_OFFER_BYTES + 1), "schema": b'{}',
                "float-epoch": response.body.replace(b'"epoch":1', b'"epoch":1.0')}[kind]
        response = Response(body)
    connections = client_connection(lab, monkeypatch, response)
    with pytest.raises((ValueError, ConnectionError, ApiError)):
        lab.client.offer()
    assert response.closed and connections[-1].closed


@pytest.mark.parametrize("key", ["material_sha256", "material_bytes", "runtime_sha256", "permit_id", "permit_sha256"])
def test_client_checks_offer_exact_original_material_permit_runtime(lab, monkeypatch, key):
    current = offer(lab)
    current[key] = current[key] + 1 if key == "material_bytes" else "a" * 64
    connections = client_connection(lab, monkeypatch, Response(lab.case.raw))
    with pytest.raises(ValueError):
        lab.client.material(current)
    assert connections[-1].closed


def test_client_rejects_wrong_signer_verifier_profile_before_connecting(lab, monkeypatch):
    connections = client_connection(lab, monkeypatch)
    client = job_http.JobMaterialClient(**{**lab.client_args, "signer": MemorySigner(lab.case.keys[MAID_ACTOR])})
    with pytest.raises(ValueError, match="signature"):
        client.offer()
    assert len(connections) == 1 and connections[0].closed  # configuration-only construction
    with pytest.raises(ValueError, match="Verifier"):
        job_http.JobMaterialClient(**{**lab.client_args, "verifier": SimpleNamespace(network=lab.bootstrap["network"])})
    profile = {**lab.case.kwargs["profile"], "release_id": "a" * 64}
    with pytest.raises(ValueError):
        job_http.JobMaterialClient(**{**lab.client_args, "profile": profile})


def test_missing_nonce_ledger_no_reinitialization_or_assignment_reads(lab):
    lab.server.job_service = job_http.JobMaterialService(**{
        **lab.service_args, "nonce_db": lab.nonce_db.with_name("not-installed.sqlite")})
    assert wire(lab)[0] == 503
    assert not lab.nonce_db.with_name("not-installed.sqlite").exists()
    assert lab.saved["reads"] == 0


def test_release_proof_appears_without_changing_material_and_is_verified(lab, monkeypatch):
    old = offer(lab)
    proof = release_proof(lab)
    lab.saved["proof"] = proof
    new = offer(lab)
    assert new == {**old, "terminal_proof": proof}
    assert job_http.verify_release_proof(proof, lab.bootstrap, lab.case.binding)["phase"] == "RELEASED"
    client_connection(lab, monkeypatch)
    assert lab.client.offer() == new
    assert lab.client.material(old) == lab.client.material(new) == lab.case.raw
    lab.saved["proof"] = None
    assert wire(lab)[0] == 503


@pytest.mark.parametrize("change", ["controller-signature", "authority-signature", "response-request-hash",
    "response-request-id", "response-method", "request-session", "request-job", "result-session",
    "result-phase", "result-epoch-bool", "result-scope", "checkpoint-unresolved", "checkpoint-hash",
    "error-response", "unknown-field"])
def test_even_signed_release_proof_must_match_original_binding_and_settlement(lab, change):
    proof = release_proof(lab)
    request, response = proof["request"]["payload"], proof["response"]["payload"]
    if change == "controller-signature":
        proof["request"]["signature"] = "A" * 86 + "=="
    elif change == "authority-signature":
        proof["response"]["signature"] = "A" * 86 + "=="
    elif change.startswith("response-"):
        key = {"response-request-hash": "request_sha256", "response-request-id": "request_id",
               "response-method": "method"}[change]
        response[key] = "changed"
    elif change.startswith("request-"):
        request["params"]["session_id" if change == "request-session" else "job_id"] = "changed"
        proof["request"] = sign_object(lab.case.keys["controller"], "request", request)
        response["request_sha256"] = hashlib.sha256(canonical_json(request)).hexdigest()
    elif change == "result-session":
        response["result"]["session_id"] = "new-session"
    elif change == "result-phase":
        response["result"]["phase"] = "ACTIVE"
    elif change == "result-epoch-bool":
        response["result"]["epoch"] = True
    elif change == "result-scope":
        response["result"]["scope"]["collector_url"] = "https://100.64.0.99:18767"
    elif change == "checkpoint-unresolved":
        response["result"]["checkpoint"]["worker_outboxes_pending"] = 1
    elif change == "checkpoint-hash":
        response["result"]["checkpoint"]["ledger_sha256"] = "3" * 64
    elif change == "error-response":
        response["ok"] = False
    else:
        response["extra"] = True
    if change != "authority-signature":
        proof["response"] = sign_object(lab.case.keys["authority"], "response", response)
    with pytest.raises(ValueError):
        job_http.verify_release_proof(proof, lab.bootstrap, lab.case.binding)
    lab.saved["proof"] = proof
    assert wire(lab)[0] == 503


def test_service_bootstrap_and_inputs_are_copied_and_pinned(lab):
    lab.service.assert_bootstrap(lab.bootstrap)
    other = copy.deepcopy(lab.bootstrap)
    other["collector"]["tls_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        lab.service.assert_bootstrap(other)
    with pytest.raises(ValueError):
        job_http.JobMaterialService(**{**lab.service_args, "csv_data": b"arbitrary,csv\n"})
    with pytest.raises(ValueError):
        job_http.JobMaterialService(**{**lab.service_args, "read_assignment": None})


def test_real_loopback_tls_uses_socket_source_pinned_certificate_and_original_bytes(lab, monkeypatch, tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    from shore_control.authority import AuthorityError

    # The material fixture forbids sockets. Enable only this ephemeral loopback
    # listener/client; all existing subprocess/launch guards remain in force.
    monkeypatch.setattr(socket, "socket", _REAL_SOCKET)
    ports = set()

    def connect(sock, address):
        assert address[0] == "127.0.0.1" and address[1] in ports
        return _REAL_CONNECT(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    real_address = tailnet_rpc.tailnet_address
    monkeypatch.setattr(tailnet_rpc, "tailnet_address",
        lambda address: address if address == "127.0.0.1" else real_address(address))
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "job-http-loopback-only")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
        .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "loopback.pem", tmp_path / "loopback-key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    peer = lab.bootstrap["network"]["local_node_id"]
    network = {"local_address": "127.0.0.1", "local_node_id": peer,
               "nodes": {peer: {"address": "127.0.0.1"}, LOCAL_NODE_ID: {"address": "127.0.0.1"}}}
    observed = []

    def verify_peer(address, port, expected_node_id=None):
        assert address == "127.0.0.1" and 0 < port <= 65535
        observed.append((address, port, expected_node_id))
        return expected_node_id or LOCAL_NODE_ID

    verifier = SimpleNamespace(network=network, verify_local=lambda: {}, verify_peer=verify_peer)
    server = tailnet_rpc.create_tls_server(remote.RemoteCollectorHandler, "127.0.0.1", 0,
        str(cert_path), str(key_path), verifier, {LOCAL_NODE_ID})
    server.remote_policy, server.job_service = lab.policy, lab.service
    ports.add(server.server_port)
    # Only test address/source resolution changes. Production connection, TLS
    # pin enforcement, signed HTTP requests, handler and service are untouched.
    lab.client._endpoint = f"https://127.0.0.1:{server.server_port}"
    lab.client._pin = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    lab.client._verifier = verifier
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        current = lab.client.offer()
        assert lab.client.material(current) == lab.case.raw
        assert any(expected is None for _, _, expected in observed), "Server did not verify actual socket source"
        assert any(expected == peer for _, _, expected in observed), "Client did not pin target node"
        before = lab.saved["reads"]
        lab.client._pin = "a" * 64
        with pytest.raises(ConnectionError) as rejected:
            lab.client.offer()
        assert isinstance(rejected.value.__cause__, AuthorityError)
        assert rejected.value.__cause__.code == "TLS_PIN"
        assert lab.saved["reads"] == before, "Wrong certificate pin emitted application data"
        lab.client._pin = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
        verifier.verify_peer = lambda _address, _port, expected_node_id=None: expected_node_id or "wrong-source"
        with pytest.raises(ConnectionError):
            lab.client.offer()
        assert lab.saved["reads"] == before, "Wrong socket source reached the material service"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()
