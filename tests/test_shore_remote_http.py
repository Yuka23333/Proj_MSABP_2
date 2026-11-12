"""Offline signed collector contract tests: temporary keys, mocked sockets only."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("shore_control")
pytest.importorskip("cryptography")

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID
from shore_control.client import MemorySigner
from shore_control.crypto import canonical_json, fingerprint, public_key_text, sign_object
from shore_control.registry import registration_sha256
import shore_control.tailnet_rpc as tailnet_rpc

from msabp_opt.simulation.distributed import control_remote_http as remote
from msabp_opt.simulation.distributed.http_api import ApiError, ARTIFACT_PATH_PREFIX, MESSAGE_PATH
from msabp_opt.simulation.distributed.protocol import make_message


@pytest.fixture
def lab(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Offline HTTP tests must never use real sockets/processes")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    keys = {actor: Ed25519PrivateKey.generate() for actor in ("authority", "controller", "maid", "bell")}
    network = {"tailnet_name": "test@example.test", "magic_dns_suffix": "example.ts.net",
        "local_node_id": "nController", "local_address": "100.64.0.1", "nodes": {
            "nController": {"address": "100.64.0.1", "dns_name": "collector.example.ts.net."},
            "nWorker": {"address": "100.64.0.2", "dns_name": "worker.example.ts.net."}}}
    workers = {actor: {"public_key": public_key_text(keys[actor].public_key()),
        "device_id": "synthetic-device", "tailnet_node_id": "nWorker", "role": actor,
        "allowed_actions": ["attempt" if actor == "maid" else "wake"]} for actor in ("maid", "bell")}
    bootstrap = {"schema": "shore.remote-bootstrap", "schema_version": 1, "mode": remote.MODE,
        "fleet_id": "shore-princess-" + "1" * 24, "authority_id": fingerprint(keys["authority"].public_key()),
        "authority_public_key": public_key_text(keys["authority"].public_key()),
        "endpoint": "https://100.64.0.1:18766", "tls_sha256": "a" * 64, "network": network,
        "controllers": {"princess-281ea981-06cf-4cfb-ac14-4c86a736b471": public_key_text(keys["controller"].public_key())},
        "workers": workers, "controller_nodes": ["nController", "nWorker"],
        "collector": {"endpoint": "https://100.64.0.1:18767", "tls_sha256": "b" * 64}}
    verifier = SimpleNamespace(network=network)
    worker_verifier = SimpleNamespace(network={**network, "local_node_id": "nWorker", "local_address": "100.64.0.2"})
    database = tmp_path / "state.sqlite"

    @contextmanager
    def transaction():
        db = sqlite3.connect(database)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    with transaction() as db:
        db.executescript("""CREATE TABLE control_runs(run_id TEXT,registration TEXT);
            CREATE TABLE tasks(run_id TEXT,case_id TEXT,row_sha256 TEXT);
            CREATE TABLE attempts(attempt_id TEXT,run_id TEXT,worker_id TEXT,case_id TEXT);""")
        db.execute("INSERT INTO control_runs VALUES(?,?)", ("run", json.dumps({"input_manifest_sha256": "c" * 64})))
        db.execute("INSERT INTO tasks VALUES('run','case',?)", ("d" * 64,))
        db.execute("INSERT INTO attempts VALUES('attempt','run','maid','case')")
    binding = {"job_id": "job", "epoch": 1, "host_id": "nController", "session_id": "session-new"}
    scope = {"project_id": "project", "collector_url": bootstrap["collector"]["endpoint"],
        "data_root": str(tmp_path), "job_spec": {"schema": "shore.job-spec", "schema_version": 1,
        "manifest_sha256": "a" * 64, "approved_units": 4, "budget_unit": "objective_evaluation",
        "accounting_policy_sha256": "b" * 64}}
    gate = SimpleNamespace(controller=SimpleNamespace(authority_id=bootstrap["authority_id"], fleet_id=bootstrap["fleet_id"]),
        binding=binding, job={"scope": scope}, registrations=workers, journal=SimpleNamespace(transaction=transaction))
    descriptor = {**binding, **scope, "session_id": "session-old", "fleet_id": bootstrap["fleet_id"],
        "operation_id": "op", "run_id": "run", "worker_id": "maid", "attempt_id": "attempt",
        "action": "attempt", "input_sha256": "d" * 64, "device_id": workers["maid"]["device_id"],
        "worker_role": "maid", "worker_registration_sha256": registration_sha256(workers["maid"]),
        "action_context": None}
    signed = sign_object(keys["authority"], "permit", {**descriptor, "version": 1,
        "authority_id": bootstrap["authority_id"], "permit_id": "permit", "issued_at_ms": 1})
    saved = {"descriptor": descriptor, "permit": signed}
    calls, uploads, ends, wakes = [], [], [], []
    state = SimpleNamespace(path=database, _authority_for_run=lambda _: gate,
                            control_for_attempt=lambda attempt: saved if attempt == "attempt" else None)
    coordinator = SimpleNamespace(state=state, controlled=True, control_collector=object(), run_id="run",
        csv_sha256="c" * 64, handle_message=lambda value: calls.append(value) or {"type": "ack", "elapsed_seconds": 0.25},
        begin_artifact_upload=lambda *args: uploads.append(args) or "ticket",
        end_artifact_upload=ends.append,
        handle_artifact=lambda *args: {"sha256": args[-2], "size": args[-1], "status": "staged"})
    def wake(path, body, **kwargs):
        wakes.append((path, body, kwargs))
        return {"assignment": None}

    policy = remote._CollectorPolicy(coordinator, bootstrap, verifier, tmp_path / "uploads", wake)
    server = SimpleNamespace(remote_policy=policy, server_address=("100.64.0.1", 18767))
    client = remote.RemotePrincessClient(bootstrap, "maid", MemorySigner(keys["maid"]), worker_verifier, run_id="run")
    bell = remote.RemotePrincessClient(bootstrap, "bell", MemorySigner(keys["bell"]), worker_verifier, run_id="run")
    return SimpleNamespace(**locals())


def message(kind="hello", **payload):
    return make_message(kind, run_id="run", worker_id="maid", protocol_version=2, payload=payload)


def wire(lab, body=None, *, path=MESSAGE_PATH, actor="maid", headers=(), node="nWorker", auth=None,
         before_body_read=None):
    if body is None:
        body = remote._json_bytes(message(elapsed_seconds=1.25))
    kind = "application/zip" if path.startswith(ARTIFACT_PATH_PREFIX) else "application/json"
    client = lab.client if actor == "maid" else lab.bell
    if auth is None:
        auth = client._auth(path, hashlib.sha256(body).hexdigest(), len(body), kind,
            attempt_id="attempt" if kind == "application/zip" else "",
            lease_token="lease" if kind == "application/zip" else "")
    fields = [("Host", "100.64.0.1:18767"), ("Content-Type", kind),
              ("Content-Length", str(len(body))), (remote.AUTH_HEADER, auth), *headers]
    raw = (f"POST {path} HTTP/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in fields) + "\r\n").encode() + body

    class MockReader(io.BytesIO):
        def read(self, size=-1):
            if before_body_read is not None:
                before_body_read()
            return super().read(size)

    class MockSocket:
        client_node_id = node
        output = b""

        def makefile(self, *_args):
            return MockReader(raw)

        def sendall(self, data):
            self.output += bytes(data)

    sock = MockSocket()
    remote.RemoteCollectorHandler(sock, ("127.0.0.1", 10000), lab.server)
    status = int(sock.output.split(b" ")[1])
    header, result = sock.output.split(b"\r\n\r\n", 1)
    assert b"Connection: close" in header
    assert sock.output.startswith(b"HTTP/1.0")
    return status, json.loads(result)


def test_actor_signed_finite_float_message_and_old_session_attempt(lab):
    assert wire(lab)[0] == 200
    assert lab.calls[0]["payload"]["elapsed_seconds"] == 1.25
    assert wire(lab, remote._json_bytes(message("heartbeat", attempt_id="attempt", lease_token="lease")))[0] == 200


@pytest.mark.parametrize("node", [None, "nController", "attacker"])
def test_actual_source_cannot_be_overridden_by_header(lab, node):
    assert wire(lab, node=node, headers=[("X-Client-Node-ID", "nWorker")])[0] == 403
    assert not lab.calls


def test_same_node_bell_and_maid_cannot_replace_each_other(lab):
    assert wire(lab, actor="bell")[0] == 403
    assert wire(lab, b"{}", path="/v1/wake/poll")[0] == 403
    assert wire(lab, b"{}", path="/v1/wake/poll", actor="bell")[0] == 200
    assert lab.wakes == [("/v1/wake/poll", {}, {"actor_id": "bell", "node_id": "nWorker"})]


@pytest.mark.parametrize("actor", ["bell", "maid"])
def test_idle_callback_preserves_each_signed_actor_and_actual_source(lab, actor):
    body = {"record": {"payload": {"schema": "synthetic-test"}, "signature": "callback-verifies-this"}}
    assert wire(lab, remote._json_bytes(body), path=remote.IDLE_PATH, actor=actor)[0] == 200
    assert lab.wakes == [(remote.IDLE_PATH, body, {"actor_id": actor, "node_id": "nWorker"})]
    assert wire(lab, remote._json_bytes(body), path=remote.IDLE_PATH, actor=actor, node="nController")[0] == 403


@pytest.mark.parametrize("actor", ["bell", "maid"])
def test_idle_client_allows_only_explicit_idle_callback(lab, monkeypatch, actor):
    client_connection(monkeypatch, MockResponse())
    client = lab.bell if actor == "bell" else lab.client
    assert client.request_json(remote.IDLE_PATH, {"record": {}}) == {"ok": True}
    with pytest.raises(ValueError):
        client.request_json("/v1/control/arbitrary", {})


@pytest.mark.parametrize("headers", [[("Content-Length", "1")], [("Content-Encoding", "gzip")],
    [("Content-Encoding", "identity")], [("Transfer-Encoding", "chunked")],
    [("Content-Type", "application/json")], [("Host", "100.64.0.1:18767")], [("Expect", "100-continue")]])
def test_reject_ambiguous_or_encoded_request(lab, headers):
    assert wire(lab, headers=headers)[0] == 400
    assert not lab.calls


@pytest.mark.parametrize("body", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]'])
def test_reject_duplicate_or_nonfinite_json(lab, body):
    assert wire(lab, body)[0] == 400


def test_signature_covers_body_path_actor_run_and_replay(lab):
    body = remote._json_bytes(message())
    auth = lab.client._auth(MESSAGE_PATH, hashlib.sha256(body).hexdigest(), len(body), "application/json")
    assert wire(lab, body, auth=auth)[0] == 200
    assert wire(lab, body, auth=auth)[0] == 409
    # Persistent transport nonces survive collector reconstruction.
    lab.server.remote_policy = remote._CollectorPolicy(lab.coordinator, lab.bootstrap, lab.verifier,
                                                       lab.tmp_path / "uploads", lab.wake)
    assert wire(lab, body, auth=auth)[0] == 409
    envelope = json.loads(base64.b64decode(auth))
    envelope["payload"]["run_id"] = "other"
    forged = base64.b64encode(canonical_json(envelope)).decode()
    assert wire(lab, body, auth=forged)[0] == 403
    assert wire(lab, body.replace(b'"hello"', b'"HELLO"'), auth=auth)[0] == 400
    assert wire(lab, body, auth=auth, path="/v1/wake/poll")[0] == 403
    assert len(lab.calls) == 1


def test_replay_expiring_during_body_read_never_reaches_business(lab, monkeypatch):
    clock = {"now": 1000}
    monkeypatch.setattr(remote.time, "time", lambda: clock["now"])
    body = remote._json_bytes(message())
    auth = lab.client._auth(MESSAGE_PATH, hashlib.sha256(body).hexdigest(), len(body), "application/json")
    assert wire(lab, body, auth=auth)[0] == 200
    clock["now"] = 1119

    # Header authentication succeeds within the window. A two-second body
    # delay still fits the five-second connection limit, but expires the nonce.
    status, _ = wire(lab, body, auth=auth,
                     before_body_read=lambda: clock.update(now=1121))
    assert status == 403
    assert len(lab.calls) == 1
    with lab.transaction() as db:
        assert db.execute("SELECT issued_at FROM remote_http_nonces").fetchall()[0][0] == 1000


def test_replay_expiring_while_waiting_for_write_lock_never_reaches_business(lab, monkeypatch):
    clock = {"now": 1000}
    monkeypatch.setattr(remote.time, "time", lambda: clock["now"])
    body = remote._json_bytes(message())
    auth = lab.client._auth(MESSAGE_PATH, hashlib.sha256(body).hexdigest(), len(body), "application/json")
    assert wire(lab, body, auth=auth)[0] == 200
    clock["now"] = 1119
    entered, finished = threading.Event(), threading.Event()
    original_connect = sqlite3.connect
    blocker = original_connect(lab.database)
    blocker.execute("BEGIN IMMEDIATE")

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "BEGIN IMMEDIATE":
                entered.set()
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(remote.sqlite3, "connect",
                        lambda *args, **kwargs: original_connect(*args, **kwargs, factory=ObservedConnection))
    outcome = {}

    def request():
        try:
            outcome["response"] = wire(lab, body, auth=auth)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    try:
        assert entered.wait(2), "Request did not reach the nonce write transaction"
        assert not finished.wait(0.05), "Request must wait for the held SQLite write lock"
        clock["now"] = 1121
    finally:
        blocker.rollback()
        blocker.close()
        thread.join(5)
    assert not thread.is_alive()
    assert "error" not in outcome, repr(outcome.get("error"))
    assert outcome["response"][0] == 403
    assert len(lab.calls) == 1
    with lab.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM remote_http_nonces").fetchone()[0] == 1


def test_request_becoming_too_far_in_future_during_body_read_is_rejected(lab, monkeypatch):
    clock = {"now": 1000}
    monkeypatch.setattr(remote.time, "time", lambda: clock["now"])
    body = remote._json_bytes(message())
    auth = lab.client._auth(MESSAGE_PATH, hashlib.sha256(body).hexdigest(), len(body), "application/json")
    status, _ = wire(lab, body, auth=auth,
                     before_body_read=lambda: clock.update(now=994))
    assert status == 403
    assert not lab.calls
    with lab.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM remote_http_nonces").fetchone()[0] == 0


def test_unregistered_attempt_and_changed_frozen_input_rejected(lab):
    raw = remote._json_bytes(message("heartbeat", attempt_id="unknown", lease_token="lease"))
    assert wire(lab, raw)[0] == 403
    with lab.transaction() as db:
        db.execute("UPDATE tasks SET row_sha256=?", ("e" * 64,))
    raw = remote._json_bytes(message("heartbeat", attempt_id="attempt", lease_token="lease"))
    assert wire(lab, raw)[0] == 403
    assert not lab.calls


def test_token_alone_never_authenticates(lab):
    assert wire(lab, auth="", headers=[("Authorization", "Bearer shared-token")])[0] == 403
    assert not lab.calls


def archive_bytes(content=b"synthetic", compression=zipfile.ZIP_STORED):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        archive.writestr("synthetic.txt", content)
    return stream.getvalue()


def test_streamed_artifact_hash_cleanup_and_upload_journal(lab):
    data = archive_bytes()
    status, result = wire(lab, data, path=ARTIFACT_PATH_PREFIX + "attempt")
    assert status == 200
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert lab.uploads == [("attempt", "maid", "lease")]
    assert lab.ends == ["ticket"]
    assert not list((lab.tmp_path / "uploads").iterdir())


def test_upload_mismatch_and_expansion_are_bounded_and_cleaned(lab):
    data = archive_bytes()
    path = ARTIFACT_PATH_PREFIX + "attempt"
    auth = lab.client._auth(path, "e" * 64, len(data), "application/zip", attempt_id="attempt", lease_token="lease")
    assert wire(lab, data, path=path, auth=auth)[0] == 400
    bomb = archive_bytes(b"x" * (remote.MAX_WIRE_BYTES + 1), zipfile.ZIP_DEFLATED)
    assert wire(lab, bomb, path=path)[0] == 413
    assert len(lab.ends) == 2
    assert not list((lab.tmp_path / "uploads").iterdir())


class MockResponse:
    def __init__(self, body=b'{"ok":true}', *, status=200, headers=(), version=10):
        self.body, self.status, self.version, self.closed = body, status, version, False
        self.headers = Message()
        for key, value in [("Content-Type", "application/json"), ("Content-Length", str(len(body))), *headers]:
            self.headers[key] = value

    def read(self, _length):
        return self.body

    def close(self):
        self.closed = True


def client_connection(monkeypatch, response):
    connections = []

    class Connection:
        def __init__(self, *args):
            self.args, self.closed = args, False
            connections.append(self)

        def request(self, *args, **kwargs):
            self.request_args, self.request_kwargs = args, kwargs

        def getresponse(self):
            return response

        def close(self):
            self.closed = True

    monkeypatch.setattr(tailnet_rpc, "PinnedHTTPSConnection", Connection)
    return connections


@pytest.mark.parametrize("response", [MockResponse(status=302), MockResponse(version=11),
    MockResponse(headers=[("Content-Length", "2")]), MockResponse(headers=[("Content-Encoding", "gzip")]),
    MockResponse(headers=[("Transfer-Encoding", "chunked")]), MockResponse(b'{"a":1,"a":2}'),
    MockResponse(b'{"a":NaN}'), MockResponse(b'x' * (remote.MAX_WIRE_BYTES + 1))])
def test_client_rejects_invalid_response_and_always_closes(lab, monkeypatch, response):
    connections = client_connection(monkeypatch, response)
    with pytest.raises((ConnectionError, ApiError)):
        lab.client.send(message())
    assert response.closed and connections[-1].closed
    assert connections[-1]._http_vsn_str == "HTTP/1.0"


def test_client_pinned_target_and_signed_headers_no_token(lab, monkeypatch):
    response = MockResponse()
    connections = client_connection(monkeypatch, response)
    assert lab.client.send(message(elapsed_seconds=0.1)) == {"ok": True}
    connection = connections[-1]
    assert connection.args[:2] == (lab.bootstrap["collector"]["endpoint"], "b" * 64)
    assert connection.args[3] == "nController"
    headers = connection.request_kwargs["headers"]
    assert "Authorization" not in headers and remote.AUTH_HEADER in headers
    assert response.closed and connection.closed


def test_oversized_upload_never_connects(lab):
    path = lab.tmp_path / "oversized.zip"
    path.write_bytes(b"x" * (remote.MAX_WIRE_BYTES + 1))
    with pytest.raises(ApiError, match="synthetic limit"):
        lab.client.upload_artifact("attempt", "maid", "lease", path)


@pytest.mark.parametrize("value", [None, "-1", "+1", "01", "1,1", " 1", "1048577"])
def test_strict_content_length(value):
    headers = Message()
    if value is not None:
        headers["Content-Length"] = value
    with pytest.raises((ValueError, ApiError)):
        remote._length(headers)


def test_bounded_stream_does_not_follow_growth_and_rejects_truncation():
    reader = remote._BoundedReader(io.BytesIO(b"good-extra"), 4)
    assert reader.read(65536) == b"good"
    assert reader.read(65536) == b""
    reader = remote._BoundedReader(io.BytesIO(b"short"), 6)
    assert reader.read(65536) == b"short"
    with pytest.raises(OSError, match="changed"):
        reader.read(65536)


def test_invalid_signer_and_wrong_verifier_never_connect(lab):
    wrong = remote.RemotePrincessClient(lab.bootstrap, "maid", MemorySigner(lab.keys["bell"]),
                                       lab.worker_verifier, run_id="run")
    with pytest.raises(ValueError, match="signature"):
        wrong.send(message())
    with pytest.raises(ValueError, match="Verifier"):
        remote.RemotePrincessClient(lab.bootstrap, "maid", MemorySigner(lab.keys["maid"]),
                                   lab.verifier, run_id="run")


def test_expired_signature_is_rejected_before_scheduler(lab, monkeypatch):
    body = remote._json_bytes(message())
    auth = lab.client._auth(MESSAGE_PATH, hashlib.sha256(body).hexdigest(), len(body), "application/json")
    issued = json.loads(base64.b64decode(auth))["payload"]["issued_at"]
    monkeypatch.setattr(remote.time, "time", lambda: issued + 121)
    assert wire(lab, body, auth=auth)[0] == 403
    assert not lab.calls


def test_connection_is_closed_when_getresponse_fails(lab, monkeypatch):
    connections = client_connection(monkeypatch, MockResponse())
    original = tailnet_rpc.PinnedHTTPSConnection

    class Broken(original):
        def getresponse(self):
            raise OSError("synthetic dropped response")

    monkeypatch.setattr(tailnet_rpc, "PinnedHTTPSConnection", Broken)
    with pytest.raises(ConnectionError):
        lab.client.send(message())
    assert connections[-1].closed


def test_response_truncation_and_overlarge_server_response(lab, monkeypatch):
    response = MockResponse()
    response.headers.replace_header("Content-Length", str(len(response.body) + 1))
    connections = client_connection(monkeypatch, response)
    with pytest.raises(ConnectionError):
        lab.client.send(message())
    assert response.closed and connections[-1].closed
    lab.coordinator.handle_message = lambda _: {"data": "x" * remote.MAX_WIRE_BYTES}
    assert wire(lab)[0] == 500


def test_factory_uses_tls_helper_and_pinned_public_cert_only(lab, monkeypatch):
    key = Ed25519PrivateKey.generate()
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "synthetic-test")])
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
        .public_key(key.public_key()).serial_number(1)
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1)).sign(key, None))
    path = lab.tmp_path / "synthetic-cert.pem"
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    lab.bootstrap["collector"]["tls_sha256"] = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    called = []
    monkeypatch.setattr(tailnet_rpc, "create_tls_server", lambda *args: called.append(args) or SimpleNamespace())
    server = remote.create_remote_server(lab.coordinator, token="ignored", upload_dir=lab.tmp_path / "uploads",
        bootstrap=lab.bootstrap, verifier=lab.verifier, cert_path=path,
        key_path=lab.tmp_path / "synthetic-not-opened-key.pem", wake_handler=lab.wake)
    assert server.remote_policy.coordinator is lab.coordinator
    assert called[0][0] is remote.RemoteCollectorHandler
    assert called[0][1:3] == ("100.64.0.1", 18767)
    assert called[0][-1] == {"nWorker"}
