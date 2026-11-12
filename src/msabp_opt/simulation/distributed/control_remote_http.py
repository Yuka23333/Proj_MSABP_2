"""Pinned, actor-signed data plane for ``shore-remote-synthetic-v1`` only.

This adapter delegates scheduling and settlement to Princess/ControlCollector.
Its signature authenticates a project HTTP request, never authorizes an attempt:
the durable authority permit is independently checked before attempt traffic.
No constructors discover keys, probe a daemon, or retry an uncertain operation.
"""
from __future__ import annotations

import base64
import binascii
from contextlib import closing
import copy
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import tempfile
import time
from urllib.parse import urlsplit
import zipfile

from .http_api import ApiError, ARTIFACT_PATH_PREFIX, MESSAGE_PATH
from .protocol import validate_message

MODE = "shore-remote-synthetic-v1"
MAX_WIRE_BYTES = 1024 * 1024
AUTH_HEADER = "X-MSABP-Actor-Signature"
AUTH_DOMAIN = "msabp-remote-synthetic-http-request-v1"
WAKE_PATHS = frozenset(("/v1/wake/poll", "/v1/wake/complete"))
IDLE_PATH = "/v1/control/idle"
_CALLBACK_PATHS = WAKE_PATHS | {IDLE_PATH}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")
_AUTH_FIELDS = frozenset(("version", "mode", "fleet_id", "run_id", "actor_id",
    "method", "path", "content_type", "body_bytes", "body_sha256", "attempt_id",
    "lease_token", "request_id", "issued_at"))


def _identifier(value):
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError("Invalid identifier")
    return value


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True, allow_nan=False).encode("utf-8")


def _json_object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Nonfinite JSON number")
        return parsed

    def invalid_constant(_value):
        raise ValueError("Nonfinite JSON constant")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                       parse_float=number, parse_constant=invalid_constant)
    if type(value) is not dict:
        raise ValueError("JSON object required")
    return value


def _length(headers):
    values = headers.get_all("Content-Length", [])
    if (len(values) != 1 or not re.fullmatch(r"0|[1-9][0-9]{0,7}", values[0])
            or headers.get_all("Transfer-Encoding", [])
            or headers.get_all("Content-Encoding", [])):
        raise ValueError("Unique bounded Content-Length without encodings required")
    length = int(values[0])
    if length > MAX_WIRE_BYTES:
        raise ApiError(413, "Body exceeds synthetic limit")
    return length


def _content_type(headers, expected):
    values = headers.get_all("Content-Type", [])
    allowed = {expected}
    if expected == "application/json":
        allowed.add("application/json; charset=utf-8")
    if len(values) != 1 or values[0].lower() not in allowed:
        raise ValueError("Unexpected or ambiguous Content-Type")
    return values[0].lower()


def _bootstrap(value):
    from shore_control.remote import _validate_bootstrap

    return copy.deepcopy(_validate_bootstrap(value))


class _BoundedReader:
    """Never emit bytes past the signed Content-Length if a file grows."""

    def __init__(self, source, remaining):
        self.source, self.remaining = source, remaining

    def read(self, size):
        if not self.remaining:
            return b""
        chunk = self.source.read(min(size, self.remaining))
        if not chunk:
            raise OSError("Artifact changed while streaming")
        self.remaining -= len(chunk)
        return chunk


class RemotePrincessClient:
    """Outbound-only client compatible with Maid's send/upload interface.

    ``signer`` is the existing enrolled worker Client.signer; no private path is
    accepted or read. ``token`` is accepted for caller compatibility, but is not
    sent and is not an authentication mechanism. Each call uses a new pinned
    connection and never retries. Bell uses request_json on the two wake paths.
    """

    def __init__(self, bootstrap, actor_id, signer, verifier, *, run_id,
                 timeout=5.0, token=None):
        from shore_control.tailnet import valid_timeout
        from shore_control.tailnet_rpc import PinnedHTTPSConnection

        self.bootstrap = _bootstrap(bootstrap)
        self.actor_id, self.run_id = _identifier(actor_id), _identifier(run_id)
        self.registration = self.bootstrap["workers"].get(actor_id)
        if self.registration is None:
            raise ValueError("Explicit enrolled worker required")
        network = self.bootstrap["network"]
        expected_network = {**network,
            "local_node_id": self.registration["tailnet_node_id"],
            "local_address": network["nodes"][self.registration["tailnet_node_id"]]["address"]}
        if verifier.network != expected_network:
            raise ValueError("Verifier differs from enrolled worker source")
        self.signer, self.verifier = signer, verifier
        self.timeout = valid_timeout(timeout)
        self.endpoint = self.bootstrap["collector"]["endpoint"]
        self.peer_node_id = network["local_node_id"]
        self.pin = self.bootstrap["collector"]["tls_sha256"]
        check = PinnedHTTPSConnection(self.endpoint, self.pin, verifier,
                                       self.peer_node_id, self.timeout)
        check.close()  # Configuration validation only: no connect or key read.

    def _auth(self, path, body_hash, size, content_type, *, attempt_id="", lease_token=""):
        from shore_control.crypto import canonical_json, verify_object

        payload = {"version": 1, "mode": MODE, "fleet_id": self.bootstrap["fleet_id"],
            "run_id": self.run_id, "actor_id": self.actor_id, "method": "POST", "path": path,
            "content_type": content_type, "body_bytes": size, "body_sha256": body_hash,
            "attempt_id": attempt_id, "lease_token": lease_token,
            "request_id": secrets.token_hex(16), "issued_at": int(time.time())}
        envelope = self.signer.sign(AUTH_DOMAIN, payload)
        if verify_object(self.registration["public_key"], AUTH_DOMAIN, envelope) != payload:
            raise ValueError("Signer differs from enrolled actor")
        return base64.b64encode(canonical_json(envelope)).decode("ascii")

    def send(self, message):
        normalized = validate_message(message)
        if (self.registration["role"] != "maid" or normalized.get("protocol_version") != 2
                or normalized.get("run_id") != self.run_id
                or normalized.get("worker_id") != self.actor_id):
            raise ValueError("Message differs from enrolled Maid/run")
        return self._request(MESSAGE_PATH, _json_bytes(normalized), "application/json")

    def request_json(self, path, body):
        allowed = (path == IDLE_PATH and self.registration["role"] in {"bell", "maid"}
                   or path in WAKE_PATHS and self.registration["role"] == "bell")
        if (not allowed
                or type(body) is not dict):
            raise ValueError("Only enrolled wake/idle requests are supported")
        return self._request(path, _json_bytes(body), "application/json")

    def upload_artifact(self, attempt_id, worker_id, lease_token, archive_path):
        if self.registration["role"] != "maid" or worker_id != self.actor_id:
            raise ValueError("Only the enrolled Maid may upload its attempt")
        attempt_id = _identifier(attempt_id)
        if type(lease_token) is not str or not 1 <= len(lease_token) <= 256:
            raise ValueError("Bound lease token required")
        # Bound the file before hashing or opening a network connection. Stream
        # the same open file descriptor; receiver verifies bytes against digest.
        with Path(archive_path).open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if not 0 < size <= MAX_WIRE_BYTES:
                raise ApiError(413, "Artifact exceeds synthetic limit")
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            return self._request(ARTIFACT_PATH_PREFIX + attempt_id, _BoundedReader(stream, size),
                "application/zip", size=size, digest=digest,
                attempt_id=attempt_id, lease_token=lease_token)

    def _request(self, path, body, content_type, *, size=None, digest=None,
                 attempt_id="", lease_token=""):
        from shore_control.tailnet_rpc import PinnedHTTPSConnection

        if size is None:
            size, digest = len(body), hashlib.sha256(body).hexdigest()
        if not 0 < size <= MAX_WIRE_BYTES:
            raise ApiError(413, "Body exceeds synthetic limit")
        auth = self._auth(path, digest, size, content_type,
                          attempt_id=attempt_id, lease_token=lease_token)
        connection = response = None
        try:
            connection = PinnedHTTPSConnection(self.endpoint, self.pin, self.verifier,
                                               self.peer_node_id, self.timeout)
            connection._http_vsn, connection._http_vsn_str = 10, "HTTP/1.0"
            connection.request("POST", path, body=body, headers={
                "Host": urlsplit(self.endpoint).netloc, "Content-Type": content_type,
                "Content-Length": str(size), "Connection": "close", AUTH_HEADER: auth})
            response = connection.getresponse()
            if response.version != 10 or 300 <= response.status < 400:
                raise ValueError("Only direct HTTP/1.0 responses are permitted")
            _content_type(response.headers, "application/json")
            length = _length(response.headers)
            raw = response.read(length)
            if len(raw) != length:
                raise ValueError("Truncated response")
            result = _json_object(raw)
            if response.status != 200:
                raise ApiError(response.status, "Remote synthetic collector rejected request")
            return result
        except (OSError, http.client.HTTPException, ValueError, RecursionError) as exc:
            raise ConnectionError("Remote synthetic collector unavailable or invalid response") from exc
        finally:
            try:
                if response is not None:
                    response.close()
            finally:
                if connection is not None:
                    connection.close()


class _CollectorPolicy:
    def __init__(self, coordinator, bootstrap, verifier, upload_dir, wake_handler, *, receipt_only=False):
        self.coordinator, self.bootstrap = coordinator, _bootstrap(bootstrap)
        self.wake_handler = wake_handler
        self.receipt_only = receipt_only
        self.upload_dir = Path(upload_dir).resolve()
        self.database = coordinator.state.path
        if (not coordinator.controlled or coordinator.control_collector is None
                or verifier.network != self.bootstrap["network"]):
            raise ValueError("Explicit controlled collector on pinned local node required")
        self._bound_gate()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        # Transport anti-replay only; this table never allocates or settles work.
        # Retain recent IDs across process restarts. Uncertain business requests
        # recover through existing operation/outbox journals, never auto-retry.
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS remote_http_nonces ("
                       "actor_id TEXT, request_id TEXT, issued_at INTEGER, "
                       "PRIMARY KEY(actor_id,request_id))")

    def _bound_gate(self):
        c, b = self.coordinator, self.bootstrap
        gate = c.state._authority_for_run(c.run_id)
        if (gate.controller.authority_id != b["authority_id"]
                or gate.controller.fleet_id != b["fleet_id"]
                or gate.binding["host_id"] != b["network"]["local_node_id"]
                or gate.job["scope"]["collector_url"] != b["collector"]["endpoint"]):
            raise ValueError("Collector differs from frozen job authority/host")
        with gate.journal.transaction() as db:
            registered = db.execute("SELECT registration FROM control_runs WHERE run_id=?",
                                    (c.run_id,)).fetchone()
        if (registered is None or json.loads(registered["registration"]).get(
                "input_manifest_sha256") != c.csv_sha256):
            raise ValueError("Collector differs from registered frozen input")
        return gate

    def authenticate(self, headers, *, node_id, path, length, content_type):
        from shore_control.crypto import verify_object

        if self.receipt_only and path != IDLE_PATH:
            raise ApiError(403, "Receipt-only listener rejects work endpoints")
        values = headers.get_all(AUTH_HEADER, [])
        if len(values) != 1 or len(values[0]) > 8192:
            raise ApiError(401, "Unique actor signature required")
        try:
            envelope = _json_object(base64.b64decode(values[0], validate=True))
            actor = envelope.get("payload", {}).get("actor_id")
            row = self.bootstrap["workers"].get(actor)
            if row is None or row["tailnet_node_id"] != node_id:
                raise ValueError("Source differs from enrolled actor")
            payload = verify_object(row["public_key"], AUTH_DOMAIN, envelope)
            expected = {"version": 1, "mode": MODE, "fleet_id": self.bootstrap["fleet_id"],
                "run_id": self.coordinator.run_id, "actor_id": actor, "method": "POST",
                "path": path, "content_type": content_type, "body_bytes": length}
            if (set(payload) != _AUTH_FIELDS or type(payload["version"]) is not int
                    or type(payload["body_bytes"]) is not int
                    or any(payload.get(k) != v for k, v in expected.items())
                    or not re.fullmatch(r"[a-f0-9]{64}", payload["body_sha256"])
                    or not re.fullmatch(r"[a-f0-9]{32}", payload["request_id"])
                    or type(payload["issued_at"]) is not int
                    or not int(time.time()) - 120 <= payload["issued_at"] <= int(time.time()) + 5):
                raise ValueError("Signed HTTP binding differs")
            role = row["role"] if path == IDLE_PATH else ("bell" if path in WAKE_PATHS else "maid")
            action = "wake" if role == "bell" else "attempt"
            if role not in {"bell", "maid"} or row["role"] != role or row["allowed_actions"] != [action]:
                raise ValueError("Actor role forbids endpoint")
            gate = self._bound_gate()
            if gate.registrations.get(actor) != row:
                raise ValueError("Actor differs from frozen project registration")
            if path.startswith(ARTIFACT_PATH_PREFIX):
                if (payload["attempt_id"] != _identifier(path[len(ARTIFACT_PATH_PREFIX):])
                        or type(payload["lease_token"]) is not str
                        or not 1 <= len(payload["lease_token"]) <= 256):
                    raise ValueError("Upload binding differs")
                self.check_attempt(actor, payload["attempt_id"])
            elif payload["attempt_id"] != "" or payload["lease_token"] != "":
                raise ValueError("Unexpected upload binding")
            return payload
        except (ValueError, TypeError, AttributeError, KeyError, binascii.Error) as exc:
            raise ApiError(403, "Actor signature/source/scope rejected") from exc

    def consume(self, auth):
        with closing(sqlite3.connect(self.database, timeout=5)) as db, db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            # Authentication precedes body reads and may precede a write-lock
            # wait. Recheck under the acquired lock before pruning: otherwise
            # an expired, already consumed request could erase its own nonce
            # and enter the business handler a second time.
            now = int(time.time())
            if not now - 120 <= auth["issued_at"] <= now + 5:
                raise ApiError(403, "Signed request expired before consumption")
            db.execute("DELETE FROM remote_http_nonces WHERE issued_at < ?", (now - 120,))
            if db.execute("SELECT COUNT(*) FROM remote_http_nonces").fetchone()[0] >= 4096:
                raise ApiError(429, "Signed request window is full")
            try:
                db.execute("INSERT INTO remote_http_nonces VALUES(?,?,?)",
                           (auth["actor_id"], auth["request_id"], auth["issued_at"]))
            except sqlite3.IntegrityError as exc:
                raise ApiError(409, "Signed request already consumed") from exc

    def check_attempt(self, actor, attempt_id, claimed_control=None):
        from shore_control.permits import verify_permit
        from shore_control.registry import registration_sha256

        c, gate = self.coordinator, self._bound_gate()
        saved = c.state.control_for_attempt(_identifier(attempt_id))
        if saved is None:
            raise ApiError(403, "No durable authority permit for attempt")
        permit = verify_permit(self.bootstrap["authority_public_key"], saved["permit"], saved["descriptor"])
        row = self.bootstrap["workers"][actor]
        expected = {"fleet_id": self.bootstrap["fleet_id"], "run_id": c.run_id,
            "worker_id": actor, "attempt_id": attempt_id, "action": "attempt",
            "device_id": row["device_id"], "worker_role": "maid",
            "worker_registration_sha256": registration_sha256(row),
            **{k: gate.binding[k] for k in ("job_id", "epoch", "host_id")},
            **gate.job["scope"]}
        # Deliberately do not compare current session_id: late old-session
        # results retain their original valid permit and are still collectible.
        if any(permit.get(k) != value for k, value in expected.items()):
            raise ApiError(403, "Attempt differs from frozen job/actor")
        with closing(sqlite3.connect(self.database)) as db:
            row = db.execute("SELECT a.run_id,a.worker_id,t.row_sha256 FROM attempts a "
                "JOIN tasks t ON t.run_id=a.run_id AND t.case_id=a.case_id "
                "WHERE a.attempt_id=?", (attempt_id,)).fetchone()
        if row != (c.run_id, actor, permit["input_sha256"]):
            raise ApiError(403, "Permit differs from persisted attempt/input")
        if claimed_control is not None and (type(claimed_control) is not dict
                or claimed_control.get("permit_id") != permit["permit_id"]):
            raise ApiError(403, "Completion differs from registered permit")

    def message(self, auth, value):
        message = validate_message(value)
        if (message["protocol_version"] != 2 or message["run_id"] != self.coordinator.run_id
                or message.get("worker_id") != auth["actor_id"]):
            raise ApiError(403, "Message differs from signed actor/run")
        kind, payload = message["type"], message["payload"]
        if kind not in {"hello", "request_task", "heartbeat", "complete", "control_terminal"}:
            raise ApiError(403, "Unsupported controlled message")
        if kind in {"heartbeat", "complete", "control_terminal"}:
            self.check_attempt(auth["actor_id"], payload.get("attempt_id"),
                               payload.get("control_authority"))
        return self.coordinator.handle_message(message)


class RemoteCollectorHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "MSABPRemoteSynthetic"

    def log_message(self, *_args):
        pass  # Do not leak signed payloads, local paths or secret material.

    def do_POST(self):  # noqa: N802
        self.close_connection = True
        self._body_remaining = None
        try:
            # Pin a single bounded framing length before any early rejection.
            # A rejected request may send headers and body in separate TLS
            # records: closing with unread body can discard our 403/404 on
            # Windows. Draining these bytes never authenticates/parses them.
            length = _length(self.headers)
            self._body_remaining = length
            host = f"{self.server.server_address[0]}:{self.server.server_address[1]}"
            if (self.request_version != "HTTP/1.0"
                    or self.headers.get_all("Host", []) != [host]
                    or self.headers.get_all("Origin", []) or self.headers.get_all("Expect", [])):
                raise ValueError("Explicit closed HTTP/1.0 request required")
            # Separate opt-in job authentication: never invent a discovery run
            # or send these requests through the existing run authentication.
            # Keep the legacy run-only deployed closure independent of the new
            # supervisor modules. Import job policy only after explicit opt-in.
            if self.path in {"/v1/job/offer", "/v1/job/material"}:
                policy = self.server.remote_policy
                if policy.receipt_only and not getattr(self.server, "allow_job_reads", False):
                    raise ApiError(403, "Receipt-only listener rejects job endpoints")
                service = getattr(self.server, "job_service", None)
                if service is None:
                    raise ApiError(404, "Job endpoints disabled")
                from .control_job_auth import MAX_REQUEST_BYTES

                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ApiError(413, "Job request exceeds bounded limit")
                raw = self._read_body(length)
                if len(raw) != length:
                    raise ValueError("Truncated job request")
                result = service.handle(self.path, raw, headers=self.headers,
                    client_node_id=getattr(self.connection, "client_node_id", None))
                self._reply(200, result)
                return
            artifact = self.path.startswith(ARTIFACT_PATH_PREFIX)
            if self.path != MESSAGE_PATH and self.path not in _CALLBACK_PATHS and not artifact:
                raise ApiError(404, "Unknown endpoint")
            content_type = _content_type(self.headers, "application/zip" if artifact else "application/json")
            policy = self.server.remote_policy
            auth = policy.authenticate(self.headers,
                node_id=getattr(self.connection, "client_node_id", None),
                path=self.path, length=length, content_type=content_type)
            if artifact:
                policy.consume(auth)
                result = self._artifact(policy, auth, length)
            else:
                raw = self._read_body(length)
                if len(raw) != length or hashlib.sha256(raw).hexdigest() != auth["body_sha256"]:
                    raise ValueError("Signed body size/digest differs")
                value = _json_object(raw)
                policy.consume(auth)
                if self.path in _CALLBACK_PATHS:
                    if policy.wake_handler is None:
                        raise ApiError(404, "Wake endpoint disabled")
                    result = policy.wake_handler(self.path, value, actor_id=auth["actor_id"],
                                                 node_id=self.connection.client_node_id)
                else:
                    result = policy.message(auth, value)
            self._reply(200, result)
        except ApiError as exc:
            self._reject(exc.status, {"error": "Remote synthetic request rejected"})
        except (ValueError, TypeError, KeyError, RecursionError, UnicodeError):
            self._reject(400, {"error": "Invalid remote synthetic request"})
        except (TimeoutError, OSError):
            self.close_connection = True
        except Exception:
            self._reject(500, {"error": "Remote synthetic request failed"})

    def _read_body(self, size):
        if self._body_remaining is None or not 0 <= size <= self._body_remaining:
            raise ValueError("Request body exceeds its bounded framing")
        raw = self.rfile.read(size)
        self._body_remaining -= len(raw)
        return raw

    def _reject(self, status, value):
        # Only an already checked unique Content-Length can drive this loop.
        # Ambiguous/oversized framing is never used to consume an arbitrary
        # stream. The Shore TLS server's whole-connection deadline still applies
        # to a client that withholds rejected body bytes (no new wait budget).
        try:
            while self._body_remaining:
                if not self._read_body(min(64 * 1024, self._body_remaining)):
                    self.close_connection = True
                    return
        except (OSError, TimeoutError):
            self.close_connection = True
            return
        self._reply(status, value)

    def _artifact(self, policy, auth, length):
        c = policy.coordinator
        ticket = c.begin_artifact_upload(auth["attempt_id"], auth["actor_id"], auth["lease_token"])
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(prefix="remote-", suffix=".part", dir=policy.upload_dir)
            temporary = Path(name)
            digest, remaining = hashlib.sha256(), length
            with os.fdopen(descriptor, "wb") as stream:
                while remaining:
                    chunk = self._read_body(min(64 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Truncated artifact")
                    stream.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if digest.hexdigest() != auth["body_sha256"]:
                raise ValueError("Artifact differs from signed digest")
            with zipfile.ZipFile(temporary) as archive:
                members = archive.infolist()
                if len(members) > 64 or sum(item.file_size for item in members) > MAX_WIRE_BYTES:
                    raise ApiError(413, "Expanded artifact exceeds synthetic limit")
            return c.handle_artifact(auth["attempt_id"], auth["actor_id"], auth["lease_token"],
                                     temporary, digest.hexdigest(), length)
        finally:
            # Mark finished only after cleanup succeeds; a surviving partial
            # stream must continue blocking drain through ControlCollector.
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            c.end_artifact_upload(ticket)

    def _reply(self, status, value):
        self.close_connection = True
        try:
            if type(value) is not dict:
                raise ValueError("Response object required")
            body = _json_bytes(value)
            if len(body) > MAX_WIRE_BYTES:
                raise ValueError("Response exceeds synthetic limit")
        except (ValueError, TypeError, RecursionError):
            status, body = 500, b'{"error":"Invalid bounded response"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def create_remote_server(coordinator, *, token, upload_dir, bootstrap, verifier,
                         cert_path, key_path, wake_handler=None, receipt_only=False,
                         job_service=None, allow_job_reads=False):
    """Create (but do not run) a source-verified collector on its pinned endpoint.

    Caller owns serve_forever/shutdown/server_close. ``token`` is intentionally
    ignored: a bearer secret must not replace the worker signature or F permit.
    Bell callback: ``wake_handler(path, body, *, actor_id, node_id)``; only the
    two WAKE_PATHS and IDLE_PATH are allowed. Callback must validate wake permits
    and the separate signed idle record against actual reconciled outboxes.
    Job discovery/download stays disabled unless an explicit JobMaterialService
    is supplied. Receipt-only listeners additionally require the explicit
    ``allow_job_reads`` opt-in; that never enables wake/attempt/run traffic.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from shore_control.tailnet_rpc import create_tls_server

    bootstrap = _bootstrap(bootstrap)
    if type(allow_job_reads) is not bool or allow_job_reads and job_service is None:
        raise ValueError("Read-only recovery opt-in requires an explicit job service")
    if job_service is not None:
        from .control_job_http import JobMaterialService

        if not isinstance(job_service, JobMaterialService):
            raise ValueError("Explicit locally configured job service required")
        job_service.assert_bootstrap(bootstrap)
    cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    if hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest() != bootstrap["collector"]["tls_sha256"]:
        raise ValueError("Collector certificate differs from trusted bootstrap")
    policy = _CollectorPolicy(coordinator, bootstrap, verifier, upload_dir, wake_handler,
                              receipt_only=receipt_only)
    endpoint = urlsplit(bootstrap["collector"]["endpoint"])
    server = create_tls_server(RemoteCollectorHandler, endpoint.hostname, endpoint.port,
        str(cert_path), str(key_path), verifier,
        {row["tailnet_node_id"] for row in bootstrap["workers"].values()})
    server.remote_policy = policy
    server.job_service = job_service
    server.allow_job_reads = allow_job_reads
    return server
