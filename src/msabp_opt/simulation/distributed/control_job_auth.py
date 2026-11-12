"""Bell-only job-scoped HTTP authentication, independent of run authentication.

Wire v1 is a Shore Ed25519 envelope ``{payload, signature}``, signed under
``AUTH_DOMAIN``.  Its exact payload fields are ``AUTH_FIELDS`` below; no run_id,
permit, session credential, command, URL or public key is accepted here.  Both
endpoints are POST/application-json and only authenticate a bounded request.
They neither issue a wake permit nor prove that a job is currently ACTIVE.

The caller must obtain ``expected_job`` from its freshly verified authority
status/local trusted job binding, verify TLS and derive ``client_node_id`` from
the actual accepted socket, never a request header.  Bootstrap and the nonce
database path must come from protected local policy.  HTTP framing, duplicate
JSON fields and application body/response schemas remain the caller's job.

The nonce ledger is explicitly installed once with ``initialize_job_nonce_db``.
Verification never creates/rebuilds it, and commits consumption before returning
the authenticated actor.  A lost HTTP response must be retried with a new signed
request id, while the application returns only the *original* issued materials.
"""
from __future__ import annotations

from contextlib import closing
import copy
import hashlib
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import time

from .control_remote_worker import checked_path


MODE = "shore-remote-synthetic-v1"
AUTH_DOMAIN = "msabp-job-http-request-v1"
AUTH_VERSION = 1
AUTH_HEADER = "X-MSABP-Job-Signature"
OFFER_PATH = "/v1/job/offer"
MATERIAL_PATH = "/v1/job/material"
JOB_PATHS = frozenset((OFFER_PATH, MATERIAL_PATH))
CONTENT_TYPE = "application/json"
MAX_REQUEST_BYTES = 64 * 1024
MAX_ENVELOPE_BYTES = 8192
MAX_AGE_MS = 120_000
MAX_FUTURE_MS = 5_000
MAX_NONCES = 4096
NONCE_TABLE = "job_http_nonces"
CONTEXT_FIELDS = frozenset(("fleet_id", "job_id", "epoch", "actor_id", "device_id", "node_id"))
EXPECTED_JOB_FIELDS = frozenset(("fleet_id", "job_id", "epoch"))
AUTH_FIELDS = CONTEXT_FIELDS | frozenset((
    "version", "mode", "method", "path", "content_type", "body_bytes",
    "body_sha256", "issued_at_ms", "request_id",
))
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_FLEET = re.compile(r"shore-princess-[a-f0-9]{24}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_REQUEST_ID = re.compile(r"[a-f0-9]{32}\Z")
_SIGNATURE = re.compile(r"[A-Za-z0-9+/]{86}==\Z")
_MAX_INTEGER = (1 << 63) - 1


class JobAuthError(ValueError):
    """Fail-closed authentication error; ``status`` is an optional HTTP mapping."""

    def __init__(self, message, *, status=403):
        super().__init__(message)
        self.status = status


def _require(condition, message, *, status=403):
    if not condition:
        raise JobAuthError(message, status=status)


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= _MAX_INTEGER


def _text(value, pattern):
    return type(value) is str and pattern.fullmatch(value) is not None


def _fields(value, expected):
    _require(type(value) is dict and set(value) == expected, "Unexpected authentication fields")


def _context(context, *, expected=False):
    _fields(context, EXPECTED_JOB_FIELDS if expected else CONTEXT_FIELDS)
    _require(_text(context["fleet_id"], _FLEET), "Invalid fleet identifier")
    _require(_text(context["job_id"], _ID), "Invalid job identifier")
    _require(_integer(context["epoch"], 1), "Invalid job epoch")
    if not expected:
        for name in ("actor_id", "device_id", "node_id"):
            _require(_text(context[name], _ID), "Invalid actor/device/node identifier")


def _request(method, path, body):
    _require(type(method) is str and method == "POST", "Job requests require POST")
    _require(type(path) is str and path in JOB_PATHS, "Unknown job endpoint")
    _require(type(body) is bytes, "Job request body must be exact bytes")
    _require(0 < len(body) <= MAX_REQUEST_BYTES, "Job request body exceeds limit", status=413)


def _payload(payload):
    _fields(payload, AUTH_FIELDS)
    _context({key: payload[key] for key in CONTEXT_FIELDS})
    _require(type(payload["version"]) is int and payload["version"] == AUTH_VERSION,
             "Unknown job authentication version")
    _require(type(payload["mode"]) is str and payload["mode"] == MODE,
             "Unknown job authentication mode")
    _require(type(payload["method"]) is str and payload["method"] == "POST",
             "Job requests require POST")
    _require(type(payload["path"]) is str and payload["path"] in JOB_PATHS,
             "Unknown job endpoint")
    _require(type(payload["content_type"]) is str and payload["content_type"] == CONTENT_TYPE,
             "Job requests require JSON")
    _require(_integer(payload["body_bytes"], 1) and payload["body_bytes"] <= MAX_REQUEST_BYTES,
             "Invalid signed body size")
    _require(_text(payload["body_sha256"], _SHA256), "Invalid body digest")
    _require(_text(payload["request_id"], _REQUEST_ID), "Invalid request id")
    _require(_integer(payload["issued_at_ms"]), "Invalid issued-at time")


def _envelope(envelope):
    from shore_control.crypto import canonical_json

    _fields(envelope, {"payload", "signature"})
    _payload(envelope["payload"])
    _require(_text(envelope["signature"], _SIGNATURE), "Invalid signature encoding")
    _require(len(canonical_json(envelope)) <= MAX_ENVELOPE_BYTES,
             "Job authentication envelope exceeds limit", status=413)


def _wall_clock_ms():
    return time.time_ns() // 1_000_000


def _check_time(issued_at_ms, clock_ms):
    now = clock_ms()
    _require(_integer(now), "Invalid local authentication clock")
    _require(now - MAX_AGE_MS <= issued_at_ms <= now + MAX_FUTURE_MS,
             "Signed job request outside time window")
    return now


def _nonce_path(path, *, allow_missing=False):
    """Require a local absolute path without redirected parents or file aliases."""
    _require(isinstance(path, (str, Path)) and str(path) not in {"", ":memory:"},
             "Explicit persistent nonce database required")
    try:
        path = checked_path(path)
        try:
            info = path.lstat()
        except FileNotFoundError:
            if allow_missing:
                return path
            raise
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                 "Job authentication ledger must be regular and non-aliased", status=503)
        return path
    except JobAuthError:
        raise
    except (OSError, ValueError) as exc:
        raise JobAuthError("Job authentication ledger path unavailable or redirected", status=503) from exc


def initialize_job_nonce_db(nonce_db):
    """Install a new, empty SQLite nonce ledger; refuse any existing path.

    This is an explicit administrator/installation operation, not a recovery
    fallback.  Its parent must already exist.  An interrupted initialization
    leaves its evidence in place and requires maintenance, never replacement.
    The path must be absolute/local, without traversal or redirected parents;
    an existing ledger must be a regular file with exactly one hard link.
    """
    path = _nonce_path(nonce_db, allow_missing=True)
    with path.open("xb"):
        pass
    path = _nonce_path(path)
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as db, db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE job_http_auth_meta (schema_version INTEGER NOT NULL)")
        db.execute("INSERT INTO job_http_auth_meta VALUES (?)", (AUTH_VERSION,))
        db.execute("CREATE TABLE job_http_nonces (actor_id TEXT NOT NULL, "
                   "request_id TEXT NOT NULL, issued_at_ms INTEGER NOT NULL, "
                   "PRIMARY KEY (actor_id, request_id))")


def _consume(payload, nonce_db, clock_ms):
    path = _nonce_path(nonce_db)
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=5)) as db, db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            # The signature/time check may precede a write-lock wait. Recheck
            # BEFORE pruning: an expired replay must not delete its own nonce.
            now = _check_time(payload["issued_at_ms"], clock_ms)
            _require(db.execute("SELECT schema_version FROM job_http_auth_meta").fetchall()
                     == [(AUTH_VERSION,)], "Invalid job authentication ledger", status=503)
            db.execute("DELETE FROM job_http_nonces WHERE issued_at_ms < ?", (now - MAX_AGE_MS,))
            existing = db.execute("SELECT 1 FROM job_http_nonces WHERE actor_id=? AND request_id=?",
                                  (payload["actor_id"], payload["request_id"])).fetchone()
            _require(existing is None, "Signed job request already consumed", status=409)
            _require(db.execute("SELECT COUNT(*) FROM job_http_nonces").fetchone()[0] < MAX_NONCES,
                     "Job authentication window is full", status=429)
            db.execute("INSERT INTO job_http_nonces VALUES (?, ?, ?)",
                       (payload["actor_id"], payload["request_id"], payload["issued_at_ms"]))
    except sqlite3.Error as exc:
        raise JobAuthError("Job authentication ledger unavailable", status=503) from exc


def sign_job_request(signer, context, method, path, body, *, now_ms=None, request_id=None):
    """Sign one job request with the existing Bell signer; never load any key.

    ``context`` is the exact dictionary ``{fleet_id, job_id, epoch, actor_id,
    device_id, node_id}``.  Epoch is a positive integer; identifiers follow the
    enrolled Shore registry grammar (1-128 ASCII characters). ``body`` is the
    exact nonempty HTTP bytes, bounded to 64 KiB.  ``now_ms`` is an integer Unix
    millisecond time and ``request_id`` is 32 lowercase hex characters; omitted
    values use the wall clock and a fresh cryptographically random 128-bit id.
    """
    _context(context)
    _request(method, path, body)
    payload = {**context, "version": AUTH_VERSION, "mode": MODE,
        "method": method, "path": path, "content_type": CONTENT_TYPE,
        "body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
        "issued_at_ms": _wall_clock_ms() if now_ms is None else now_ms,
        "request_id": secrets.token_hex(16) if request_id is None else request_id}
    _payload(payload)
    envelope = signer.sign(AUTH_DOMAIN, copy.deepcopy(payload))
    _envelope(envelope)
    _require(envelope["payload"] == payload, "Signer changed job request binding")
    return envelope


def verify_job_request(envelope, body, method, path, client_node_id, bootstrap,
                       expected_job, nonce_db, *, clock_ms=None):
    """Verify and consume a signed request, returning its Bell ``actor_id``.

    ``expected_job`` has exactly ``{fleet_id, job_id, epoch}``, independent of
    caller-controlled request fields. ``bootstrap`` is the existing complete
    Shore remote bootstrap, including the pinned actor/device/node registry.
    ``client_node_id`` MUST come from the source-verified TLS socket.  Optional
    ``clock_ms`` is a callable (not a frozen timestamp), invoked before and after
    the SQLite write-lock acquisition. Errors are ``JobAuthError(ValueError)``.

    This function does not perform I/O beyond the explicit nonce database, does
    not establish TLS, and does not authorize material changes or execution.
    """
    from shore_control.crypto import verify_object
    from shore_control.remote import _validate_bootstrap

    _request(method, path, body)
    _envelope(envelope)
    _context(expected_job, expected=True)
    _require(_text(client_node_id, _ID), "Explicit verified socket source required")
    try:
        trusted = _validate_bootstrap(copy.deepcopy(bootstrap))
        payload = copy.deepcopy(envelope["payload"])
        actor = payload["actor_id"]
        row = trusted["workers"].get(actor)
        _require(row is not None and row["role"] == "bell" and row["allowed_actions"] == ["wake"],
                 "Only an enrolled Bell may read job materials")
        _require(payload["fleet_id"] == trusted["fleet_id"]
                 and all(payload[key] == expected_job[key] for key in EXPECTED_JOB_FIELDS),
                 "Signed job differs from expected fleet/job/epoch")
        _require(payload["device_id"] == row["device_id"]
                 and payload["node_id"] == row["tailnet_node_id"] == client_node_id,
                 "Signed actor/device/source differs from enrolled Bell")
        _require(payload["method"] == method and payload["path"] == path
                 and payload["body_bytes"] == len(body)
                 and payload["body_sha256"] == hashlib.sha256(body).hexdigest(),
                 "Signed endpoint/body differs from actual request")
        _require(verify_object(row["public_key"], AUTH_DOMAIN, envelope) == payload,
                 "Invalid job request signature")
    except JobAuthError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise JobAuthError("Invalid job signing identity or local bootstrap") from exc
    clock_ms = _wall_clock_ms if clock_ms is None else clock_ms
    _require(callable(clock_ms), "Authentication clock must be callable")
    _check_time(payload["issued_at_ms"], clock_ms)
    _consume(payload, nonce_db, clock_ms)
    return actor
