"""Opt-in, read-only job discovery/download for the fixed synthetic supervisor.

The caller supplies an approved LOCAL release profile/bootstrap, the ORIGINAL
job binding and a read-only callback for the durable, already issued assignment.
Neither endpoint calls a controller, issues a permit, accepts work or establishes
ACTIVE. An offer is only a TLS-authenticated hint; downloaded bytes must pass the
original authority permit and fixed runtime checks before local staging. The
same issued material remains readable in drain, using fresh transport nonces.

Wire: POST /v1/job/offer with canonical {}; POST /v1/job/material with the exact
four MATERIAL_REQUEST_FIELDS from that offer. Both use the independent job auth
header/domain and closed HTTP/1.0. Material is the existing canonical object,
not a wrapper, an archive, executable code or an instruction to launch.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import re
import threading
from urllib.parse import urlsplit

from . import control_job_auth as auth
from .control_job_material import (
    MAX_CSV_BYTES, MAX_MATERIAL_BYTES, MAX_RUNTIME_BYTES, _binding, _local_bootstrap,
    encode_material, synthetic_csv, validate_material,
)
from .control_release import safe_job_id, validate_profile
from .control_remote_http import _json_bytes, _json_object, _length
from .control_remote_worker import BELL_ACTOR, MODE
from .http_api import ApiError

OFFER_SCHEMA = "msabp.supervisor-job-offer"
MAX_OFFER_BYTES = 64 * 1024
MATERIAL_REQUEST_FIELDS = frozenset((
    "material_sha256", "release_id", "permit_id", "permit_sha256",
))
OFFER_FIELDS = MATERIAL_REQUEST_FIELDS | frozenset((
    "schema", "schema_version", "mode", "fleet_id", "job_id", "epoch", "run_id",
    "runtime_sha256", "material_bytes", "terminal_proof",
))
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_PERMIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(raw, limit):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError("Missing or oversized canonical JSON")
    value = _json_object(raw)
    if _json_bytes(value) != raw:
        raise ValueError("Exact canonical JSON required")
    return value


def verify_release_proof(proof, bootstrap, expected_binding):
    """Verify original controller/authority release evidence, without any RPC.

    A release proof does not prove local outboxes/UNKNOWN or processes are
    clear. The supervisor must independently reconcile those before retiring a
    job. The original session cannot be replaced with a resumed session here.
    """
    from shore_control.crypto import verify_object

    binding = expected_binding
    if type(proof) is not dict or set(proof) != {"request", "response"}:
        raise ValueError("Original signed release request/response required")
    proof = _canonical(_json_bytes(proof), MAX_OFFER_BYTES)
    request = proof["request"]
    if (type(request) is not dict or set(request) != {"payload", "signature"}
            or type(request["payload"]) is not dict):
        raise ValueError("Original signed release request required")
    actor = request.get("payload", {}).get("actor_id")
    if type(actor) is not str or actor not in bootstrap["controllers"]:
        raise ValueError("Release requires the pinned controller signature")
    payload = verify_object(bootstrap["controllers"][actor], "request", request)
    if (set(payload) != {"version", "authority_id", "actor_id", "request_id", "nonce", "method", "params"}
            or type(payload["version"]) is not int or payload["version"] != 1
            or payload["authority_id"] != bootstrap["authority_id"]
            or payload["method"] != "release_job"
            or type(payload["request_id"]) is not str
            or not _PERMIT_ID.fullmatch(payload["request_id"])
            or type(payload["nonce"]) is not str or not _SHA256.fullmatch(payload["nonce"])):
        raise ValueError("Invalid original release request")
    params = payload["params"]
    bound_fields = {"fleet_id", "job_id", "epoch", "host_id", "session_id"}
    if (type(params) is not dict
            or set(params) != bound_fields | {"checkpoint_sha256", "disposition", "reason"}
            or type(params["epoch"]) is not int
            or any(params[key] != binding[key] for key in bound_fields)
            or type(params["disposition"]) is not str
            or params["disposition"] not in {"completed", "closed"}
            or type(params["reason"]) is not str or not 0 < len(params["reason"]) <= 512
            or type(params["checkpoint_sha256"]) is not str
            or not _SHA256.fullmatch(params["checkpoint_sha256"])):
        raise ValueError("Release differs from original job/session")
    response = verify_object(bootstrap["authority_public_key"], "response", proof["response"])
    if (set(response) != {"version", "authority_id", "request_id", "actor_id", "request_sha256",
                          "method", "ok", "result"}
            or type(response["version"]) is not int or response["version"] != 1
            or response["authority_id"] != bootstrap["authority_id"]
            or response["request_id"] != payload["request_id"] or response["actor_id"] != actor
            or response["request_sha256"] != _digest(_json_bytes(payload))
            or response["method"] != "release_job" or response["ok"] is not True):
        raise ValueError("Authority release response differs from signed request")
    result = response["result"]
    if (type(result) is not dict
            or set(result) != bound_fields | {"phase", "lease_until_ms", "scope", "checkpoint", "disposition"}
            or type(result["epoch"]) is not int
            or any(result[key] != binding[key] for key in bound_fields)
            or result["phase"] != "RELEASED"
            or type(result["lease_until_ms"]) is not int or result["lease_until_ms"] != 0
            or _json_bytes(result["scope"]) != _json_bytes({k: binding[k] for k in (
                "project_id", "collector_url", "data_root", "job_spec")})
            or result["disposition"] != params["disposition"]):
        raise ValueError("Release summary differs from original job/scope")
    checkpoint = result["checkpoint"]
    counters = {"pending_count", "uploads_in_progress", "worker_outboxes_pending",
                "receipts_pending", "local_unresolved_count", "gpu_uncollected"}
    if (type(checkpoint) is not dict
            or set(checkpoint) != counters | {"manifest_sha256", "ledger_sha256", "checkpoint_sha256"}
            or any(type(checkpoint[k]) is not int or checkpoint[k] < 0 for k in counters)
            or any(checkpoint[k] for k in counters - {"pending_count"})
            or params["disposition"] == "completed" and checkpoint["pending_count"] != 0
            or any(type(checkpoint[k]) is not str or not _SHA256.fullmatch(checkpoint[k])
                   for k in ("manifest_sha256", "ledger_sha256", "checkpoint_sha256"))
            or checkpoint["checkpoint_sha256"] != params["checkpoint_sha256"]
            or _digest(_json_bytes({k: v for k, v in checkpoint.items()
                                   if k != "checkpoint_sha256"})) != params["checkpoint_sha256"]):
        raise ValueError("Release checkpoint is not settled or differs from its digest")
    return result


def _offer(raw, verified, binding, terminal_proof=None):
    record = verified.record
    runtime = _json_object(verified.files["maid-runtime.json"])
    return {"schema": OFFER_SCHEMA, "schema_version": 1, "mode": MODE,
        **{key: binding[key] for key in auth.EXPECTED_JOB_FIELDS},
        "release_id": verified.release_id, "run_id": runtime["run_id"],
        "runtime_sha256": record["runtime_sha256"], "permit_id": record["permit_id"],
        "permit_sha256": _digest(_json_bytes(record["control_authority"]["permit"])),
        "material_sha256": verified.material_sha256, "material_bytes": len(raw),
        "terminal_proof": terminal_proof}


def _validate_offer(value, profile, binding):
    if (type(value) is not dict or set(value) != OFFER_FIELDS
            or value["schema"] != OFFER_SCHEMA or type(value["schema_version"]) is not int
            or value["schema_version"] != 1 or value["mode"] != MODE
            or value["release_id"] != profile["release_id"]
            or type(value["epoch"]) is not int
            or any(value[key] != binding[key] for key in auth.EXPECTED_JOB_FIELDS)
            or type(value["material_bytes"]) is not int
            or not 0 < value["material_bytes"] <= MAX_MATERIAL_BYTES):
        raise ValueError("Offer differs from the approved release/original job")
    for key in ("release_id", "material_sha256", "runtime_sha256", "permit_sha256"):
        if type(value[key]) is not str or not _SHA256.fullmatch(value[key]):
            raise ValueError("Invalid offer digest")
    if type(value["permit_id"]) is not str or not _PERMIT_ID.fullmatch(value["permit_id"]):
        raise ValueError("Invalid original permit identifier")
    safe_job_id(value["run_id"])
    return _canonical(_json_bytes(value), MAX_OFFER_BYTES)


class JobMaterialService:
    """Read-only bridge from original issued assignment to bounded HTTP data.

    ``read_assignment`` MUST read only the original durable assignment and return
    ``None`` when it was never issued. Do not pass wake polling, ``_assignment``
    or ``_drain_assignment``: those may issue/reconcile through authority. This
    service intentionally has no gate/session/controller reference. Nonce DB
    initialization is an explicit installer/controller step, never a fallback.
    """

    def __init__(self, *, profile, bootstrap_data, expected_binding, read_assignment,
                 csv_data, nonce_db, read_release_proof=None):
        self._profile = validate_profile(profile)
        self._bootstrap = _local_bootstrap(self._profile, bootstrap_data)
        self._bootstrap_data = bytes(bootstrap_data)
        self._binding = _binding(expected_binding, self._bootstrap)
        if not callable(read_assignment):
            raise ValueError("Explicit read-only original assignment source required")
        if read_release_proof is not None and not callable(read_release_proof):
            raise ValueError("Release proof source must be an explicit read-only callable")
        if (type(csv_data) is not bytes or len(csv_data) > MAX_CSV_BYTES
                or csv_data != synthetic_csv(self._binding["job_spec"]["approved_units"])):
            raise ValueError("Only the exact 1/4-point synthetic CSV is allowed")
        self._read_assignment = read_assignment
        self._csv_data, self._nonce_db = csv_data, nonce_db
        self._lock = threading.RLock()
        self._original_bytes = None
        self._read_release_proof = read_release_proof
        self._release_bytes = None

    def _terminal(self):
        proof = None if self._read_release_proof is None else self._read_release_proof()
        if proof is None:
            if self._release_bytes is not None:
                raise ValueError("Observed original release proof disappeared")
            return None
        proof_raw = _json_bytes(proof)
        proof = _canonical(proof_raw, MAX_OFFER_BYTES)
        verify_release_proof(proof, self._bootstrap, self._binding)
        if self._release_bytes is not None and proof_raw != self._release_bytes:
            raise ValueError("Observed original release proof changed")
        self._release_bytes = proof_raw
        return proof

    def assert_bootstrap(self, bootstrap):
        """Factory checks the service cannot install another trust configuration."""
        if _json_bytes(bootstrap) != _json_bytes(self._bootstrap):
            raise ValueError("Job service differs from the pinned collector bootstrap")

    def _original(self):
        with self._lock:
            assignment = self._read_assignment()
            if assignment is None:
                if self._original_bytes is not None:
                    raise ApiError(503, "Original issued material is unavailable")
                raise ApiError(404, "No original issued assignment")
            try:
                # Snapshot caller-owned nested dictionaries before validating.
                # The size limit applies before canonical parsing or responses.
                assignment_raw = _json_bytes(assignment)
                if len(assignment_raw) > MAX_RUNTIME_BYTES + MAX_OFFER_BYTES:
                    raise ValueError("Oversized original assignment")
                assignment = _canonical(assignment_raw, MAX_MATERIAL_BYTES)
                raw = encode_material(self._profile, assignment, self._csv_data)
                verified = validate_material(raw, profile=self._profile,
                    bootstrap_data=self._bootstrap_data, expected_binding=self._binding)
                if self._original_bytes is not None and self._original_bytes != raw:
                    raise ValueError("Original issued material changed")
                self._original_bytes = raw
                offer = _offer(raw, verified, self._binding, self._terminal())
                return raw, _validate_offer(offer, self._profile, self._binding)
            except (ValueError, TypeError, KeyError, RecursionError) as exc:
                raise ApiError(503, "Original issued material is invalid or changed") from exc

    def handle(self, path, body, *, headers, client_node_id):
        if path not in auth.JOB_PATHS:
            raise ApiError(404, "Unknown job endpoint")
        if headers.get_all("Content-Type", []) != [auth.CONTENT_TYPE]:
            raise ValueError("Job requests require one exact application/json type")
        signatures = headers.get_all(auth.AUTH_HEADER, [])
        encoded_limit = 4 * ((auth.MAX_ENVELOPE_BYTES + 2) // 3)
        if len(signatures) != 1 or len(signatures[0]) > encoded_limit:
            raise ApiError(403, "Unique bounded job signature required")
        try:
            envelope_raw = base64.b64decode(signatures[0], validate=True)
            if base64.b64encode(envelope_raw).decode("ascii") != signatures[0]:
                raise ValueError("Canonical signature encoding required")
            envelope = _canonical(envelope_raw, auth.MAX_ENVELOPE_BYTES)
            value = _canonical(body, auth.MAX_REQUEST_BYTES)
            if path == auth.OFFER_PATH:
                if value:
                    raise ValueError("Offer requests require an empty object")
            elif set(value) != MATERIAL_REQUEST_FIELDS:
                raise ValueError("Exact original material selector required")
            auth.verify_job_request(envelope, body, "POST", path, client_node_id,
                self._bootstrap, {k: self._binding[k] for k in auth.EXPECTED_JOB_FIELDS},
                self._nonce_db)
        except auth.JobAuthError as exc:
            raise ApiError(exc.status, "Job signature/source/scope rejected") from exc
        raw, offer = self._original()
        if path == auth.OFFER_PATH:
            return offer
        if value != {key: offer[key] for key in MATERIAL_REQUEST_FIELDS}:
            raise ApiError(409, "Requested material differs from original offer")
        return _json_object(raw)


class JobMaterialClient:
    """Bell-only pinned downloader; no retry, authority call, staging or launch.

    ``offer`` returns a hint, not ACTIVE evidence. ``material(offer)`` returns
    exact canonical bytes only after full local policy/original-permit checks.
    The supervisor independently obtains fresh signed status and controls its
    durable download/commit/accept lifecycle.
    """

    def __init__(self, *, profile, bootstrap_data, expected_binding, signer, verifier,
                 timeout=5.0):
        from shore_control.tailnet import valid_timeout
        from shore_control.tailnet_rpc import PinnedHTTPSConnection

        self._profile = validate_profile(profile)
        self._bootstrap = _local_bootstrap(self._profile, bootstrap_data)
        self._bootstrap_data = bytes(bootstrap_data)
        self._binding = _binding(expected_binding, self._bootstrap)
        row = self._bootstrap["workers"][BELL_ACTOR]
        network = self._bootstrap["network"]
        worker_network = {**network, "local_node_id": row["tailnet_node_id"],
            "local_address": network["nodes"][row["tailnet_node_id"]]["address"]}
        if verifier.network != worker_network:
            raise ValueError("Verifier differs from the fixed enrolled Bell source")
        self._context = {**{k: self._binding[k] for k in auth.EXPECTED_JOB_FIELDS},
            "actor_id": BELL_ACTOR, "device_id": row["device_id"],
            "node_id": row["tailnet_node_id"]}
        self._signer, self._verifier = signer, verifier
        self._timeout = valid_timeout(timeout)
        self._endpoint = self._bootstrap["collector"]["endpoint"]
        self._pin = self._bootstrap["collector"]["tls_sha256"]
        self._peer_node_id = network["local_node_id"]
        check = PinnedHTTPSConnection(self._endpoint, self._pin, verifier,
                                      self._peer_node_id, self._timeout)
        check.close()  # Configuration validation only, no network or key reads.

    def _request(self, path, value, response_limit):
        from shore_control.crypto import verify_object
        from shore_control.tailnet_rpc import PinnedHTTPSConnection

        body = _json_bytes(value)
        signed = auth.sign_job_request(self._signer, self._context, "POST", path, body)
        # Fail before connecting if an injected signer is not the fixed Bell.
        verify_object(self._bootstrap["workers"][BELL_ACTOR]["public_key"],
                      auth.AUTH_DOMAIN, signed)
        header = base64.b64encode(_json_bytes(signed)).decode("ascii")
        connection = response = None
        try:
            connection = PinnedHTTPSConnection(self._endpoint, self._pin, self._verifier,
                                              self._peer_node_id, self._timeout)
            connection._http_vsn, connection._http_vsn_str = 10, "HTTP/1.0"
            connection.request("POST", path, body=body, headers={
                "Host": urlsplit(self._endpoint).netloc, "Content-Type": auth.CONTENT_TYPE,
                "Content-Length": str(len(body)), "Connection": "close", auth.AUTH_HEADER: header})
            response = connection.getresponse()
            if response.version != 10 or 300 <= response.status < 400:
                raise ValueError("Only direct HTTP/1.0 responses are permitted")
            if response.headers.get_all("Content-Type", []) != [auth.CONTENT_TYPE]:
                raise ValueError("One exact application/json response type required")
            if response.headers.get_all("Connection", []) != ["close"]:
                raise ValueError("Explicit closed response required")
            length = _length(response.headers)
            if not 0 < length <= response_limit:
                raise ValueError("Job response exceeds its bounded limit")
            raw = response.read(length)
            if len(raw) != length:
                raise ValueError("Truncated job response")
            result = _canonical(raw, response_limit)
            if response.status != 200:
                if set(result) != {"error"} or type(result["error"]) is not str:
                    raise ValueError("Invalid job error response")
                raise ApiError(response.status, "Job collector rejected request")
            return raw
        except (OSError, http.client.HTTPException, ValueError, RecursionError) as exc:
            raise ConnectionError("Job collector unavailable or invalid response") from exc
        finally:
            try:
                if response is not None:
                    response.close()
            finally:
                if connection is not None:
                    connection.close()

    def offer(self):
        raw = self._request(auth.OFFER_PATH, {}, MAX_OFFER_BYTES)
        offer = _validate_offer(_json_object(raw), self._profile, self._binding)
        if offer["terminal_proof"] is not None:
            verify_release_proof(offer["terminal_proof"], self._bootstrap, self._binding)
        return offer

    def material(self, offer):
        offer = _validate_offer(offer, self._profile, self._binding)
        if offer["terminal_proof"] is not None:
            verify_release_proof(offer["terminal_proof"], self._bootstrap, self._binding)
        raw = self._request(auth.MATERIAL_PATH,
            {key: offer[key] for key in MATERIAL_REQUEST_FIELDS}, MAX_MATERIAL_BYTES)
        if len(raw) != offer["material_bytes"] or _digest(raw) != offer["material_sha256"]:
            raise ValueError("Downloaded material differs from original offer bytes")
        verified = validate_material(raw, profile=self._profile,
            bootstrap_data=self._bootstrap_data, expected_binding=self._binding)
        if _offer(raw, verified, self._binding, offer["terminal_proof"]) != offer:
            raise ValueError("Downloaded original permit/runtime differs from offer")
        return raw
