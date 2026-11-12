"""Durable signed RPC replay and reconciled, same-business nonce renewal.

Transport errors are never authority evidence. Every business call first replays
its current exact envelope (including after expiry); only an unsigned stale
challenge rejection opens reconciliation. Independent signed reads then check
the original job/operation and the method's permissions before a new generation
is committed. All envelopes, signed replies and reconciliation reads are kept.
Shore remains an optional dependency until a controlled call is actually made.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
from uuid import uuid4


def _json(value: Any) -> str:
    from shore_control.crypto import canonical_json

    return canonical_json(value).decode("utf-8")


def __getattr__(name):
    # Keep importing the adapter safe for historical, uncontrolled installations.
    if name != "VerifiedRpcError":
        raise AttributeError(name)
    from shore_control.authority import AuthorityError

    class VerifiedRpcError(AuthorityError):
        """A business refusal authenticated against this exact signed request."""

        def __init__(self, code, message, *, request, response):
            super().__init__(code, message)
            self.request = request
            self.response = response
            self.proof = {"request": request, "response": response}

    VerifiedRpcError.__module__ = __name__
    globals()[name] = VerifiedRpcError
    return VerifiedRpcError


def _blocked(message):
    from .control_authority import ControlError

    raise ControlError("RPC recovery blocked: " + message)


def _verified_response(client, request, envelope):
    """Same pin/binding checks as Shore Client.send, retaining the envelope."""
    from shore_control.authority import AuthorityError, digest
    from shore_control.crypto import verify_object

    response = verify_object(client.public_key, "response", envelope)
    expected = request["payload"]
    if (type(response.get("version")) is not int or response["version"] != 1
            or response.get("authority_id") != client.authority_id
            or response.get("request_id") != expected["request_id"]
            or response.get("actor_id") != client.actor_id
            or response.get("request_sha256") != digest(expected)
            or response.get("method") != expected["method"]
            or type(response.get("ok")) is not bool):
        raise AuthorityError("AUTH", "Response does not match the request")
    if response["ok"]:
        if "result" not in response:
            raise AuthorityError("AUTH", "Signed response omitted result")
    elif (not isinstance(response.get("error"), dict)
          or not isinstance(response["error"].get("code"), str)
          or not isinstance(response["error"].get("message"), str)):
        raise AuthorityError("AUTH", "Malformed signed business refusal")
    return response


def _result(request, envelope, response):
    if not response["ok"]:
        error = response["error"]
        cls = globals().get("VerifiedRpcError") or __getattr__("VerifiedRpcError")
        raise cls(error["code"], error["message"], request=request, response=envelope)
    return response["result"]


class SignedRpcJournal:
    """One immutable business identity, with durable signed-request generations.

    ``recovery_permit`` is the caller's already independently verified original
    signed permit. It is required for refreshing permit-ID-only worker/collector
    requests; old callers remain valid, but cannot silently guess their job.
    ``proof(key)`` returns the full signed request/response pair, or None.
    ``pending()`` retains the original public API and shows current generations.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS rpc_journal (
                key TEXT PRIMARY KEY, identity TEXT NOT NULL, method TEXT NOT NULL,
                params TEXT NOT NULL, request TEXT NOT NULL, result TEXT, error TEXT)""")
            db.execute("""CREATE TABLE IF NOT EXISTS rpc_generations (
                key TEXT NOT NULL, generation INTEGER NOT NULL, request TEXT NOT NULL,
                response TEXT, legacy_error TEXT, reconciliation TEXT,
                PRIMARY KEY(key,generation))""")
            db.execute("""CREATE TABLE IF NOT EXISTS rpc_recovery_permits (
                key TEXT PRIMARY KEY, permit TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS rpc_recovery_probes (
                probe_id TEXT PRIMARY KEY, key TEXT NOT NULL, generation INTEGER NOT NULL,
                request TEXT NOT NULL, response TEXT)""")
            # No original request/error is lost when opening a pre-R2 journal.
            db.execute("""INSERT OR IGNORE INTO rpc_generations
                (key,generation,request,legacy_error)
                SELECT key,0,request,error FROM rpc_journal""")
            # Old 'error' has no authenticated envelope: retry, never trust it.
            db.execute("""UPDATE rpc_journal SET error=NULL WHERE error IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM rpc_generations g
                    WHERE g.key=rpc_journal.key AND g.response IS NOT NULL)""")

    @contextmanager
    def transaction(self):
        with self._lock:
            db = sqlite3.connect(self.path, timeout=30)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    @staticmethod
    def _identity(client):
        return _json({"actor_id": client.actor_id, "authority_id": client.authority_id,
                      "fleet_id": client.fleet_id})

    @staticmethod
    def _permit(client, envelope):
        from shore_control.crypto import verify_object

        permit = verify_object(client.public_key, "permit", envelope)
        if (type(permit.get("version")) is not int or permit["version"] != 1
                or permit.get("authority_id") != client.authority_id
                or permit.get("fleet_id") != client.fleet_id):
            _blocked("permit does not match pinned authority/fleet")
        return permit

    def call(self, client: Any, method: str, params: dict, *, key: str,
             recovery_permit: dict | None = None) -> dict:
        from shore_control.authority import AuthorityError

        identity, encoded = self._identity(client), _json(params)
        if recovery_permit is not None:
            permit = self._permit(client, recovery_permit)
            if permit.get("permit_id") != params.get("permit_id"):
                _blocked("recovery permit differs from immutable business permit")
        with self.transaction() as db:
            row = db.execute("SELECT * FROM rpc_journal WHERE key=?", (key,)).fetchone()
            if row is None:
                request = client.prepare(method, params)
                db.execute("INSERT INTO rpc_journal VALUES(?,?,?,?,?,NULL,NULL)",
                           (key, identity, method, encoded, _json(request)))
                db.execute("INSERT INTO rpc_generations(key,generation,request) VALUES(?,0,?)",
                           (key, _json(request)))
            elif (row["identity"], row["method"], row["params"]) != (identity, method, encoded):
                _blocked("journal key reused with changed identity or content")
            if recovery_permit is not None:
                previous = db.execute("SELECT permit FROM rpc_recovery_permits WHERE key=?", (key,)).fetchone()
                if previous is not None and previous["permit"] != _json(recovery_permit):
                    _blocked("original recovery permit cannot change")
                db.execute("INSERT OR IGNORE INTO rpc_recovery_permits VALUES(?,?)",
                           (key, _json(recovery_permit)))

        # At most one fresh mutation per call. A second interruption/stale nonce
        # remains durably pending for the next restart, never a busy retry loop.
        refreshed = False
        while True:
            with self.transaction() as db:
                row = db.execute("SELECT * FROM rpc_journal WHERE key=?", (key,)).fetchone()
                generation = db.execute("SELECT * FROM rpc_generations WHERE key=? ORDER BY generation DESC LIMIT 1",
                                        (key,)).fetchone()
                request = json.loads(generation["request"])
                if generation["response"] is not None:
                    envelope = json.loads(generation["response"])
                    return _result(request, envelope, _verified_response(client, request, envelope))
                if row["result"] is not None:
                    # Backward-compatible historical success, not current proof.
                    return json.loads(row["result"])
            try:
                envelope = client.transport.handle(request)
                response = _verified_response(client, request, envelope)
            except AuthorityError as exc:
                if exc.code != "CHALLENGE_STALE" or refreshed:
                    raise
                evidence = self._reconcile(client, method, params, key, generation["generation"])
                # A testable crash boundary: successful reads are already durable.
                self._after_reconcile(key)
                with self.transaction() as db:
                    current = db.execute("SELECT * FROM rpc_generations WHERE key=? ORDER BY generation DESC LIMIT 1",
                                         (key,)).fetchone()
                    if current["generation"] == generation["generation"] and current["response"] is None:
                        fresh = client.prepare(method, params)
                        # Identity/method/business params must be byte-equivalent;
                        # only nonce, request_id and the signature may change.
                        old_payload, new_payload = request["payload"], fresh["payload"]
                        fields = set(old_payload) - {"nonce", "request_id"}
                        if set(old_payload) != set(new_payload) or any(old_payload[k] != new_payload[k] for k in fields):
                            _blocked("request preparation changed immutable signed business fields")
                        db.execute("INSERT INTO rpc_generations(key,generation,request,reconciliation) VALUES(?,?,?,?)",
                                   (key, current["generation"] + 1, _json(fresh), _json(evidence)))
                        db.execute("UPDATE rpc_journal SET request=?,result=NULL,error=NULL WHERE key=?",
                                   (_json(fresh), key))
                self._after_refresh(key)
                refreshed = True
                continue
            self._store_response(key, generation["generation"], envelope, response)
            return _result(request, envelope, response)

    def fresh_read(self, client: Any, method: str, params: dict) -> dict:
        """Recover identical interrupted reads, then request a current snapshot.

        Replaying a committed read can return a historical AWAITING result, so
        recovery is bookkeeping, not the return value. Every original envelope,
        authenticated response and any stale-nonce reconciliation stays in the
        journal. A transport/signature/recovery failure blocks the new read;
        an authenticated historical refusal is retained but is not today's
        snapshot. This helper deliberately cannot issue mutating RPCs.
        """
        if method not in {"status", "get_operation"}:
            raise ValueError("Fresh reads support status/get_operation only")
        identity, encoded = self._identity(client), _json(params)
        with self._lock:
            with self.transaction() as db:
                pending = db.execute(
                    "SELECT key FROM rpc_journal WHERE identity=? AND method=? "
                    "AND params=? AND result IS NULL AND error IS NULL ORDER BY rowid",
                    (identity, method, encoded),
                ).fetchall()
            verified_error = globals().get("VerifiedRpcError") or __getattr__("VerifiedRpcError")
            for row in pending:
                try:
                    self.call(client, method, params, key=row["key"])
                except verified_error:
                    # call() persisted the authenticated historical refusal.
                    # It authorizes no work and must not mask a newer snapshot.
                    pass
            return self.call(client, method, params, key=method + ":" + uuid4().hex)

    def _store_response(self, key, generation, envelope, response):
        with self.transaction() as db:
            db.execute("UPDATE rpc_generations SET response=? WHERE key=? AND generation=?",
                       (_json(envelope), key, generation))
            latest = db.execute("SELECT MAX(generation) FROM rpc_generations WHERE key=?", (key,)).fetchone()[0]
            if latest == generation:
                db.execute("UPDATE rpc_journal SET result=?,error=? WHERE key=?",
                           (_json(response["result"]) if response["ok"] else None,
                            None if response["ok"] else _json(response["error"]), key))

    def _after_reconcile(self, key):
        """Crash-injection boundary after durable signed reconciliation reads."""

    def _after_refresh(self, key):
        """Crash-injection boundary after durable generation, before delivery."""

    def _probe(self, client, method, params, key, generation):
        # Reads only. Every interrupted probe is retained, but it never blocks a
        # future independently signed read. No recursive stale-write recovery.
        if method not in {"status", "get_operation"}:
            _blocked("reconciliation probes must be read-only")
        request = client.prepare(method, params)
        probe_id = uuid4().hex
        with self.transaction() as db:
            db.execute("INSERT INTO rpc_recovery_probes VALUES(?,?,?,?,NULL)",
                       (probe_id, key, generation, _json(request)))
        envelope = client.transport.handle(request)
        response = _verified_response(client, request, envelope)
        with self.transaction() as db:
            db.execute("UPDATE rpc_recovery_probes SET response=? WHERE probe_id=?", (_json(envelope), probe_id))
        proof = {"request": request, "response": envelope}
        return response, proof

    def _reconcile(self, client, method, params, key, generation):
        """Authenticate a current snapshot, then enforce method-specific scope.

        The reads do not prove an original request was uncommitted: concurrent
        original delivery is possible. Safety additionally depends on Shore's
        immutable operation/permit and receipt idempotence, and rechecking all
        permissions transactionally when it receives the refreshed request.
        """
        response, proof = self._probe(client, "status", {}, key, generation)
        status = _result(proof["request"], proof["response"], response)
        if not isinstance(status, dict) or status.get("fleet_id") != client.fleet_id:
            _blocked("status is for a different fleet")
        evidence = [proof]
        job = status.get("bound_job")
        if method == "status":
            return evidence
        if method == "get_operation":
            response, proof = self._probe(client, method, params, key, generation)
            evidence.append(proof)
            # An authenticated refusal is a valid read result, never permission.
            if not response["ok"] and response["error"]["code"] != "UNKNOWN_PERMIT":
                _result(proof["request"], proof["response"], response)
            return evidence
        if method == "acquire":
            # An empty fleet is permission to retry the *same* acquisition, not
            # proof that this job ID has never existed. The authority still
            # checks enrolled controller, historical job and session-key reuse.
            # Never adopt or displace any currently bound job from a snapshot.
            if job is not None or client.actor_id.startswith("session:"):
                _blocked("acquire requires an empty fleet; reconcile the existing binding explicitly")
            return evidence
        if not isinstance(job, dict) or job.get("fleet_id") != client.fleet_id:
            _blocked("original job is no longer bound")

        def binding_matches(values, *, session=True):
            names = ("job_id", "epoch", "host_id", "session_id") if session else ("job_id", "epoch", "host_id")
            if any(values.get(name) != job.get(name) for name in names):
                _blocked("original work binding has been fenced")

        def current_session():
            if client.actor_id != "session:" + job["session_id"]:
                _blocked("current bound collector session required")

        if method == "resume_bound_job":
            binding_matches(params, session=False)
            if params.get("previous_session_id") != job.get("session_id"):
                _blocked("resume previous session changed; reconcile that lifecycle explicitly")
            if params.get("mode") not in {"recover", "resume"}:
                _blocked("unsupported resume mode")
            if params["mode"] == "resume" and (job["phase"] != "PAUSED_BOUND"
                    or job.get("checkpoint") is None or status.get("unresolved_count") != 0):
                _blocked("explicit resume requires a reconciled paused job")
            self._controller_provenance(client, job, evidence)
            return evidence

        if method == "validate_dispatch":
            binding_matches(params)
            if job["phase"] != "ACTIVE" or client.actor_id.startswith("session:"):
                _blocked("new dispatch requires original controller and ACTIVE job")
            operation = params["operation"]
            response, proof = self._probe(client, "get_operation",
                {"epoch": params["epoch"], "operation_id": operation["operation_id"]}, key, generation)
            evidence.append(proof)
            if response["ok"]:
                permit = self._permit(client, response["result"]["permit"])
                if any(permit.get(k) != v for k, v in {**operation, **{k: params[k] for k in ("job_id", "epoch", "host_id", "session_id")}}.items()):
                    _blocked("existing operation differs from original dispatch")
            elif response["error"]["code"] != "UNKNOWN_PERMIT":
                _result(proof["request"], proof["response"], response)
            self._controller_provenance(client, job, evidence)
            return evidence

        if method in {"accept_permit", "worker_report", "controller_ack"}:
            with self.transaction() as db:
                saved = db.execute("SELECT permit FROM rpc_recovery_permits WHERE key=?", (key,)).fetchone()
            if saved is None:
                _blocked("original signed recovery_permit is required for " + method)
            permit = self._permit(client, json.loads(saved["permit"]))
            binding_matches(permit, session=False)
            response, proof = self._probe(client, "get_operation",
                {"epoch": permit["epoch"], "operation_id": permit["operation_id"]}, key, generation)
            evidence.append(proof)
            operation = _result(proof["request"], proof["response"], response)
            if operation.get("permit") != json.loads(saved["permit"]):
                _blocked("authority operation changed the original signed permit")
            if method == "accept_permit":
                fields = ("permit_id", "operation_id", "run_id", "attempt_id", "action", "input_sha256", "device_id")
                if client.actor_id != permit["worker_id"] or any(params.get(k) != permit.get(k) for k in fields):
                    _blocked("acceptance actor/descriptor differs from original permit")
                if operation.get("worker_receipt") is not None or operation.get("controller_receipt") is not None or operation.get("status") == "RESOLVED":
                    worker_receipt = operation.get("worker_receipt")
                    collector_receipt = operation.get("controller_receipt")
                    # A cancelled wake's old, never-delivered acceptance can
                    # outlive its nonce. The same signed business call may be
                    # renewed only after BOTH parties permanently closed it as
                    # not_started. Shore checks WORK_CLOSED before all accept /
                    # phase paths, so this can obtain only a signed refusal,
                    # never a new permission. Keep the terminal signed probes.
                    if (permit["action"] == "wake" and operation.get("status") == "RESOLVED"
                            and isinstance(worker_receipt, dict)
                            and set(worker_receipt) == {"outcome", "durable", "evidence_sha256"}
                            and worker_receipt == collector_receipt
                            and worker_receipt["outcome"] == "not_started"
                            and worker_receipt["durable"] is True
                            and isinstance(worker_receipt["evidence_sha256"], str)
                            and len(worker_receipt["evidence_sha256"]) == 64
                            and set(worker_receipt["evidence_sha256"]) <= set("0123456789abcdef")):
                        return evidence
                    _blocked("receipted work cannot be accepted")
                if operation.get("acceptance") is None:
                    binding_matches(permit)
                    if job["phase"] != "ACTIVE":
                        _blocked("unaccepted work cannot start after the drain barrier")
            else:
                if method == "worker_report" and client.actor_id != permit["worker_id"]:
                    _blocked("receipt actor differs from original worker")
                if method == "controller_ack":
                    current_session()
                if params.get("outcome") != "not_started" and operation.get("acceptance") is None:
                    _blocked("execution receipt requires accepted work")
                receipt = {k: params[k] for k in ("outcome", "evidence_sha256", "durable")}
                if any(operation.get(k) not in (None, receipt) for k in ("worker_receipt", "controller_receipt")):
                    _blocked("terminal receipt differs from durable authority evidence")
            return evidence

        binding_matches(params)
        if method in {"renew", "register_run"}:
            if job["phase"] != "ACTIVE":
                _blocked("renewal/registration cannot clear a drain barrier")
            self._controller_provenance(client, job, evidence)
        elif method == "begin_drain":
            if client.actor_id != "session:" + job["session_id"]:
                self._controller_provenance(client, job, evidence)
        elif method == "finish_drain":
            current_session()
            if job["phase"] not in {"DRAINING", "PAUSED_BOUND"} or status.get("unresolved_count") != 0:
                _blocked("finish_drain requires reconciled drained work")
        elif method == "release_job":
            self._controller_provenance(client, job, evidence)
            if (job["phase"] != "PAUSED_BOUND" or status.get("unresolved_count") != 0
                    or (job.get("checkpoint") or {}).get("checkpoint_sha256") != params.get("checkpoint_sha256")):
                _blocked("release requires the same reconciled checkpoint")
        else:
            _blocked("no safe refresh rule for " + method)
        return evidence

    def _controller_provenance(self, client, job, evidence):
        # Status deliberately does not expose controller_id. Recover that role
        # from an authenticated original acquisition, including legacy journals
        # whose cached result predates signed-response capture. Exact replay is
        # safe even after its nonce expires, and never renews a lease.
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM rpc_journal WHERE identity=? AND method='acquire'",
                              (self._identity(client),)).fetchall()
        for row in rows:
            params = json.loads(row["params"])
            if params.get("job_id") != job["job_id"] or params.get("host_id") != job["host_id"]:
                continue
            request = json.loads(row["request"])
            envelope = client.transport.handle(request)
            response = _verified_response(client, request, envelope)
            acquired = _result(request, envelope, response)
            if (acquired.get("job_id"), acquired.get("epoch"), acquired.get("host_id")) != (job["job_id"], job["epoch"], job["host_id"]):
                _blocked("controller acquisition proves a different job")
            with self.transaction() as db:
                generation = db.execute("SELECT generation FROM rpc_generations WHERE key=? AND request=?",
                                        (row["key"], row["request"])).fetchone()[0]
            self._store_response(row["key"], generation, envelope, response)
            evidence.append({"request": request, "response": envelope})
            return
        _blocked("signed original controller acquisition provenance is unavailable")

    def proof(self, key: str) -> dict | None:
        with self.transaction() as db:
            row = db.execute("SELECT request,response FROM rpc_generations WHERE key=? ORDER BY generation DESC LIMIT 1",
                             (key,)).fetchone()
        if row is None or row["response"] is None:
            return None
        return {"request": json.loads(row["request"]), "response": json.loads(row["response"])}

    def pending(self) -> list[dict]:
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM rpc_journal WHERE result IS NULL AND error IS NULL ORDER BY rowid").fetchall()
        return [{"key": row["key"], "method": row["method"], "params": json.loads(row["params"]),
                 "request": json.loads(row["request"])} for row in rows]
