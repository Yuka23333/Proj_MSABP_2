"""Project adapter for the single pinned Shore authority (not an authority).

Shore is an optional, explicitly supplied dependency.  No keys, transports,
registry defaults, server, or credentials are discovered here.  The SQLite file
is a project-side RPC/evidence journal; it never grants control authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
from uuid import uuid4

from .control_rpc import SignedRpcJournal


class ControlError(RuntimeError):
    """Fail-closed project control or reconciliation error."""


def _json(value: Any) -> str:
    # Import only when control mode is explicitly used; historical runs need
    # neither Shore nor its cryptography dependency installed.
    from shore_control.crypto import canonical_json

    return canonical_json(value).decode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ControlAuthority:
    """Injected controller/session clients plus persistent project provenance.

    Controller signs every new work permit.  Session can collect/drain only.
    Caller retains session key material securely across process restart; this
    journal stores public bindings and signed requests, never private keys.
    """

    def __init__(self, journal_path: str | Path, *, controller_client: Any,
                 session_client: Any, worker_clients: dict | None = None,
                 worker_registrations: dict | None = None):
        from shore_control.registry import validate_workers

        self.journal = SignedRpcJournal(journal_path)
        self.controller = controller_client
        self.session = session_client
        self.worker_clients = dict(worker_clients or {})
        self.registrations = validate_workers(worker_registrations or {})
        self._lock = threading.RLock()
        clients = [self.controller, self.session, *self.worker_clients.values()]
        if any((c.authority_id, c.fleet_id) !=
               (self.controller.authority_id, self.controller.fleet_id) for c in clients):
            raise ControlError("All clients must pin the same authority and fleet")
        with self.journal.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS control_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS control_runs (run_id TEXT PRIMARY KEY,registration TEXT NOT NULL,state_database TEXT,plan_path TEXT);
                CREATE TABLE IF NOT EXISTS control_operations (operation_id TEXT PRIMARY KEY,intent TEXT NOT NULL,descriptor TEXT NOT NULL,permit TEXT);
                CREATE TABLE IF NOT EXISTS control_evidence (permit_id TEXT PRIMARY KEY,evidence TEXT NOT NULL,root TEXT NOT NULL,aliases TEXT,evidence_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS control_evaluations (evaluation_id TEXT PRIMARY KEY,run_id TEXT NOT NULL,case_id TEXT NOT NULL,record TEXT NOT NULL,UNIQUE(run_id,case_id));
            """)
            if "plan_path" not in {r[1] for r in db.execute("PRAGMA table_info(control_runs)")}:
                db.execute("ALTER TABLE control_runs ADD COLUMN plan_path TEXT")
            operation_columns = {r[1] for r in db.execute("PRAGMA table_info(control_operations)")}
            if "state" not in operation_columns:
                db.execute("ALTER TABLE control_operations ADD COLUMN state TEXT NOT NULL DEFAULT 'intent'")
                db.execute("UPDATE control_operations SET state='issued' WHERE permit IS NOT NULL")
            if "unissued_proof_json" not in operation_columns:
                db.execute("ALTER TABLE control_operations ADD COLUMN unissued_proof_json TEXT")
        pins = {"authority_id": self.controller.authority_id,
                "fleet_id": self.controller.fleet_id,
                "controller_id": self.controller.actor_id,
                "worker_registrations": self.registrations}
        old_pins = self._get("pins")
        if old_pins is not None and old_pins != pins:
            raise ControlError("Pinned authority/controller/worker registry changed")
        self._set("pins", pins)
        binding = self._get("job")
        if binding and binding["phase"] != "RELEASED" and self.session.actor_id != "session:" + binding["session_id"]:
            raise ControlError("Inject the currently bound session client for journal recovery")

    def _get(self, key: str, default=None):
        with self.journal.transaction() as db:
            row = db.execute("SELECT value FROM control_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def _set(self, key: str, value: Any):
        with self.journal.transaction() as db:
            db.execute("INSERT INTO control_meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       (key, _json(value)))

    @property
    def binding(self) -> dict:
        job = self._get("job")
        if job is None:
            raise ControlError("No persisted job binding; explicitly acquire first")
        return {name: job[name] for name in ("job_id", "epoch", "host_id", "session_id")}

    @property
    def authority_public_key(self):
        return self.controller.public_key

    @property
    def job(self) -> dict:
        job = self._get("job")
        if job is None:
            raise ControlError("No acquired job")
        return job

    def rpc(self, role: str, method: str, params: dict, *, key: str,
            recovery_permit: dict | None = None) -> dict:
        client = {"controller": self.controller, "session": self.session,
                  **self.worker_clients}.get(role)
        if client is None:
            raise ControlError("No explicitly injected client for " + role)
        return self.journal.call(client, method, params, key=key,
                                 recovery_permit=recovery_permit)

    def _fresh(self, role: str, method: str, params: dict) -> dict:
        # Reconcile any previous uncertain identical request before asking for
        # a fresh signed snapshot. Never treat its historical result as current.
        actor = self.controller.actor_id if role == "controller" else self.session.actor_id
        for pending in self.journal.pending():
            if (pending["method"] == method and pending["params"] == params
                    and pending["request"]["payload"]["actor_id"] == actor):
                self.rpc(role, method, params, key=pending["key"])
        return self.rpc(role, method, params, key=method + ":" + uuid4().hex)

    def acquire(self, *, job_id: str, host_id: str, session_id: str,
                session_public_key: str, project_id: str, collector_url: str,
                data_root: str, job_spec: dict) -> dict:
        from shore_control.job_contract import validate_job_spec

        validate_job_spec(job_spec)
        if self.session.actor_id != "session:" + session_id:
            raise ControlError("Session client does not match acquire session_id")
        params = dict(job_id=job_id, host_id=host_id, session_id=session_id,
                      session_public_key=session_public_key, project_id=project_id,
                      collector_url=collector_url, data_root=data_root, job_spec=job_spec)
        old = self._get("acquire")
        if old is not None and old != params:
            raise ControlError("This project journal is bound to one immutable complete job")
        self._set("acquire", params)
        result = self.rpc("controller", "acquire", params, key="acquire:" + job_id)
        # Do not roll back a newer drain/resume snapshot on historical replay.
        if self._get("job") is None:
            self._set("job", result)
            self._set("local_drain", False)
        return result

    def status(self) -> dict:
        return self._fresh("session", "status", {})

    def require_new_work(self) -> dict:
        with self._lock:
            if self._get("local_drain", False):
                raise ControlError("New work is locally draining; explicit resume required")
            try:
                result = self._fresh("controller", "renew", self.binding)
            except Exception:
                # No cached ACTIVE response, local signer, alternate identity or
                # lease grace can stand in for this failed controller proof.
                try:
                    self.begin_drain("Controller renewal unavailable; new work stopped")
                except Exception:
                    # begin_drain persists local stop before attempting RPC.
                    # Its exact signed request, if prepared, remains pending.
                    pass
                raise
            if result["phase"] != "ACTIVE":
                raise ControlError("New work requires a fresh ACTIVE controller response")
            self._set("job", result)
            return result

    def poll_control(self) -> dict:
        """Renew while ACTIVE; monitor failures drain without killing work.

        Call at an interval shorter than the pinned authority lease.  When
        already draining, only retry an uncertain drain RPC; never renew or
        resume.  The caller may log the propagated failure and keep collection
        and existing work heartbeats running.
        """
        with self._lock:
            if self._get("local_drain", False):
                if self.job["phase"] == "ACTIVE":
                    return self.begin_drain(self._get(
                        "local_drain_reason", "Controller renewal unavailable; new work stopped"))
                return self.job
            if self.job["phase"] != "ACTIVE":
                return self.job
            return self.require_new_work()

    def register_run(self, run_id: str, original_plan_sha256: str,
                     input_manifest_sha256: str) -> dict:
        registration = dict(run_id=run_id, original_plan_sha256=original_plan_sha256,
                            input_manifest_sha256=input_manifest_sha256)
        with self.journal.transaction() as db:
            old = db.execute("SELECT registration FROM control_runs WHERE run_id=?", (run_id,)).fetchone()
            if old and json.loads(old["registration"]) != registration:
                raise ControlError("Registered immutable run changed")
            if old:
                return json.loads(old["registration"])
        result = self.rpc("controller", "register_run", {**self.binding, **registration},
                          key="register:" + str(self.binding["epoch"]) + ":" + run_id)
        with self.journal.transaction() as db:
            db.execute("INSERT OR IGNORE INTO control_runs(run_id,registration) VALUES(?,?)", (run_id, _json(result)))
        return result

    def descriptor_for_attempt(self, intent: dict) -> dict:
        from shore_control.registry import registration_sha256

        operation = {name: intent[name] for name in
                     ("operation_id", "run_id", "worker_id", "attempt_id", "input_sha256")}
        operation.update(action=intent.get("action", "attempt"), action_context=intent.get("action_context"))
        with self.journal.transaction() as db:
            old = db.execute("SELECT intent,descriptor FROM control_operations WHERE operation_id=?",
                             (intent["operation_id"],)).fetchone()
            if old:
                if json.loads(old["intent"]) != intent:
                    raise ControlError("Operation intent changed")
                return json.loads(old["descriptor"])
            if db.execute("SELECT 1 FROM control_runs WHERE run_id=?", (intent["run_id"],)).fetchone() is None:
                raise ControlError("Run must be registered before preparing a descriptor")
        registration = self.registrations.get(intent["worker_id"])
        if registration is None:
            raise ControlError("Worker requires an explicit pinned registration")
        if operation["action"] not in registration["allowed_actions"]:
            raise ControlError("Worker registration forbids requested action")
        expected = {**operation, **self.binding, "fleet_id": self.controller.fleet_id,
                **self.job["scope"], "device_id": registration["device_id"],
                "worker_role": registration["role"],
                "worker_registration_sha256": registration_sha256(registration)}
        if "descriptor" in intent:
            persisted = intent["descriptor"]
            if not isinstance(persisted, dict) or set(persisted) != set(expected):
                raise ControlError("Persisted descriptor fields changed")
            # An old session may only recover the original operation; the
            # authority's get_operation supplies the signed recovery proof.
            if any(persisted[k] != v for k, v in expected.items() if k != "session_id"):
                raise ControlError("Persisted descriptor differs from independent job/input binding")
            return json.loads(_json(persisted))
        return expected

    def issue_attempt(self, intent: dict) -> dict:
        if intent.get("action", "attempt") != "attempt":
            raise ControlError("issue_attempt only issues attempt actions")
        return self.issue_operation(intent)

    def issue_operation(self, intent: dict) -> dict:
        from shore_control.permits import verify_permit

        with self._lock:
            descriptor = self.descriptor_for_attempt(intent)
            operation_id = intent["operation_id"]
            draining = self._get("local_drain", False)
            approved_units = self.job["scope"]["job_spec"]["approved_units"]
            if descriptor["action"] == "attempt" and (
                    type(intent.get("case_id")) is not str or not intent["case_id"]):
                raise ControlError("Project evaluation dispatch requires an explicit case_id")
            with self.journal.transaction() as db:
                cancelled = db.execute("SELECT intent,unissued_proof_json FROM control_operations WHERE state='cancelled_unissued'").fetchall()
            for prior in cancelled:
                if prior["unissued_proof_json"] is None:
                    raise ControlError("Cancelled operation cannot release budget without signed proof")
                self.verify_unissued_intent(json.loads(prior["intent"]), json.loads(prior["unissued_proof_json"]))
            with self.journal.transaction() as db:
                old = db.execute("SELECT * FROM control_operations WHERE operation_id=?", (operation_id,)).fetchone()
                if old is None:
                    if draining and "descriptor" not in intent:
                        raise ControlError("New operation forbidden by local drain")
                    if descriptor["action"] == "attempt":
                        reserved = {}
                        for row in db.execute("SELECT intent,descriptor FROM control_operations WHERE state!='cancelled_unissued'"):
                            prior = json.loads(row["descriptor"])
                            if prior["action"] != "attempt":
                                continue
                            previous_intent = json.loads(row["intent"])
                            if not previous_intent.get("case_id"):
                                raise ControlError("Historical attempt has no evaluation budget identity")
                            unit = (prior["run_id"], previous_intent["case_id"])
                            reserved[unit] = prior["input_sha256"]
                        unit = (descriptor["run_id"], intent["case_id"])
                        if unit in reserved and reserved[unit] != descriptor["input_sha256"]:
                            raise ControlError("Retry changed its reserved evaluation input")
                        if unit not in reserved and len(reserved) >= approved_units:
                            raise ControlError("Full-job approved evaluation budget is already reserved")
                    db.execute("INSERT INTO control_operations(operation_id,intent,descriptor,permit) VALUES(?,?,?,NULL)",
                               (operation_id, _json(intent), _json(descriptor)))
                elif old["state"] == "cancelled_unissued":
                    raise ControlError("Permanently fenced unissued operation cannot be dispatched again")
                elif old["permit"]:
                    permit = json.loads(old["permit"])
                    verify_permit(self.controller.public_key, permit, descriptor)
                    return {"descriptor": descriptor, "permit": permit}
                signed_request_exists = db.execute("SELECT 1 FROM rpc_journal WHERE key=?",
                                                   ("dispatch:" + operation_id,)).fetchone() is not None
            # Old-session reconciliation must not regenerate a descriptor with
            # the replacement session. Query the original operation instead.
            if (descriptor["session_id"] != self.binding["session_id"]
                    or (draining and not signed_request_exists)):
                permit = self.get_operation(operation_id)["permit"]
            else:
                operation = {name: descriptor[name] for name in
                             ("operation_id", "run_id", "worker_id", "attempt_id", "action", "input_sha256", "action_context")}
                params = {name: descriptor[name] for name in ("job_id", "epoch", "host_id", "session_id")}
                permit = self.rpc("controller", "validate_dispatch", {**params, "operation": operation},
                                  key="dispatch:" + operation_id)
            verify_permit(self.controller.public_key, permit, descriptor)
            with self.journal.transaction() as db:
                db.execute("UPDATE control_operations SET permit=?,state='issued' WHERE operation_id=?", (_json(permit), operation_id))
            return {"descriptor": descriptor, "permit": permit}

    def _verify_unissued_rpc(self, pair: dict, *, method: str, params: dict) -> dict:
        """Authenticate a retained response against its exact request and pins."""
        from shore_control.crypto import verify_object

        if not isinstance(pair, dict) or set(pair) != {"request", "response"}:
            raise ControlError("Unissued proof requires signed request/response pairs")
        try:
            request = pair["request"]["payload"]
            response = verify_object(self.controller.public_key, "response", pair["response"])
            expected = {"fleet_id": self.controller.fleet_id, **params}
            if (request.get("version") != 1 or request.get("authority_id") != self.controller.authority_id
                    or request.get("method") != method or request.get("params") != expected
                    or not str(request.get("actor_id", "")).startswith("session:")
                    or response.get("version") != 1 or response.get("authority_id") != self.controller.authority_id
                    or response.get("actor_id") != request["actor_id"]
                    or response.get("request_id") != request["request_id"]
                    or response.get("request_sha256") != _digest(request)
                    or response.get("method") != method or type(response.get("ok")) is not bool):
                raise ControlError("Signed unissued proof differs from its pinned request")
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlError("Invalid signed unissued proof") from exc
        return response

    def verify_unissued_intent(self, intent: dict, proof: dict) -> dict:
        """Recheck durable fencing/absence evidence without granting new work.

        Shore never reactivates a drained session: both recover and resume use
        globally fresh session identifiers. A signed drain snapshot or rotated
        session, followed by a fresh signed absence, therefore closes the old
        descriptor permanently. The ordered journal retains the causal reads.
        """
        descriptor = self.descriptor_for_attempt(intent)
        fields = {"schema", "schema_version", "operation_id", "descriptor_sha256",
                  "fence_key", "absence_key", "fence", "absence"}
        if (not isinstance(proof, dict) or set(proof) != fields
                or proof["schema"] != "msabp.cancelled-unissued" or proof["schema_version"] != 1
                or proof["operation_id"] != descriptor["operation_id"]
                or proof["descriptor_sha256"] != _digest(descriptor)):
            raise ControlError("Unissued cancellation proof does not match its descriptor")
        fence = self._verify_unissued_rpc(proof["fence"], method="status", params={})
        absence = self._verify_unissued_rpc(proof["absence"], method="get_operation", params={
            "epoch": descriptor["epoch"], "operation_id": descriptor["operation_id"]})
        bound = fence.get("result", {}).get("bound_job") if fence["ok"] else None
        if (not isinstance(bound, dict)
                or any(bound.get(k) != descriptor[k] for k in ("fleet_id", "job_id", "epoch", "host_id"))
                or bound.get("scope") != {k: descriptor[k] for k in self.job["scope"]}
                or (bound.get("session_id") == descriptor["session_id"]
                    and bound.get("phase") not in {"DRAINING", "PAUSED_BOUND"})):
            raise ControlError("Unissued operation lacks a permanent session fence")
        actor = "session:" + bound["session_id"]
        if (proof["fence"]["request"]["payload"]["actor_id"] != actor
                or proof["absence"]["request"]["payload"]["actor_id"] != actor
                or absence["ok"] or absence.get("error", {}).get("code") != "UNKNOWN_PERMIT"):
            raise ControlError("Unissued operation lacks a signed UNKNOWN_PERMIT proof")
        with self.journal.transaction() as db:
            operation = db.execute("SELECT * FROM control_operations WHERE operation_id=?",
                                   (descriptor["operation_id"],)).fetchone()
            fence_row = db.execute("SELECT rowid FROM rpc_journal WHERE key=?", (proof["fence_key"],)).fetchone()
            absence_row = db.execute("SELECT rowid FROM rpc_journal WHERE key=?", (proof["absence_key"],)).fetchone()
        if (operation is None or operation["permit"] is not None
                or (operation["unissued_proof_json"] is not None
                    and json.loads(operation["unissued_proof_json"]) != proof)):
            raise ControlError("Unissued proof conflicts with durable operation history")
        if (fence_row is None or absence_row is None or fence_row[0] >= absence_row[0]
                or self.journal.proof(proof["fence_key"]) != proof["fence"]
                or self.journal.proof(proof["absence_key"]) != proof["absence"]):
            raise ControlError("Unissued proof lacks ordered durable signed RPC history")
        return json.loads(_json(proof))

    def reconcile_unissued_intent(self, intent: dict) -> dict:
        """Cancel only a permanently fenced descriptor with signed fresh absence.

        No response/transport exception, cached absence or local drain flag is
        evidence. A real permit, even one whose reply was lost, stays unresolved
        until the existing worker and collector terminal receipts arrive.
        """
        from .control_rpc import VerifiedRpcError

        with self._lock:
            descriptor = self.descriptor_for_attempt(intent)
            operation_id = descriptor["operation_id"]
            if not self._get("local_drain", False):
                raise ControlError("Unissued reconciliation requires explicit drain or recovery")
            with self.journal.transaction() as db:
                row = db.execute("SELECT * FROM control_operations WHERE operation_id=?", (operation_id,)).fetchone()
                if row is None:
                    if "descriptor" not in intent:
                        raise ControlError("Unissued reconciliation requires a persisted descriptor")
                    db.execute("INSERT INTO control_operations(operation_id,intent,descriptor) VALUES(?,?,?)",
                               (operation_id, _json(intent), _json(descriptor)))
                elif row["permit"] is not None:
                    raise ControlError("Existing permit cannot be cancelled as unissued")
                elif row["state"] == "cancelled_unissued":
                    saved_proof = json.loads(row["unissued_proof_json"])
                else:
                    saved_proof = None
            if row is not None and row["state"] == "cancelled_unissued":
                return self.verify_unissued_intent(intent, saved_proof)
            fence_key = "unissued-fence:" + uuid4().hex
            status = self.rpc("session", "status", {}, key=fence_key)
            bound = status.get("bound_job")
            if (not isinstance(bound, dict)
                    or any(bound.get(k) != descriptor[k] for k in ("fleet_id", "job_id", "epoch", "host_id"))
                    or (bound.get("session_id") == descriptor["session_id"]
                        and bound.get("phase") not in {"DRAINING", "PAUSED_BOUND"})):
                raise ControlError("Unissued reconciliation requires an authority-proven session fence")
            absence_key = "unissued-absence:" + uuid4().hex
            try:
                self.rpc("session", "get_operation", {"epoch": descriptor["epoch"], "operation_id": operation_id},
                         key=absence_key)
            except VerifiedRpcError as exc:
                if exc.code != "UNKNOWN_PERMIT":
                    raise
                absence = exc.proof
            else:
                raise ControlError("Existing authority permit cannot be cancelled as unissued")
            proof = {"schema": "msabp.cancelled-unissued", "schema_version": 1,
                     "operation_id": operation_id, "descriptor_sha256": _digest(descriptor),
                     "fence_key": fence_key, "absence_key": absence_key,
                     "fence": self.journal.proof(fence_key), "absence": absence}
            self.verify_unissued_intent(intent, proof)
            with self.journal.transaction() as db:
                saved = db.execute("SELECT * FROM control_operations WHERE operation_id=?", (operation_id,)).fetchone()
                if saved["permit"] is not None:
                    raise ControlError("Concurrent permit cannot be cancelled as unissued")
                if saved["unissued_proof_json"] is not None:
                    proof = json.loads(saved["unissued_proof_json"])
                else:
                    db.execute("UPDATE control_operations SET state='cancelled_unissued',unissued_proof_json=? WHERE operation_id=?",
                               (_json(proof), operation_id))
            return self.verify_unissued_intent(intent, proof)

    def get_operation(self, operation_id: str) -> dict:
        return self._fresh("session", "get_operation", {"epoch": self.binding["epoch"], "operation_id": operation_id})

    def begin_drain(self, reason: str) -> dict:
        with self._lock:
            self._set("local_drain", True)
            self._set("local_drain_reason", reason)
            result = self.rpc("session", "begin_drain", {**self.binding, "reason": reason},
                              key="drain:" + self.binding["session_id"] + ":" + _digest(reason))
            self._set("job", result)
            return result

    def resume(self, *, session_client: Any, session_id: str,
               session_public_key: str, mode: str = "recover") -> dict:
        if (session_client.actor_id != "session:" + session_id or
                session_client.authority_id != self.controller.authority_id or
                session_client.fleet_id != self.controller.fleet_id):
            raise ControlError("Replacement session must pin the same authority/fleet")
        old = self.binding
        params = {k: old[k] for k in ("job_id", "epoch", "host_id")}
        params.update(previous_session_id=old["session_id"], session_id=session_id,
                      session_public_key=session_public_key, mode=mode)
        result = self.rpc("controller", "resume_bound_job", params, key="resume:" + session_id)
        self._set("job", result)
        self.session = session_client
        self._set("local_drain", result["phase"] != "ACTIVE")
        return result

    def ack_evidence(self, permit: dict, evidence: dict, root: str | Path,
                     aliases: dict | None = None) -> dict:
        from shore_control.evidence import verify_artifacts
        from shore_control.permits import verify_permit

        with self.journal.transaction() as db:
            row = db.execute("SELECT descriptor FROM control_operations WHERE operation_id=?",
                             (permit["payload"]["operation_id"],)).fetchone()
        if row is None:
            raise ControlError("Collector has no independently persisted operation descriptor")
        descriptor = json.loads(row["descriptor"])
        payload = verify_permit(self.controller.public_key, permit, descriptor)
        evidence_hash = verify_artifacts(evidence, payload, root, aliases)
        self._check_terminal_record(payload, evidence, root, aliases, evidence_hash)
        record = (payload["permit_id"], _json(evidence), str(Path(root).resolve()),
                  _json(aliases) if aliases is not None else None, evidence_hash)
        with self.journal.transaction() as db:
            old = db.execute("SELECT * FROM control_evidence WHERE permit_id=?", (payload["permit_id"],)).fetchone()
            if old and tuple(old) != record:
                raise ControlError("Durable collector evidence cannot be rewritten")
            db.execute("INSERT OR IGNORE INTO control_evidence VALUES(?,?,?,?,?)", record)
        receipt = {"outcome": evidence["outcome"], "evidence_sha256": evidence_hash, "durable": True}
        self.rpc("session", "controller_ack", {"permit_id": payload["permit_id"], **receipt},
            key="ack:" + self.binding["session_id"] + ":" + payload["permit_id"],
            recovery_permit=permit)
        # A replayed acknowledgement is a historical snapshot. The worker may
        # have submitted its matching receipt since our original AWAITING reply.
        current = self.get_operation(payload["operation_id"])
        self._check_operation_receipts(current, permit, descriptor, receipt)
        return {"permit_id": payload["permit_id"], "status": (
            "RESOLVED" if current["status"] == "RESOLVED" else "AWAITING_PEER_RECEIPT")}

    def _check_terminal_record(self, payload: dict, evidence: dict, root: str | Path,
                               aliases: dict | None, evidence_hash: str) -> None:
        if payload["action"] == "wake":
            from .control_remote_bell import verify_wake_evidence

            target = payload["action_context"]["target_actor_id"]
            checked = verify_wake_evidence(
                evidence, payload, root, authority_public_key=self.authority_public_key,
                bell_registration=self.registrations[payload["worker_id"]],
                target_registration=self.registrations[target],
            )
            if checked != evidence_hash:
                raise ControlError("Wake records differ from durable evidence commitment")
            return
        if evidence["outcome"] in {"failed", "not_started"}:
            from .control_failure import terminal_record_bytes

            # not_started deliberately has no Shore artifacts. Its tombstone
            # still needs an actual safe byte read against the committed hash.
            terminal_record_bytes({"root": str(root), "evidence": evidence,
                                   "aliases": aliases, "receipt": {
                                       "evidence_sha256": evidence_hash}}, payload)

    def _check_operation_receipts(self, operation: dict, permit: dict,
                                  descriptor: dict, receipt: dict) -> None:
        from shore_control.permits import verify_acceptance, verify_permit

        original = verify_permit(self.controller.public_key, permit, descriptor)
        current = verify_permit(self.controller.public_key, operation["permit"], descriptor)
        if current != original or operation["permit"] != permit:
            raise ControlError("Signed operation read changed the original permit")
        acceptance = operation.get("acceptance")
        if acceptance is not None:
            verify_acceptance(self.controller.public_key, acceptance, current)
        elif receipt["outcome"] != "not_started":
            raise ControlError("Executed terminal evidence lacks authority acceptance")
        if operation.get("controller_receipt") != receipt:
            raise ControlError("Fresh collector receipt differs from durable verified evidence")
        peer = operation.get("worker_receipt")
        if peer is not None and peer != receipt:
            raise ControlError("Fresh worker receipt differs from durable verified evidence")
        if operation["status"] == "RESOLVED" and peer is None:
            raise ControlError("RESOLVED operation lacks the matching worker receipt")

    def _verify_operation_evidence(self, descriptor: dict, operation: dict) -> None:
        from shore_control.evidence import verify_artifacts
        from shore_control.permits import verify_permit

        with self.journal.transaction() as db:
            row = db.execute("SELECT permit FROM control_operations WHERE operation_id=?",
                             (descriptor["operation_id"],)).fetchone()
            if row is None or row["permit"] is None:
                raise ControlError("Operation lacks locally persisted permit provenance")
            permit = json.loads(row["permit"])
            saved = db.execute("SELECT * FROM control_evidence WHERE permit_id=?",
                               (permit["payload"]["permit_id"],)).fetchone()
        if saved is None:
            raise ControlError("Resolved operation lacks durable collector evidence")
        payload = verify_permit(self.controller.public_key, permit, descriptor)
        evidence = json.loads(saved["evidence"])
        aliases = json.loads(saved["aliases"]) if saved["aliases"] is not None else None
        checked = verify_artifacts(evidence, payload, saved["root"], aliases)
        if checked != saved["evidence_sha256"]:
            raise ControlError("Saved collector commitment differs from actual evidence bytes")
        self._check_terminal_record(payload, evidence, saved["root"], aliases, checked)
        self._check_operation_receipts(operation, permit, descriptor, {
            "outcome": evidence["outcome"], "evidence_sha256": checked, "durable": True})

    def bind_job_documents(self, manifest_path: str | Path,
                           accounting_policy_path: str | Path) -> None:
        spec = self.job["scope"]["job_spec"]
        paths = {"manifest_path": str(Path(manifest_path).resolve()),
                 "accounting_policy_path": str(Path(accounting_policy_path).resolve())}
        if (_file_hash(paths["manifest_path"]) != spec["manifest_sha256"] or
                _file_hash(paths["accounting_policy_path"]) != spec["accounting_policy_sha256"]):
            raise ControlError("Job documents do not match immutable job_spec")
        policy = json.loads(Path(paths["accounting_policy_path"]).read_text(encoding="utf-8"))
        if (set(policy) != {"schema", "schema_version", "unit_key", "settled_outcomes"}
                or policy["schema"] != "msabp.budget-policy" or policy["schema_version"] != 1
                or policy["unit_key"] != "evaluation_id"
                or not policy["settled_outcomes"]
                or not set(policy["settled_outcomes"]) <= {"completed", "failed"}):
            raise ControlError("Unsupported explicit evaluation accounting policy")
        old = self._get("job_documents")
        if old is not None and old != paths:
            raise ControlError("Job document provenance cannot change")
        self._set("job_documents", paths)

    def _policy(self) -> dict:
        paths = self._get("job_documents")
        if not paths:
            raise ControlError("Verify actual full-job and accounting-policy documents first")
        self.bind_job_documents(**paths)
        return json.loads(Path(paths["accounting_policy_path"]).read_text(encoding="utf-8"))

    def bind_run_state(self, run_id: str, state_database: str | Path) -> None:
        database = str(Path(state_database).resolve())
        if not Path(database).is_file():
            raise ControlError("Actual run state database is missing")
        with self.journal.transaction() as db:
            row = db.execute("SELECT * FROM control_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise ControlError("Register run before binding its state database")
            if row["state_database"] is not None and row["state_database"] != database:
                raise ControlError("Bound run state provenance cannot change")
            db.execute("UPDATE control_runs SET state_database=? WHERE run_id=?", (database, run_id))
        self._run_snapshot(run_id, require_plan=False)

    def bind_run_plan(self, run_id: str, plan_path: str | Path) -> None:
        path = str(Path(plan_path).resolve())
        with self.journal.transaction() as db:
            row = db.execute("SELECT * FROM control_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise ControlError("Register run before binding original plan provenance")
            if _file_hash(path) != json.loads(row["registration"])["original_plan_sha256"]:
                raise ControlError("Actual original plan bytes differ from registered digest")
            if row["plan_path"] is not None and row["plan_path"] != path:
                raise ControlError("Original plan provenance cannot change")
            db.execute("UPDATE control_runs SET plan_path=? WHERE run_id=?", (path, run_id))

    def _run_snapshot(self, run_id: str, *, require_plan: bool = True) -> dict:
        with self.journal.transaction() as db:
            entry = db.execute("SELECT * FROM control_runs WHERE run_id=?", (run_id,)).fetchone()
        if entry is None or entry["state_database"] is None:
            raise ControlError("Every registered run requires actual project state provenance")
        registration = json.loads(entry["registration"])
        if entry["plan_path"] is None:
            if require_plan:
                raise ControlError("Every registered run requires actual original plan provenance")
        elif _file_hash(entry["plan_path"]) != registration["original_plan_sha256"]:
            raise ControlError("Actual original plan bytes changed after registration")
        database = Path(entry["state_database"])
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise ControlError("Run absent from referenced project database")
            if (_file_hash(run["csv_path"]) != run["csv_sha256"] or
                    run["csv_sha256"] != registration["input_manifest_sha256"]):
                raise ControlError("Frozen CSV content differs from registered input manifest")
            tasks = [dict(r) for r in db.execute("SELECT * FROM tasks WHERE run_id=? ORDER BY case_id", (run_id,))]
            attempts = [dict(r) for r in db.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY attempt_id", (run_id,))]
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"control_collection", "control_uploads"} <= tables:
                raise ControlError("Controlled collector durability tables are missing")
            collections = [dict(r) for r in db.execute("SELECT * FROM control_collection ORDER BY attempt_id")]
            uploads = [dict(r) for r in db.execute("SELECT * FROM control_uploads ORDER BY upload_id")]
            dispatch_intents = ([dict(r) for r in db.execute(
                "SELECT * FROM control_dispatch_intents WHERE run_id=? ORDER BY operation_id", (run_id,))]
                if "control_dispatch_intents" in tables else [])
            retry_grants = ([dict(r) for r in db.execute(
                "SELECT * FROM control_retry_grants WHERE run_id=? ORDER BY grant_id", (run_id,))]
                if "control_retry_grants" in tables else [])
            # Preserve raw decimal timestamps in provenance without admitting
            # floats to the shared canonical signing format.
            def safe(row):
                return {k: repr(v) if isinstance(v, float) else v for k, v in row.items()}
            return {"state_database": str(database), "original_plan_path": entry["plan_path"],
                    "original_plan_sha256": registration["original_plan_sha256"], "run": safe(dict(run)),
                    "tasks": [safe(t) for t in tasks], "attempts": [safe(a) for a in attempts],
                    "collections": [safe(r) for r in collections], "uploads": [safe(r) for r in uploads],
                    "dispatch_intents": [safe(r) for r in dispatch_intents],
                    "retry_grants": [safe(r) for r in retry_grants]}

    def settle_evaluation(self, evaluation_id: str, run_id: str, case_id: str,
                          attempt_id: str) -> dict:
        policy = self._policy()
        snapshot = self._run_snapshot(run_id)
        task = next((t for t in snapshot["tasks"] if t["case_id"] == str(case_id)), None)
        attempt = next((a for a in snapshot["attempts"] if a["attempt_id"] == attempt_id), None)
        case_attempts = [a for a in snapshot["attempts"] if a["case_id"] == str(case_id)]
        latest = max(case_attempts, key=lambda a: a["attempt_number"]) if case_attempts else None
        if (task is None or attempt is None or latest["attempt_id"] != attempt_id
                or task["attempt_id"] not in (None, attempt_id)
                or task["status"] not in policy["settled_outcomes"]
                or attempt["status"] != task["status"] or attempt["case_id"] != str(case_id)):
            raise ControlError("Evaluation lacks matching actual terminal task and attempt")
        with self.journal.transaction() as db:
            operations = db.execute("SELECT * FROM control_operations").fetchall()
        match = [r for r in operations if json.loads(r["descriptor"])["attempt_id"] == attempt_id
                 and json.loads(r["descriptor"])["run_id"] == run_id
                 and json.loads(r["descriptor"])["action"] in ("attempt", "gpu")]
        if len(match) != 1 or match[0]["permit"] is None:
            raise ControlError("Evaluation lacks exactly one durable work operation")
        descriptor = json.loads(match[0]["descriptor"])
        if descriptor["input_sha256"] != task["row_sha256"]:
            raise ControlError("Evaluation input digest differs from the actual frozen row")
        operation = self.get_operation(descriptor["operation_id"])
        if operation["status"] != "RESOLVED":
            raise ControlError("Evaluation is not settled by both signed receipts")
        permit = json.loads(match[0]["permit"])
        with self.journal.transaction() as db:
            saved = db.execute("SELECT * FROM control_evidence WHERE permit_id=?", (permit["payload"]["permit_id"],)).fetchone()
        if saved is None:
            raise ControlError("Evaluation lacks durable collector evidence")
        from shore_control.evidence import verify_artifacts
        from shore_control.permits import verify_permit

        payload = verify_permit(self.controller.public_key, permit, descriptor)
        evidence = json.loads(saved["evidence"])
        evidence_hash = verify_artifacts(evidence, payload, saved["root"], json.loads(saved["aliases"]) if saved["aliases"] else None)
        if (evidence["outcome"] != task["status"]
                or operation["worker_receipt"]["evidence_sha256"] != evidence_hash
                or operation["controller_receipt"]["evidence_sha256"] != evidence_hash):
            raise ControlError("Actual terminal state and dual evidence commitments disagree")
        record = {"evaluation_id": evaluation_id, "run_id": run_id, "case_id": str(case_id),
                  "attempt_id": attempt_id, "operation_id": descriptor["operation_id"],
                  "permit_id": payload["permit_id"], "input_sha256": task["row_sha256"],
                  "outcome": task["status"], "evidence_sha256": evidence_hash,
                  "state_database": snapshot["state_database"],
                  "task_state_sha256": _digest(task), "attempt_state_sha256": _digest(attempt)}
        approved_units = self.job["scope"]["job_spec"]["approved_units"]
        with self.journal.transaction() as db:
            old = db.execute("SELECT record FROM control_evaluations WHERE evaluation_id=?", (evaluation_id,)).fetchone()
            if old:
                if json.loads(old["record"]) != record:
                    raise ControlError("Evaluation settlement cannot change")
                return record
            count = db.execute("SELECT COUNT(*) FROM control_evaluations").fetchone()[0]
            if count >= approved_units:
                raise ControlError("Full-job approved evaluation budget exceeded")
            try:
                db.execute("INSERT INTO control_evaluations VALUES(?,?,?,?)",
                           (evaluation_id, run_id, str(case_id), _json(record)))
            except sqlite3.IntegrityError as exc:
                raise ControlError("A retried case cannot consume another evaluation unit") from exc
        return record

    def build_checkpoint(self, **inflight_counts: int) -> tuple[dict, dict]:
        policy = self._policy()
        expected = {"uploads_in_progress", "worker_outboxes_pending", "receipts_pending", "gpu_uncollected"}
        if set(inflight_counts) - expected:
            raise ControlError("Unknown in-flight checkpoint counter")
        if any(type(v) is not int or v < 0 for v in inflight_counts.values()):
            raise ControlError("In-flight counters must be nonnegative integers")
        with self.journal.transaction() as db:
            runs = db.execute("SELECT * FROM control_runs ORDER BY run_id").fetchall()
            evaluations = [json.loads(r[0]) for r in db.execute("SELECT record FROM control_evaluations ORDER BY evaluation_id")]
            operation_rows = db.execute("SELECT * FROM control_operations ORDER BY operation_id").fetchall()
        operations, cancelled = [], {}
        for row in operation_rows:
            descriptor = json.loads(row["descriptor"])
            if row["state"] != "cancelled_unissued":
                operations.append(descriptor)
                continue
            if row["unissued_proof_json"] is None:
                raise ControlError("Cancelled unissued operation lacks durable proof")
            intent = json.loads(row["intent"])
            proof = self.verify_unissued_intent(intent, json.loads(row["unissued_proof_json"]))
            cancelled[row["operation_id"]] = {"state": "cancelled_unissued", "intent": intent,
                                               "descriptor": descriptor, "proof": proof}
        snapshots, run_entries = [], []
        for run in runs:
            snapshot = self._run_snapshot(run["run_id"])
            snapshots.append(snapshot)
            if any(not r["settled"] for r in snapshot["collections"]):
                raise ControlError("Collector completion journal has pending durable work")
            if any(r["status"] == "writing" for r in snapshot["uploads"]):
                raise ControlError("Collector upload journal has unfinished bytes")
            for entry in snapshot["dispatch_intents"]:
                if entry["state"] == "settled":
                    continue
                if entry["state"] != "cancelled_unissued":
                    raise ControlError("Project dispatch journal has unresolved persistent intent")
                audit = cancelled.get(entry["operation_id"])
                try:
                    proof = json.loads(entry["unissued_proof_json"])
                    intent = {**json.loads(entry["intent_json"]), "descriptor": json.loads(entry["descriptor_json"])}
                except (KeyError, TypeError, ValueError) as exc:
                    raise ControlError("Project cancelled intent lacks valid durable proof") from exc
                if (audit is None or entry["permit_json"] is not None
                        or audit["intent"] != intent or audit["proof"] != proof):
                    raise ControlError("Project cancelled intent differs from signed facade proof")
                self.verify_unissued_intent(intent, proof)
            settled = [e for e in evaluations if e["run_id"] == run["run_id"]]
            settled_cases = {e["case_id"] for e in settled}
            terminal_cases = {t["case_id"] for t in snapshot["tasks"] if t["status"] in policy["settled_outcomes"]}
            if settled_cases != terminal_cases:
                raise ControlError("Registered run has terminal evaluations missing from the settlement ledger")
            if any(t["status"] not in ("pending", "completed", "failed") for t in snapshot["tasks"]):
                raise ControlError("Registered run still has locally unfinished work")
            if any(a["status"] in ("prepared", "running", "artifact_ready", "unknown") for a in snapshot["attempts"]):
                raise ControlError("Registered run has an unfinished attempt or execution intent")
            for evaluation in settled:
                self.settle_evaluation(evaluation["evaluation_id"], evaluation["run_id"],
                                       evaluation["case_id"], evaluation["attempt_id"])
            run_entries.append({**json.loads(run["registration"]), "accounting_ledger_sha256": _digest(settled)})
        operation_states = [self.get_operation(d["operation_id"]) for d in operations]
        if any(op["status"] != "RESOLVED" for op in operation_states):
            raise ControlError("Issued operations remain unresolved")
        for descriptor, operation in zip(operations, operation_states, strict=True):
            self._verify_operation_evidence(descriptor, operation)
        spec = self.job["scope"]["job_spec"]
        settled_count = len(evaluations)
        manifest = {"schema": "shore.job-checkpoint", "schema_version": 1,
                    "fleet_id": self.controller.fleet_id, "job_id": self.binding["job_id"],
                    "epoch": self.binding["epoch"], "bound_host_id": self.binding["host_id"],
                    "collector_url": self.job["scope"]["collector_url"], "data_root": self.job["scope"]["data_root"],
                    "job_spec_sha256": _digest(spec), "approved_units": spec["approved_units"],
                    "settled_units": settled_count, "pending_units": spec["approved_units"] - settled_count,
                    "budget_unit": spec["budget_unit"], "accounting_policy_sha256": spec["accounting_policy_sha256"],
                    "runs": run_entries, "budget_ledger_sha256": _digest(evaluations),
                    "reconciliation_manifest_sha256": _digest({"snapshots": snapshots, "operations": operation_states,
                                                               "cancelled_unissued": list(cancelled.values())})}
        checkpoint = {"manifest_sha256": _digest(manifest), "pending_count": manifest["pending_units"],
                      "local_unresolved_count": 0, **{k: inflight_counts.get(k, 0) for k in expected}}
        self._set("reconciliation", {"manifest": manifest, "checkpoint": checkpoint,
                                     "evaluations": evaluations, "snapshots": snapshots, "operations": operation_states,
                                     "cancelled_unissued": list(cancelled.values())})
        return checkpoint, manifest

    def finish_drain(self, **inflight_counts: int) -> dict:
        checkpoint, manifest = self.build_checkpoint(**inflight_counts)
        result = self.rpc("session", "finish_drain", {**self.binding, "checkpoint": checkpoint,
                          "job_checkpoint": manifest}, key="finish:" + self.binding["session_id"]
                          + ":" + _digest({"checkpoint": checkpoint, "job_checkpoint": manifest}))
        self._set("job", result)
        return result

    def release(self, *, disposition: str, reason: str, explicit_close: bool = False) -> dict:
        if disposition == "closed" and explicit_close is not True:
            raise ControlError("Abandoning remaining budget requires explicit user close authorization")
        checkpoint = self.job.get("checkpoint")
        if checkpoint is None:
            raise ControlError("Finish drain before explicit release")
        params = {**self.binding,
                          "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                          "disposition": disposition, "reason": reason}
        result = self.rpc("controller", "release_job", params,
                          key="release:" + self.binding["job_id"] + ":" + _digest(params))
        self._set("job", result)
        self._set("local_drain", True)
        return result
