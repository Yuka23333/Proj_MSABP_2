"""Fail-closed Shore adapter for the real Maid entry, synthetic executors only.

An execution intent is durable before the injected executor runs. A restart in
that window is UNKNOWN, never permission to execute again. This intentionally
does not claim exactly-once execution for an external CST process.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Any, Mapping


class ExecutionUncertain(RuntimeError):
    """An earlier invocation may have started; reconciliation is required."""


def _json(value: Any) -> str:
    from shore_control.crypto import canonical_json

    return canonical_json(value).decode("utf-8")


def _persist_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(_json(value).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def compute_evidence(
    permit: dict,
    root: Path,
    *,
    outcome: str,
    execution_id: str | None,
    failure_code: str | None = None,
    termination_record_sha256: str | None = None,
    tombstone_sha256: str | None = None,
) -> tuple[dict, dict[str, str], str]:
    """Build common evidence while retaining every historical local filename."""
    from shore_control.authority import digest
    from shore_control.evidence import verify_artifacts
    from shore_control.permits import DESCRIPTOR_FIELDS

    root = Path(root).absolute()
    aliases: dict[str, str] = {}
    artifacts = []
    paths = [] if outcome == "not_started" else sorted(root.rglob("*"))
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        # The alias is a logical name only, not a rename of the old artifact.
        if relative == "Farfield Source [1].ffs":
            logical = "farfield.ffs"
        elif re.fullmatch(r"[A-Za-z0-9._/-]+", relative):
            logical = relative
        else:
            logical = "artifact-" + hashlib.sha256(relative.encode()).hexdigest()
        if logical.casefold() in {name.casefold() for name in aliases}:
            raise ValueError("Conflicting logical artifact names")
        # Flush executor output before admitting it into the durable outbox.
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
            contents = stream.read()
        aliases[logical] = relative
        artifacts.append({"path": logical, "size_bytes": len(contents),
                          "sha256": hashlib.sha256(contents).hexdigest()})
    evidence = {
        "schema": "shore.operation-evidence", "schema_version": 1,
        "kind": permit["action"],
        **{key: permit[key] for key in ("fleet_id", "job_id", "epoch", "permit_id",
                                       "worker_id", "device_id", "input_sha256")},
        "descriptor_sha256": digest({key: permit[key] for key in DESCRIPTOR_FIELDS}),
        "outcome": outcome, "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "result": {"state": outcome, "execution_id": execution_id,
                   "failure_code": failure_code,
                   "termination_record_sha256": termination_record_sha256,
                   "tombstone_sha256": tombstone_sha256},
    }
    return evidence, aliases, verify_artifacts(evidence, permit, root, aliases)


class ShoreWorker:
    """Pinned worker identity, durable inbox/execution log/outbox, signed RPC.

    ``binding`` is provisioned independently, never learned from an assignment.
    It contains the eleven job/session descriptor fields excluding per-attempt,
    actor/registration and action fields. ``registration`` is the actor's pinned
    Shore registration; ``csv_sha256`` pins the local full CSV separately.

    Restart contract: keep the original binding while any durable inbox/outbox
    entry is unresolved. Session rotation is permitted in this worker database
    only after every old entry has independently verified RESOLVED receipts.
    A recovered collector may still acknowledge work under its new session.
    """

    def __init__(self, root: str | Path, client: Any, authority_public_key: Any,
                 *, binding: Mapping[str, Any], registration: Mapping[str, Any],
                 csv_sha256: str) -> None:
        from shore_control.permits import DESCRIPTOR_FIELDS
        from shore_control.registry import validate_workers
        from .control_authority import SignedRpcJournal

        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "worker.sqlite"
        self.client = client
        self.authority_public_key = authority_public_key
        self.binding = json.loads(_json(dict(binding)))
        dynamic = {"operation_id", "run_id", "worker_id", "attempt_id", "action",
                   "input_sha256", "device_id", "worker_role",
                   "worker_registration_sha256", "action_context"}
        if set(self.binding) != DESCRIPTOR_FIELDS - dynamic:
            raise ValueError("Worker requires a complete independently pinned job binding")
        self.registration = validate_workers({client.actor_id: dict(registration)})[client.actor_id]
        if self.registration["role"] != "maid" or client.fleet_id != self.binding["fleet_id"]:
            raise ValueError("ShoreWorker requires the registered Maid actor and fleet")
        if not re.fullmatch(r"[a-f0-9]{64}", csv_sha256) or csv_sha256 == "0" * 64:
            raise ValueError("Worker requires an independent nonzero CSV SHA-256")
        self.csv_sha256 = csv_sha256
        self.rpc = SignedRpcJournal(self.root / "rpc.sqlite")
        with self.transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS inbox(
                attempt_id TEXT PRIMARY KEY, permit_id TEXT UNIQUE NOT NULL,
                envelope TEXT NOT NULL, descriptor TEXT NOT NULL,
                acceptance TEXT, state TEXT NOT NULL, result TEXT,
                delivery TEXT, resolved TEXT, uploaded TEXT, assignment TEXT)""")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(inbox)")}
            if "uploaded" not in columns:
                db.execute("ALTER TABLE inbox ADD COLUMN uploaded TEXT")
            if "assignment" not in columns:
                db.execute("ALTER TABLE inbox ADD COLUMN assignment TEXT")
            for row in db.execute("SELECT attempt_id,descriptor FROM inbox WHERE state!='RESOLVED'"):
                original = json.loads(row["descriptor"])
                if any(original.get(key) != value for key, value in self.binding.items()):
                    raise ValueError(
                        "Worker restart must retain the original job/session binding until all "
                        "prior inbox/outbox entries are RESOLVED; restore that binding and "
                        f"reconcile attempt {row['attempt_id']} before changing sessions"
                    )

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _record(self, attempt_id: str) -> dict:
        with self.transaction() as db:
            record = db.execute("SELECT * FROM inbox WHERE attempt_id=?", (attempt_id,)).fetchone()
        if record is None:
            raise ValueError("Attempt has no durable Shore inbox")
        return dict(record)

    def _permit(self, record: dict) -> dict:
        from shore_control.permits import verify_acceptance, verify_permit

        permit = verify_permit(self.authority_public_key, json.loads(record["envelope"]),
                               json.loads(record["descriptor"]))
        if not record["acceptance"]:
            result = json.loads(record["result"]) if record.get("result") else None
            if not result or result["receipt"]["outcome"] != "not_started":
                raise ValueError("Attempt lacks durable Shore acceptance")
        else:
            verify_acceptance(self.authority_public_key, json.loads(record["acceptance"]), permit)
        return permit

    def accept(self, assignment: Mapping[str, Any], *, run_id: str, worker_id: str,
               row_sha256: str, csv_sha256: str) -> dict:
        from shore_control.permits import verify_acceptance, verify_permit
        from shore_control.registry import registration_sha256

        gate = assignment.get("control_authority")
        if not isinstance(gate, Mapping) or set(gate) != {"descriptor", "permit"}:
            raise ValueError("Controlled assignment requires a complete Shore gate")
        if worker_id != self.client.actor_id or csv_sha256 != self.csv_sha256:
            raise ValueError("Runtime worker or CSV differs from pinned Shore identity")
        descriptor = gate["descriptor"]
        if not isinstance(descriptor, dict):
            raise ValueError("Malformed Shore descriptor")
        expected = {
            **self.binding, "operation_id": descriptor.get("operation_id"),
            "run_id": run_id, "worker_id": worker_id,
            "attempt_id": assignment["attempt_id"], "action": "attempt",
            "input_sha256": row_sha256, "device_id": self.registration["device_id"],
            "worker_role": "maid", "worker_registration_sha256": registration_sha256(self.registration),
            "action_context": None,
        }
        if _json(expected) != _json(descriptor):
            raise ValueError("Assignment descriptor differs from independently expected work")
        permit = verify_permit(self.authority_public_key, gate["permit"], expected)
        attempt_id, permit_id = assignment["attempt_id"], permit["permit_id"]
        # Protocol bookkeeping can contain fractional lease timestamps; it is
        # not Shore evidence and must not be encoded with Shore's integer-only
        # canonicalizer. Signed descriptor/evidence use their strict encoding.
        assignment_json = json.dumps(dict(assignment), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False)
        with self.transaction() as db:
            old = db.execute("SELECT * FROM inbox WHERE attempt_id=?", (attempt_id,)).fetchone()
            if old is not None:
                if old["envelope"] != _json(gate["permit"]) or old["descriptor"] != _json(expected):
                    raise ValueError("Repeated assignment changes durable Shore permit")
            else:
                db.execute("INSERT INTO inbox(attempt_id,permit_id,envelope,descriptor,state,assignment) VALUES(?,?,?,?,?,?)",
                           (attempt_id, permit_id, _json(gate["permit"]), _json(expected), "PREPARED", assignment_json))
            db.execute("UPDATE inbox SET assignment=? WHERE attempt_id=? AND assignment IS NULL",
                       (assignment_json, attempt_id))
        record = self._record(attempt_id)
        if record["acceptance"]:
            self._permit(record)
            return permit
        if record["result"]:
            # A PREPARED, explicitly tombstoned operation must never retry accept.
            self._permit(record)
            return permit
        acceptance = self.rpc.call(self.client, "accept_permit",
            {key: permit[key] for key in ("permit_id", "operation_id", "run_id", "attempt_id",
                                         "action", "input_sha256", "device_id")},
            key="accept:" + permit_id, recovery_permit=gate["permit"])
        verify_acceptance(self.authority_public_key, acceptance, permit)
        with self.transaction() as db:
            db.execute("UPDATE inbox SET acceptance=?,state='ACCEPTED' WHERE attempt_id=? AND state='PREPARED'",
                       (_json(acceptance), attempt_id))
        return permit

    def start(self, attempt_id: str) -> str:
        self._permit(self._record(attempt_id))
        with self.transaction() as db:
            record = db.execute("SELECT state,permit_id FROM inbox WHERE attempt_id=?", (attempt_id,)).fetchone()
            if record["state"] != "ACCEPTED":
                raise ExecutionUncertain("Attempt is already STARTING or terminal; never reexecute")
            db.execute("UPDATE inbox SET state='STARTING' WHERE attempt_id=?", (attempt_id,))
        return "maid-" + record["permit_id"]

    def result(self, attempt_id: str) -> dict | None:
        record = self._record(attempt_id)
        if record["result"]:
            return json.loads(record["result"])
        if record["state"] == "STARTING":
            raise ExecutionUncertain("Previous executor start is uncertain; manual reconciliation required")
        return None

    def save_result(self, attempt_id: str, root: Path, *, outcome: str,
                    manifest_path: Path | None = None, failure_code: str | None = None,
                    error_message: str | None = None) -> dict:
        from shore_control.authority import digest

        record = self._record(attempt_id)
        if outcome == "not_started" and record["state"] == "PREPARED":
            from shore_control.permits import verify_permit

            permit = verify_permit(self.authority_public_key, json.loads(record["envelope"]),
                                   json.loads(record["descriptor"]))
        else:
            permit = self._permit(record)
        expected_states = {"ACCEPTED", "PREPARED"} if outcome == "not_started" else {"STARTING"}
        if record["state"] not in expected_states:
            raise ValueError("Result cannot rewrite an existing terminal or uncertain state")
        expected_state = record["state"]
        root = Path(root).absolute()
        root.mkdir(parents=True, exist_ok=True)
        execution_id = None if outcome == "not_started" else "maid-" + permit["permit_id"]
        termination_hash = tombstone_hash = None
        if outcome in {"failed", "not_started"}:
            from .control_failure import bounded_terminal_record

            durable_record = bounded_terminal_record({
                "synthetic": True, "permit_id": permit["permit_id"],
                "execution_id": execution_id, "state": outcome,
                "failure_code": failure_code, "error_message": error_message})
            if outcome == "failed":
                _persist_json(root / "termination.json", durable_record)
                termination_hash = digest(durable_record)
            else:
                # A durable terminal inbox state is the launch-preventing tombstone.
                tombstone_hash = digest(durable_record)
                _persist_json(root / "tombstone.json", durable_record)
        evidence, aliases, evidence_hash = compute_evidence(
            permit, root, outcome=outcome, execution_id=execution_id,
            failure_code=failure_code, termination_record_sha256=termination_hash,
            tombstone_sha256=tombstone_hash)
        result = {"root": str(root), "manifest_path": str(manifest_path) if manifest_path else None,
                  "evidence": evidence, "aliases": aliases,
                  "receipt": {"permit_id": permit["permit_id"], "outcome": outcome,
                              "durable": True, "evidence_sha256": evidence_hash}}
        if outcome == "not_started":
            result["tombstone"] = durable_record
        if outcome in {"failed", "not_started"}:
            from .control_failure import terminal_record_bytes

            # A RESULT is immutable. Validate the actual persisted bytes with
            # the consumer's full schema/size/evidence contract before committing.
            terminal_record_bytes(result, permit)
        with self.transaction() as db:
            updated = db.execute("UPDATE inbox SET result=?,state='RESULT' WHERE attempt_id=? AND state=?",
                                 (_json(result), attempt_id, expected_state))
            if updated.rowcount != 1:
                raise ValueError("Concurrent worker changed execution state")
        return result

    def checked_result(self, attempt_id: str) -> dict:
        from shore_control.evidence import verify_artifacts

        record = self._record(attempt_id)
        permit = self._permit(record)
        result = self.result(attempt_id)
        if result is None:
            raise ValueError("No durable worker result")
        if result["receipt"]["outcome"] != "completed":
            from .control_failure import TerminalRecordTooLarge, terminal_record_bytes

            try:
                terminal_record_bytes(result, permit)
            except TerminalRecordTooLarge as exc:
                raise ValueError(
                    "Persisted terminal RESULT exceeds 64 KiB; manual reconciliation/migration "
                    "is required. Its immutable bytes and signed receipts must not be rewritten"
                ) from exc
        else:
            checked = verify_artifacts(result["evidence"], permit, result["root"], result["aliases"])
            if checked != result["receipt"]["evidence_sha256"]:
                raise ValueError("Stored outbox differs from artifact bytes")
        return result

    def report(self, attempt_id: str) -> dict:
        result = self.checked_result(attempt_id)
        return self.rpc.call(self.client, "worker_report", result["receipt"],
                             key="report:" + result["receipt"]["permit_id"],
                             recovery_permit=json.loads(self._record(attempt_id)["envelope"]))

    def delivery(self, attempt_id: str, value: dict | None = None) -> dict | None:
        with self.transaction() as db:
            record = db.execute("SELECT delivery FROM inbox WHERE attempt_id=?", (attempt_id,)).fetchone()
            if record is None:
                raise ValueError("Unknown attempt")
            if value is not None:
                if record["delivery"] and record["delivery"] != _json(value):
                    raise ValueError("Cannot change durable completion payload")
                db.execute("UPDATE inbox SET delivery=? WHERE attempt_id=?", (_json(value), attempt_id))
                return value
        return json.loads(record["delivery"]) if record["delivery"] else None

    def resolved(self, attempt_id: str) -> bool:
        from shore_control.permits import verify_acceptance, verify_permit

        record = self._record(attempt_id)
        permit = self._permit(record)
        # Reconcile lost replies first, then obtain a genuinely current status.
        status = self.rpc.fresh_read(self.client, "get_operation",
                                    {"epoch": permit["epoch"], "operation_id": permit["operation_id"]})
        verified = verify_permit(self.authority_public_key, status["permit"], json.loads(record["descriptor"]))
        if verified != permit:
            raise ValueError("Operation status refers to another permit")
        if status.get("status") != "RESOLVED":
            return False
        receipt = json.loads(record["result"])["receipt"]
        if receipt["outcome"] != "not_started" or status["acceptance"] is not None:
            verify_acceptance(self.authority_public_key, status["acceptance"], permit)
        for name in ("worker_receipt", "controller_receipt"):
            peer = status.get(name)
            if not isinstance(peer, dict) or any(peer.get(key) != receipt[key] for key in ("outcome", "evidence_sha256", "durable")):
                raise ValueError("Resolved operation receipts differ from the durable worker result")
        with self.transaction() as db:
            db.execute("UPDATE inbox SET resolved=?,state='RESOLVED' WHERE attempt_id=?", (_json(status), attempt_id))
        return True

    def uploaded(self, attempt_id: str, archive_sha256: str | None = None) -> bool:
        """Persist verified upload response before issuing HTTP completion."""
        with self.transaction() as db:
            record = db.execute("SELECT uploaded,delivery FROM inbox WHERE attempt_id=?", (attempt_id,)).fetchone()
            if record is None or not record["delivery"]:
                raise ValueError("Upload requires a durable delivery payload")
            expected = json.loads(record["delivery"])["payload"]["archive_sha256"]
            if archive_sha256 is not None:
                if archive_sha256 != expected:
                    raise ValueError("Upload acknowledgement differs from immutable archive")
                db.execute("UPDATE inbox SET uploaded=? WHERE attempt_id=?", (archive_sha256, attempt_id))
                return True
            if record["uploaded"] not in (None, expected):
                raise ValueError("Durable upload acknowledgement changed")
            return record["uploaded"] == expected

    def is_resolved(self, attempt_id: str) -> bool:
        return self._record(attempt_id)["state"] == "RESOLVED"

    def pending_assignments(self) -> list[dict]:
        """Recover terminal outboxes before polling, without restarting work."""
        with self.transaction() as db:
            rows = db.execute("SELECT state,result,assignment FROM inbox WHERE state!='RESOLVED' ORDER BY rowid").fetchall()
        if any(row["state"] == "STARTING" for row in rows):
            raise ExecutionUncertain("Worker startup found uncertain external execution; reconciliation required")
        assignments = []
        for row in rows:
            if row["result"] is not None:
                if row["assignment"] is None:
                    raise ValueError("Durable outbox lacks its original assignment; manual reconciliation required")
                assignments.append(json.loads(row["assignment"]))
        return assignments

