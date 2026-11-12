"""Durable completion and streaming-upload journal for controlled Princess.

This is the existing project collector's commit adapter, not another scheduler.
The authority remains the sole fleet owner. Interrupted uploads deliberately
remain unresolved until their state can be proven, never guessed from a lease.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3


class ControlCollector:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.database = coordinator.state.path
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS control_collection (
                    attempt_id TEXT PRIMARY KEY, request_json TEXT NOT NULL,
                    settled INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS control_uploads (
                    upload_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL,
                    status TEXT NOT NULL);
            """)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def begin_upload(self, attempt_id, worker_id, lease_token):
        c = self.coordinator
        if c._completion_receipt_path(attempt_id).exists():
            raise ValueError("Controlled result already committed; replay completion instead")
        c.state.heartbeat(c.run_id, worker_id, attempt_id, lease_token,
                          lease_seconds=c.lease_seconds, phase="uploading_artifact")
        if c.state.control_for_attempt(attempt_id) is None:
            raise ValueError("Upload has no controlled operation")
        upload_id = secrets.token_hex(16)
        with self.transaction() as db:
            # Serialize admission with completion. A duplicate stream cannot be
            # admitted after a completion journal has reserved the final commit.
            if db.execute("SELECT 1 FROM control_collection WHERE attempt_id=?",
                          (attempt_id,)).fetchone():
                raise ValueError("Completion already prepared; replay its request")
            db.execute("INSERT INTO control_uploads VALUES(?,?,'writing')",
                       (upload_id, attempt_id))
        return upload_id

    def end_upload(self, upload_id):
        with self.transaction() as db:
            db.execute("UPDATE control_uploads SET status='finished' WHERE upload_id=?",
                       (upload_id,))

    def complete(self, request):
        from shore_control.evidence import verify_artifacts
        from shore_control.permits import verify_permit
        from . import case_runner
        from .princess import _required_sha256, _required_text, _safe_identifier, _sha256_file, _write_json_atomic

        c, state = self.coordinator, self.coordinator.state
        gate = state.control_authority
        if gate is None:
            raise ValueError("Controlled collector has no authority/session")
        payload = request["payload"]
        worker_id = _safe_identifier(request["worker_id"], "worker_id")
        attempt_id = _safe_identifier(_required_text(payload, "attempt_id"), "attempt_id")
        lease = _required_text(payload, "lease_token")
        archive_hash = _required_sha256(payload, "archive_sha256")
        manifest_hash = _required_sha256(payload, "manifest_sha256")
        control = payload.get("control_authority")
        if not isinstance(control, dict) or control.get("outcome") != "completed":
            raise ValueError("Only verified completed fake results can settle in this adapter revision")
        envelope = state.control_for_attempt(attempt_id)
        if envelope is None:
            raise ValueError("No durable permit for this attempt")
        permit = verify_permit(gate.authority_public_key, envelope["permit"], envelope["descriptor"])
        if (permit["worker_id"] != worker_id or permit["run_id"] != c.run_id
                or permit["attempt_id"] != attempt_id or control.get("permit_id") != permit["permit_id"]):
            raise ValueError("Completion differs from the registered operation")
        evidence, aliases = control["evidence"], control["aliases"]
        identity = {"worker_id": worker_id, "attempt_id": attempt_id,
                    "lease_token_sha256": hashlib.sha256(lease.encode()).hexdigest(),
                    "archive_sha256": archive_hash, "manifest_sha256": manifest_hash,
                    "control_authority": control}
        with c._attempt_lock(attempt_id):
            with self.transaction() as db:
                old = db.execute("SELECT * FROM control_collection WHERE attempt_id=?", (attempt_id,)).fetchone()
            if old is not None:
                journal = json.loads(old["request_json"])
                if journal["identity"] != identity:
                    raise ValueError("Controlled completion replay changed its durable input")
                artifact_path = Path(journal["artifact_path"])
                case_id = journal["case_id"]
            else:
                state.heartbeat(c.run_id, worker_id, attempt_id, lease,
                                lease_seconds=c.lease_seconds, phase="verifying_control_evidence")
                worker = state.get_worker(c.run_id, worker_id)
                if worker.get("current_attempt_id") != attempt_id:
                    raise ValueError("Worker does not own this attempt")
                case_id = str(worker["current_case_id"])
                archive = c.incoming_dir / f"{attempt_id}.zip"
                if not archive.is_file() or _sha256_file(archive) != archive_hash:
                    raise ValueError("Staged archive missing or changed")
                artifact_path, _ = c._accept_archive(attempt_id, archive, manifest_hash, expected_case_id=case_id)
                journal = {"identity": identity, "case_id": case_id, "artifact_path": str(artifact_path)}

            artifact_path.relative_to(c.results_dir)
            manifest_path = artifact_path / case_runner.MANIFEST_FILENAME
            if _sha256_file(manifest_path) != manifest_hash:
                raise ValueError("Committed manifest bytes changed")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed" or manifest.get("dry_run") is not False:
                raise ValueError("Controlled evaluation requires completed artifacts, not a dry-run manifest")
            c._verify_manifest_artifacts(artifact_path, manifest)
            digest = verify_artifacts(evidence, permit, artifact_path, aliases)
            if digest != control.get("evidence_sha256"):
                raise ValueError("Completion evidence digest mismatch")
            # Common evidence must commit every project-required file and the
            # manifest, not merely an attacker-chosen subset of good files.
            required = {case_runner.MANIFEST_FILENAME}
            required.update(item["path"] for item in manifest.get("artifacts", {}).values())
            if not required.issubset(set(aliases.values())):
                raise ValueError("Shore evidence omits project artifacts")
            for relative in aliases.values():
                with (artifact_path / relative).open("r+b") as stream:
                    os.fsync(stream.fileno())
            with self.transaction() as db:
                if db.execute("SELECT 1 FROM control_uploads WHERE attempt_id=? AND status='writing'",
                              (attempt_id,)).fetchone():
                    raise ValueError("An artifact upload is still in progress")
                db.execute("INSERT OR IGNORE INTO control_collection VALUES(?,?,0)",
                           (attempt_id, json.dumps(journal, sort_keys=True)))
            state.approve_control_completion(attempt_id, digest)
            state.complete_task(c.run_id, worker_id, attempt_id, lease,
                                artifact_path=str(artifact_path), manifest_sha256=manifest_hash)
            ack = gate.ack_evidence(envelope["permit"], evidence, artifact_path, aliases=aliases)
            if ack.get("status") != "RESOLVED":
                raise ValueError("Worker and collector receipts are not yet reconciled")
            gate.settle_evaluation(f"{c.run_id}:{case_id}", c.run_id, case_id, attempt_id)
            receipt = {**identity, "run_id": c.run_id, "artifact_path": str(artifact_path),
                       "case_id": case_id, "shore_status": "RESOLVED"}
            _write_json_atomic(c._completion_receipt_path(attempt_id), receipt)
            with self.transaction() as db:
                db.execute("UPDATE control_collection SET settled=1 WHERE attempt_id=?", (attempt_id,))
            # Keep the received ZIP in the experimental collector for restart
            # reconciliation; worker cleanup has its own signed RESOLVED check.
            return c._reply(request, "completed_ack", {"artifact_path": str(artifact_path),
                            "manifest_sha256": manifest_hash, "shore_status": "RESOLVED",
                            "duplicate": old is not None, "progress": state.progress(c.run_id).as_dict()})
