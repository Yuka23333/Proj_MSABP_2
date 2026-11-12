"""Known synthetic terminal outcomes; unknown external starts never use this.

The one small durable termination/tombstone record is transported as exact
bytes. Both sides validate those bytes and the shared Shore evidence. This is
neither a process-death detector nor authorization to refund an uncertain run.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat


MAX_TERMINAL_RECORD_BYTES = 64 * 1024
TERMINAL_RECORD_FIELDS = {
    "synthetic", "permit_id", "execution_id", "state", "failure_code", "error_message",
}


class TerminalRecordTooLarge(ValueError):
    """A terminal record cannot be consumed within the protocol byte budget."""


def bounded_terminal_record(record: dict) -> dict:
    """Fit the complete canonical record, retaining an original-message proof.

    The schema is unchanged: a truncated error_message ends with an explicit
    marker, its original UTF-8 byte length, and SHA-256 of those original bytes.
    The retained prefix ends on a Unicode codepoint boundary. The byte budget
    includes every field, JSON escaping, and the marker itself. No extra
    artifact is created, because terminal evidence has a strict alias contract.
    """
    from shore_control.crypto import canonical_json

    record = dict(record)
    if set(record) != TERMINAL_RECORD_FIELDS:
        raise ValueError("Terminal record must have exactly the canonical provenance fields")
    message = record["error_message"]
    if message is not None and not isinstance(message, str):
        raise ValueError("Terminal error_message must be a string or null")
    if len(canonical_json(record)) <= MAX_TERMINAL_RECORD_BYTES:
        return record
    if message is None:
        raise TerminalRecordTooLarge("Terminal metadata exceeds the explicit 64 KiB bound")
    original = message.encode("utf-8")
    marker = (f"\n[truncated; original_utf8_bytes={len(original)}; "
              f"original_sha256={hashlib.sha256(original).hexdigest()}]")
    record["error_message"] = marker
    if len(canonical_json(record)) > MAX_TERMINAL_RECORD_BYTES:
        raise TerminalRecordTooLarge("Terminal metadata and summary proof exceed the explicit 64 KiB bound")
    low, high = 0, len(message)
    while low < high:
        middle = (low + high + 1) // 2
        record["error_message"] = message[:middle] + marker
        if len(canonical_json(record)) <= MAX_TERMINAL_RECORD_BYTES:
            low = middle
        else:
            high = middle - 1
    record["error_message"] = message[:low] + marker
    return record


def validate_terminal_record(data: bytes, evidence: dict, permit: dict) -> dict:
    """Verify actual canonical record bytes, not an unbacked claimed digest."""
    from shore_control.crypto import canonical_json
    from shore_control.evidence import load_evidence_json, validate_evidence

    validate_evidence(evidence, permit)
    if not isinstance(data, bytes) or not data:
        raise ValueError("Terminal record must contain canonical bytes")
    if len(data) > MAX_TERMINAL_RECORD_BYTES:
        raise TerminalRecordTooLarge("Terminal record exceeds the explicit 64 KiB bound")
    record = load_evidence_json(data)
    if set(record) != TERMINAL_RECORD_FIELDS or canonical_json(record) != data:
        raise ValueError("Terminal record must have exactly the canonical provenance fields")
    outcome = evidence["outcome"]
    if outcome not in {"failed", "not_started"} or record["synthetic"] is not True:
        raise ValueError("Only known synthetic failed/not-started records are supported")
    expected = evidence["result"]
    if (record["permit_id"] != permit["permit_id"] or record["state"] != outcome
            or record["execution_id"] != expected["execution_id"]
            or record["failure_code"] != expected["failure_code"]
            or (record["error_message"] is not None and not isinstance(record["error_message"], str))):
        raise ValueError("Terminal record differs from the independently verified operation")
    field = "termination_record_sha256" if outcome == "failed" else "tombstone_sha256"
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected[field]:
        raise ValueError("Terminal record actual bytes differ from the evidence commitment")
    if outcome == "failed":
        artifact = {"path": "termination.json", "size_bytes": len(data), "sha256": actual}
        if evidence["artifacts"] != [artifact]:
            raise ValueError("Synthetic failure evidence must include its exact termination record")
    elif record["execution_id"] is not None or record["failure_code"] is not None:
        raise ValueError("A not-started tombstone cannot claim an execution")
    return record


def terminal_record_bytes(result: dict, permit: dict) -> bytes:
    """Recheck durable worker bytes before report and before HTTP delivery."""
    from shore_control.evidence import verify_artifacts

    outcome = result["evidence"]["outcome"]
    filename = "termination.json" if outcome == "failed" else "tombstone.json"
    expected_aliases = {"termination.json": "termination.json"} if outcome == "failed" else {}
    if result["aliases"] != expected_aliases:
        raise ValueError("Unexpected terminal record aliases")
    root = Path(result["root"])
    _safe_directory(root)
    path = root / filename
    data = _read_record(path)
    validate_terminal_record(data, result["evidence"], permit)
    verified = verify_artifacts(result["evidence"], permit, root, result["aliases"])
    if verified != result["receipt"]["evidence_sha256"]:
        raise ValueError("Terminal evidence differs from durable worker receipt")
    return data


def terminal_payload(assignment: dict, result: dict, permit: dict) -> dict:
    data = terminal_record_bytes(result, permit)
    return {
        "attempt_id": assignment["attempt_id"], "lease_token": assignment["lease_token"],
        "control_authority": {"permit_id": permit["permit_id"],
                              "evidence": result["evidence"], "aliases": result["aliases"],
                              "evidence_sha256": result["receipt"]["evidence_sha256"],
                              "outcome": result["receipt"]["outcome"]},
        "terminal_record_b64": base64.b64encode(data).decode("ascii"),
    }


def _safe_directory(root: Path) -> None:
    if not root.is_absolute() or ".." in root.parts or str(root).startswith("\\\\"):
        raise ValueError("Terminal record requires an absolute local artifact root")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        info = current.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("Terminal path cannot contain symlinks or reparse points")


def _safe_file(path: Path) -> None:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError("Terminal record must be a regular non-reparse file")
    if info.st_size > MAX_TERMINAL_RECORD_BYTES:
        raise TerminalRecordTooLarge("Terminal record exceeds the explicit 64 KiB bound")
    if not info.st_size:
        raise ValueError("Terminal record must contain canonical bytes")


def _read_record(path: Path) -> bytes:
    # Revision-2's shared reader pins POSIX parent handles and verifies the
    # opened Windows handle's final path. Tombstones are intentionally not
    # artifacts in Shore's schema, but still need the identical safe byte read.
    from shore_control.evidence import _open_artifact

    _safe_directory(path.parent)
    _safe_file(path)
    with _open_artifact(path) as stream:
        before = os.fstat(stream.fileno())
        data = stream.read(MAX_TERMINAL_RECORD_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (len(data) != before.st_size or len(data) > MAX_TERMINAL_RECORD_BYTES
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError("Terminal record changed while being read")
    return data


def _persist_record(path: Path, data: bytes) -> None:
    # Check every existing parent before creating a child, so a pre-existing
    # junction cannot redirect even the creation of the artifact directory.
    root = path.parent
    if not root.is_absolute() or ".." in root.parts or str(root).startswith("\\\\"):
        raise ValueError("Terminal record requires an absolute local artifact root")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        _safe_directory(current)
    _safe_directory(path.parent)
    if path.exists() or path.is_symlink():
        _safe_file(path)
        if _read_record(path) != data:
            raise ValueError("Durable collector terminal record changed")
        return
    temporary = path.with_name(path.name + "." + secrets.token_hex(16) + ".tmp")
    with temporary.open("xb") as stream:
        if os.name == "nt":
            from shore_control.evidence import _verify_windows_handle

            _verify_windows_handle(stream.fileno(), temporary)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ControlFailureCollector:
    """Reconcile known terminal records using the existing collector journal."""

    def __init__(self, collector):
        self.collector = collector
        self.coordinator = collector.coordinator

    def complete(self, request: dict) -> dict:
        from shore_control.evidence import verify_artifacts
        from shore_control.permits import verify_permit
        from .princess import _required_text, _safe_identifier, _write_json_atomic

        c, collector = self.coordinator, self.collector
        state, gate = c.state, c.state.control_authority
        if gate is None:
            raise ValueError("Terminal reconciliation requires the bound collector session")
        payload = request["payload"]
        if set(payload) != {"attempt_id", "lease_token", "control_authority", "terminal_record_b64"}:
            raise ValueError("Unexpected terminal reconciliation fields")
        worker_id = _safe_identifier(request["worker_id"], "worker_id")
        attempt_id = _safe_identifier(_required_text(payload, "attempt_id"), "attempt_id")
        lease = _required_text(payload, "lease_token")
        control = payload["control_authority"]
        if (not isinstance(control, dict) or set(control) != {
                "permit_id", "evidence", "aliases", "evidence_sha256", "outcome"}
                or control["outcome"] not in {"failed", "not_started"}):
            raise ValueError("Expected an explicit known terminal outcome")
        encoded = payload["terminal_record_b64"]
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_TERMINAL_RECORD_BYTES + 2) // 3):
            raise ValueError("Terminal record exceeds transport bounds")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid exact terminal record encoding") from exc
        envelope = state.control_for_attempt(attempt_id)
        if envelope is None:
            raise ValueError("No durable permit for terminal attempt")
        permit = verify_permit(gate.authority_public_key, envelope["permit"], envelope["descriptor"])
        if (permit["worker_id"] != worker_id or permit["run_id"] != c.run_id
                or permit["attempt_id"] != attempt_id or control["permit_id"] != permit["permit_id"]):
            raise ValueError("Terminal result belongs to a different operation")
        evidence, aliases, outcome = control["evidence"], control["aliases"], control["outcome"]
        if not isinstance(evidence, dict) or evidence.get("outcome") != outcome:
            raise ValueError("Terminal evidence outcome mismatch")
        record = validate_terminal_record(data, evidence, permit)
        expected_aliases = {"termination.json": "termination.json"} if outcome == "failed" else {}
        if aliases != expected_aliases:
            raise ValueError("Terminal aliases must match the exact durable record")
        filename = "termination.json" if outcome == "failed" else "tombstone.json"
        identity = {"terminal": outcome, "worker_id": worker_id, "attempt_id": attempt_id,
                    "lease_token_sha256": hashlib.sha256(lease.encode()).hexdigest(),
                    "record_sha256": hashlib.sha256(data).hexdigest(), "control_authority": control}
        # Validate the frozen policy before recording any collector receipt.
        policy = gate._policy()
        count_failure = outcome == "failed" and "failed" in policy["settled_outcomes"]
        with c._attempt_lock(attempt_id):
            with collector.transaction() as db:
                old = db.execute("SELECT * FROM control_collection WHERE attempt_id=?", (attempt_id,)).fetchone()
                if db.execute("SELECT 1 FROM control_uploads WHERE attempt_id=? AND status='writing'",
                              (attempt_id,)).fetchone():
                    raise ValueError("Cannot reconcile while upload bytes are unfinished")
            if old:
                journal = json.loads(old["request_json"])
                if journal["identity"] != identity:
                    raise ValueError("Terminal replay changed its durable reconciliation input")
                root, case_id = Path(journal["artifact_path"]), journal["case_id"]
            else:
                state.heartbeat(c.run_id, worker_id, attempt_id, lease,
                                lease_seconds=c.lease_seconds, phase="verifying_terminal_evidence")
                worker = state.get_worker(c.run_id, worker_id)
                if worker.get("current_attempt_id") != attempt_id:
                    raise ValueError("Worker does not own this terminal attempt")
                case_id = str(worker["current_case_id"])
                root = c.results_dir / ".shore-terminal" / attempt_id
                journal = {"identity": identity, "case_id": case_id, "artifact_path": str(root)}
            root.relative_to(c.results_dir)
            _persist_record(root / filename, data)
            # Re-read the actual persisted bytes; never acknowledge just payload hashes.
            persisted = _read_record(root / filename)
            validate_terminal_record(persisted, evidence, permit)
            checked = verify_artifacts(evidence, permit, root, aliases)
            if checked != control["evidence_sha256"]:
                raise ValueError("Terminal evidence digest mismatch")
            with collector.transaction() as db:
                if db.execute("SELECT 1 FROM control_uploads WHERE attempt_id=? AND status='writing'",
                              (attempt_id,)).fetchone():
                    raise ValueError("An artifact upload started during terminal verification")
                db.execute("INSERT OR IGNORE INTO control_collection VALUES(?,?,0)",
                           (attempt_id, json.dumps(journal, sort_keys=True)))
            ack = gate.ack_evidence(envelope["permit"], evidence, root, aliases=aliases)
            if ack.get("status") != "RESOLVED":
                raise ValueError("Both signed terminal receipts must agree before state settlement")
            state.approve_control_completion(attempt_id, checked)
            settled_case = state.settle_control_terminal(
                c.run_id, worker_id, attempt_id, lease, outcome=outcome,
                evidence_sha256=checked, artifact_path=str(root))
            if str(settled_case) != case_id:
                raise ValueError("Terminal state settled a different frozen case")
            if count_failure:
                gate.settle_evaluation(f"{c.run_id}:{case_id}", c.run_id, case_id, attempt_id)
            receipt = {**identity, "run_id": c.run_id, "artifact_path": str(root),
                       "case_id": case_id, "shore_status": "RESOLVED",
                       "budget_settled": count_failure, "record": record}
            _write_json_atomic(c._completion_receipt_path(attempt_id), receipt)
            with collector.transaction() as db:
                db.execute("UPDATE control_collection SET settled=1 WHERE attempt_id=?", (attempt_id,))
            return c._reply(request, "terminal_ack", {"outcome": outcome, "shore_status": "RESOLVED",
                "budget_settled": count_failure, "duplicate": old is not None,
                "progress": state.progress(c.run_id).as_dict()})
