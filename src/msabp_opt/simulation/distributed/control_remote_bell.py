"""Independent, outbound-only synthetic Bell with a durable wake journal.

The launch boundary is deliberately conservative: STARTING is committed before
calling the fixed launcher. An exception, dead PID or restart never authorizes
another launch. Only actual Maid-signed ready bytes can settle that boundary.
No SSH, arbitrary command, legacy Bell transport or solver is used here.
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
import subprocess
from typing import Any, Callable, Mapping

from .control_failure import _persist_record, _read_record, _safe_directory
from .control_worker import ExecutionUncertain


MODE = "shore-remote-synthetic-v1"
STARTUP_DOMAIN = "msabp.synthetic-bell-startup"
READY_DOMAIN = "msabp.synthetic-maid-ready"
TOMBSTONE_DOMAIN = "msabp.synthetic-bell-tombstone"
IDLE_DOMAIN = "msabp.synthetic-worker-idle"
MAX_WAKE_BYTES = 64 * 1024
_STARTUP_FIELDS = {"schema", "schema_version", "mode", "launch_id", "runtime_sha256",
                   "control_authority", "acceptance", "worker_id", "target_actor_id"}
_READY_FIELDS = {"schema", "schema_version", "launch_id", "permit_id", "runtime_sha256",
                 "actor_id", "device_id", "fleet_id", "job_id", "epoch", "run_id", "pid",
                 "process_started_utc"}
_TOMBSTONE_FIELDS = {"schema", "schema_version", "mode", "permit_id", "worker_id",
                     "target_actor_id", "runtime_sha256", "state"}
_DYNAMIC_FIELDS = {"operation_id", "run_id", "worker_id", "attempt_id", "action",
                   "input_sha256", "device_id", "worker_role", "worker_registration_sha256",
                   "action_context"}


def _bytes(value: Any) -> bytes:
    from shore_control.crypto import canonical_json

    return canonical_json(value)


def _json(value: Any) -> str:
    return _bytes(value).decode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(_json(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _record(path: Path) -> tuple[dict, bytes]:
    from shore_control.evidence import load_evidence_json

    raw = _read_record(path)
    value = load_evidence_json(raw)
    _require(_bytes(value) == raw, "Wake record must contain exact canonical JSON bytes")
    return value, raw


def _same_device(bell: dict, maid: dict) -> None:
    _require(bell["role"] == "bell" and maid["role"] == "maid",
             "Wake requires independently registered Bell and Maid roles")
    _require(all(bell[k] == maid[k] for k in ("device_id", "tailnet_node_id")),
             "Bell can wake only its same-device registered Maid")


def verify_wake_evidence(evidence: dict, permit: dict, root: str | Path, *,
                         authority_public_key: Any, bell_registration: dict,
                         target_registration: dict) -> str:
    """Verify actual immutable signed records before acknowledgement/checkpoint.

    ``permit`` must already be independently verified by the caller. This adds
    reality-record checks that Shore's generic evidence schema cannot provide.
    A PID is informational only and is never queried as evidence of readiness.
    """
    from shore_control.crypto import verify_object
    from shore_control.evidence import validate_evidence, verify_artifacts
    from shore_control.permits import DESCRIPTOR_FIELDS, verify_acceptance, verify_permit
    from shore_control.registry import registration_sha256, validate_workers

    evidence = validate_evidence(evidence, permit)
    _require(permit["action"] == "wake", "Expected a wake permit")
    target = permit["action_context"]["target_actor_id"]
    registry = validate_workers({permit["worker_id"]: bell_registration, target: target_registration})
    bell, maid = registry[permit["worker_id"]], registry[target]
    _same_device(bell, maid)
    _require(permit["worker_registration_sha256"] == registration_sha256(bell)
             and permit["device_id"] == bell["device_id"], "Wake registration differs from permit")
    root = Path(root)
    _safe_directory(root)
    state = evidence["result"]["state"]
    if state == "not_started":
        envelope, raw = _record(root / "tombstone.json")
        value = verify_object(bell["public_key"], TOMBSTONE_DOMAIN, envelope)
        _require(set(value) == _TOMBSTONE_FIELDS and value["schema"] == TOMBSTONE_DOMAIN
                 and type(value["schema_version"]) is int and value["schema_version"] == 1
                 and value["mode"] == MODE and value["state"] == "not_started",
                 "Invalid signed Bell tombstone")
        _require(value["permit_id"] == permit["permit_id"]
                 and value["worker_id"] == permit["worker_id"]
                 and value["target_actor_id"] == target
                 and value["runtime_sha256"] == permit["input_sha256"],
                 "Bell tombstone differs from wake permit")
        _require(hashlib.sha256(raw).hexdigest() == evidence["result"]["tombstone_sha256"],
                 "Tombstone bytes differ from wake evidence")
    elif state == "started":
        _require([a["path"] for a in evidence["artifacts"]] == ["ready.json", "startup.json"],
                 "Wake success requires its exact actual startup and ready artifacts")
        startup_envelope, startup_raw = _record(root / "startup.json")
        ready_envelope, ready_raw = _record(root / "ready.json")
        startup = verify_object(bell["public_key"], STARTUP_DOMAIN, startup_envelope)
        ready = verify_object(maid["public_key"], READY_DOMAIN, ready_envelope)
        _require(set(startup) == _STARTUP_FIELDS and startup["schema"] == STARTUP_DOMAIN
                 and type(startup["schema_version"]) is int and startup["schema_version"] == 1
                 and startup["mode"] == MODE, "Invalid signed Bell startup record")
        _require(set(ready) == _READY_FIELDS and ready["schema"] == READY_DOMAIN
                 and type(ready["schema_version"]) is int and ready["schema_version"] == 1,
                 "Invalid signed Maid ready record")
        gate = startup["control_authority"]
        _require(type(gate) is dict and set(gate) == {"descriptor", "permit"},
                 "Startup requires the original permit and descriptor")
        descriptor = {key: permit[key] for key in DESCRIPTOR_FIELDS}
        _require(_bytes(gate["descriptor"]) == _bytes(descriptor), "Startup descriptor changed")
        _require(verify_permit(authority_public_key, gate["permit"], descriptor) == permit,
                 "Startup contains another wake permit")
        verify_acceptance(authority_public_key, startup["acceptance"], permit)
        launch_id = startup["launch_id"]
        _require(type(launch_id) is str and re.fullmatch(r"[a-f0-9]{32}", launch_id) is not None,
                 "Invalid launch identifier")
        _require(startup["worker_id"] == permit["worker_id"]
                 and startup["target_actor_id"] == target
                 and startup["runtime_sha256"] == permit["input_sha256"]
                 and evidence["result"]["launch_id"] == launch_id,
                 "Startup target, runtime or launch differs from permit")
        expected = {"launch_id": launch_id, "permit_id": permit["permit_id"],
                    "runtime_sha256": permit["input_sha256"], "actor_id": target,
                    **{k: permit[k] for k in ("device_id", "fleet_id", "job_id", "epoch", "run_id")}}
        _require(all(_bytes(ready[k]) == _bytes(v) for k, v in expected.items()),
                 "Maid ready record belongs to another launch or job")
        _require(type(ready["pid"]) is int and ready["pid"] > 0
                 and type(ready["process_started_utc"]) is str
                 and 1 <= len(ready["process_started_utc"]) <= 64,
                 "Invalid informational Maid process metadata")
        for field, raw in (("startup_record_sha256", startup_raw), ("ready_record_sha256", ready_raw)):
            _require(evidence["result"][field] == hashlib.sha256(raw).hexdigest(),
                     "Actual wake record differs from evidence commitment")
    else:
        # No PID-derived already_running and no guessed failed outcome.
        raise ValueError("Synthetic Bell supports only proven started or explicit not_started")
    return verify_artifacts(evidence, permit, root)


class WakeCoordinator:
    """Persistent Bell-side inbox, acceptance, launch intent and dual receipts."""

    def __init__(self, root: str | Path, client: Any, authority_public_key: Any, *,
                 binding: Mapping[str, Any], registration: Mapping[str, Any],
                 target_actor_id: str, target_registration: Mapping[str, Any],
                 runtime: Mapping[str, Any], run_id: str) -> None:
        from shore_control.crypto import fingerprint
        from shore_control.permits import DESCRIPTOR_FIELDS
        from shore_control.registry import validate_workers
        from .control_rpc import SignedRpcJournal

        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        _safe_directory(self.root)
        self.database = self.root / "wake.sqlite"
        self.client, self.authority_public_key = client, authority_public_key
        self.binding, self.runtime = _copy(dict(binding)), _copy(dict(runtime))
        _require(len(_bytes(self.runtime)) <= MAX_WAKE_BYTES,
                 "Pinned runtime exceeds the small synthetic wake record budget")
        self.run_id, self.target_actor_id = run_id, target_actor_id
        _require(set(self.binding) == DESCRIPTOR_FIELDS - _DYNAMIC_FIELDS,
                 "Bell requires a complete independently pinned job binding")
        _require(client.fleet_id == self.binding["fleet_id"]
                 and client.authority_id == fingerprint(authority_public_key),
                 "Bell client must pin the same authority and fleet")
        regs = validate_workers({client.actor_id: dict(registration), target_actor_id: dict(target_registration)})
        self.registration, self.target_registration = regs[client.actor_id], regs[target_actor_id]
        _same_device(self.registration, self.target_registration)
        self.runtime_sha256 = _digest(self.runtime)
        self.rpc = SignedRpcJournal(self.root / "rpc.sqlite")
        self.pins = {"binding": self.binding, "runtime_sha256": self.runtime_sha256,
                     "run_id": run_id, "actor_id": client.actor_id, "registry": regs,
                     "authority_id": client.authority_id}
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS pins(value TEXT NOT NULL)")
            old = db.execute("SELECT value FROM pins").fetchone()
            if old is None:
                db.execute("INSERT INTO pins VALUES(?)", (_json(self.pins),))
            else:
                _require(old["value"] == _json(self.pins), "Persisted Bell identity/runtime/binding cannot change")
            db.execute("""CREATE TABLE IF NOT EXISTS wake(
                attempt_id TEXT PRIMARY KEY, permit_id TEXT UNIQUE NOT NULL,
                assignment TEXT NOT NULL, acceptance TEXT, state TEXT NOT NULL,
                launch_id TEXT, result TEXT, resolved TEXT)""")

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

    def record(self, attempt_id: str) -> dict:
        with self.transaction() as db:
            value = db.execute("SELECT * FROM wake WHERE attempt_id=?", (attempt_id,)).fetchone()
        _require(value is not None, "Wake has no durable local inbox")
        return dict(value)

    def _permit(self, record: dict) -> dict:
        from shore_control.permits import verify_permit

        gate = json.loads(record["assignment"])["control_authority"]
        return verify_permit(self.authority_public_key, gate["permit"], gate["descriptor"])

    def prepare(self, assignment: dict) -> dict:
        """Durably receive an issued permit without accepting or launching it."""
        from shore_control.permits import verify_permit
        from shore_control.registry import registration_sha256

        _require(type(assignment) is dict and set(assignment) == {"control_authority", "runtime"},
                 "Unexpected wake assignment fields")
        _require(_bytes(assignment["runtime"]) == _bytes(self.runtime), "Wake runtime differs from pinned manifest")
        gate = assignment["control_authority"]
        _require(type(gate) is dict and set(gate) == {"descriptor", "permit"}
                 and type(gate["descriptor"]) is dict, "Wake assignment requires a complete Shore gate")
        descriptor = gate["descriptor"]
        expected = {**self.binding, "operation_id": descriptor.get("operation_id"),
                    "run_id": self.run_id, "worker_id": self.client.actor_id,
                    "attempt_id": descriptor.get("attempt_id"), "action": "wake",
                    "input_sha256": self.runtime_sha256, "device_id": self.registration["device_id"],
                    "worker_role": "bell", "worker_registration_sha256": registration_sha256(self.registration),
                    "action_context": {"target_actor_id": self.target_actor_id,
                                       "runtime_sha256": self.runtime_sha256}}
        _require(_bytes(descriptor) == _bytes(expected), "Wake descriptor differs from pinned work")
        permit = verify_permit(self.authority_public_key, gate["permit"], expected)
        attempt_id, permit_id = permit["attempt_id"], permit["permit_id"]
        with self.transaction() as db:
            old = db.execute("SELECT * FROM wake WHERE attempt_id=?", (attempt_id,)).fetchone()
            if old is None:
                db.execute("INSERT INTO wake(attempt_id,permit_id,assignment,state) VALUES(?,?,?,'PREPARED')",
                           (attempt_id, permit_id, _json(assignment)))
            else:
                _require(old["assignment"] == _json(assignment), "Repeated wake changed its durable assignment")
        return permit

    def accept(self, assignment: dict) -> dict:
        from shore_control.permits import verify_acceptance

        permit = self.prepare(assignment)
        attempt_id, permit_id = permit["attempt_id"], permit["permit_id"]
        gate = assignment["control_authority"]
        record = self.record(attempt_id)
        if record["acceptance"]:
            verify_acceptance(self.authority_public_key, json.loads(record["acceptance"]), permit)
        elif not record["result"]:
            acceptance = self.rpc.call(self.client, "accept_permit",
                {k: permit[k] for k in ("permit_id", "operation_id", "run_id", "attempt_id",
                                        "action", "input_sha256", "device_id")},
                key="accept:" + permit_id, recovery_permit=gate["permit"])
            verify_acceptance(self.authority_public_key, acceptance, permit)
            with self.transaction() as db:
                db.execute("UPDATE wake SET acceptance=?,state='ACCEPTED' WHERE attempt_id=? AND state='PREPARED'",
                           (_json(acceptance), attempt_id))
        return permit

    def draining(self) -> bool:
        """A signed bound-job snapshot, never a collector hint or timeout."""
        status = self.rpc.fresh_read(self.client, "status", {})
        job = status.get("bound_job")
        _require(type(job) is dict and status.get("fleet_id") == self.client.fleet_id
            and all(job.get(k) == self.binding[k] for k in ("job_id", "epoch", "host_id", "session_id"))
            and job.get("scope") == {k: self.binding[k] for k in
                ("project_id", "collector_url", "data_root", "job_spec")},
            "Wake status differs from its pinned bound job")
        _require(job["phase"] in {"ACTIVE", "DRAINING", "PAUSED_BOUND"}, "Unexpected wake job phase")
        return job["phase"] != "ACTIVE"

    def reconcile_terminal_accept(self, attempt_id: str) -> None:
        """Keep and settle a lost acceptance request after proven not_started."""
        from .control_rpc import VerifiedRpcError

        record = self.record(attempt_id)
        if record["state"] != "RESOLVED" or self.checked_result(attempt_id)["receipt"]["outcome"] != "not_started":
            return
        gate = json.loads(record["assignment"])["control_authority"]
        for pending in self.rpc.pending():
            if pending["key"] != "accept:" + record["permit_id"]:
                continue
            try:
                self.rpc.call(self.client, "accept_permit", pending["params"],
                              key=pending["key"], recovery_permit=gate["permit"])
            except VerifiedRpcError as exc:
                # The authenticated refusal remains in the original journal.
                if exc.code not in {"WORK_CLOSED", "NOT_ACTIVE", "FENCED"}:
                    raise

    def launch_paths(self, attempt_id: str) -> dict[str, Path]:
        record = self.record(attempt_id)
        root = self.root / "outbox" / record["permit_id"]
        return {"root": root, "runtime": root / "runtime.json", "startup": root / "startup.json",
                "ready": root / "ready.json"}

    def launch(self, attempt_id: str, launcher: Callable[..., Any], *, ready_path: str | Path) -> None:
        """Commit intent then call one injected, fixed executable launcher once."""
        from shore_control.permits import verify_acceptance

        record = self.record(attempt_id)
        permit = self._permit(record)
        _require(record["acceptance"] is not None, "Wake launch requires durable authority acceptance")
        verify_acceptance(self.authority_public_key, json.loads(record["acceptance"]), permit)
        launch_id = secrets.token_hex(16)
        with self.transaction() as db:
            current = db.execute("SELECT state FROM wake WHERE attempt_id=?", (attempt_id,)).fetchone()
            if current["state"] != "ACCEPTED":
                raise ExecutionUncertain("Wake already STARTING or terminal; never relaunch")
            previous = db.execute("SELECT * FROM wake WHERE attempt_id!=? AND launch_id IS NOT NULL",
                                  (attempt_id,)).fetchall()
            if previous:
                raise ExecutionUncertain("An earlier Maid launch exists; no PID-derived restart is permitted")
            db.execute("UPDATE wake SET state='STARTING',launch_id=? WHERE attempt_id=?", (launch_id, attempt_id))
        startup = self.client.signer.sign(STARTUP_DOMAIN, {
            "schema": STARTUP_DOMAIN, "schema_version": 1, "mode": MODE,
            "launch_id": launch_id, "runtime_sha256": self.runtime_sha256,
            "control_authority": json.loads(record["assignment"])["control_authority"],
            "acceptance": json.loads(record["acceptance"]), "worker_id": self.client.actor_id,
            "target_actor_id": self.target_actor_id})
        paths = self.launch_paths(attempt_id)
        # _persist_record checks canonical local paths and refuses overwrites.
        _persist_record(paths["runtime"], _bytes(self.runtime))
        _persist_record(paths["startup"], _bytes(startup))
        # Any exception after STARTING intentionally leaves an unresolved intent.
        launcher(paths["runtime"], paths["startup"], Path(ready_path))

    def _evidence(self, permit: dict, *, state: str, artifacts: list,
                  launch_id: str | None = None, startup_hash: str | None = None,
                  ready_hash: str | None = None, tombstone_hash: str | None = None) -> dict:
        from shore_control.permits import DESCRIPTOR_FIELDS

        return {"schema": "shore.operation-evidence", "schema_version": 1, "kind": "wake",
                **{k: permit[k] for k in ("fleet_id", "job_id", "epoch", "permit_id", "worker_id",
                                          "device_id", "input_sha256")},
                "descriptor_sha256": _digest({k: permit[k] for k in DESCRIPTOR_FIELDS}),
                "outcome": "completed" if state == "started" else "not_started", "artifacts": artifacts,
                "result": {"state": state, "target_actor_id": self.target_actor_id,
                           "runtime_sha256": self.runtime_sha256, "launch_id": launch_id,
                           "startup_record_sha256": startup_hash, "ready_record_sha256": ready_hash,
                           "failure_code": None, "termination_record_sha256": None,
                           "tombstone_sha256": tombstone_hash}}

    def _save(self, attempt_id: str, permit: dict, evidence: dict, expected_state: str) -> dict:
        root = self.launch_paths(attempt_id)["root"]
        sha = verify_wake_evidence(evidence, permit, root, authority_public_key=self.authority_public_key,
                                  bell_registration=self.registration, target_registration=self.target_registration)
        result = {"root": str(root), "evidence": evidence,
                  "receipt": {"permit_id": permit["permit_id"], "outcome": evidence["outcome"],
                              "durable": True, "evidence_sha256": sha}}
        with self.transaction() as db:
            changed = db.execute("UPDATE wake SET result=?,state='RESULT' WHERE attempt_id=? AND state=?",
                                 (_json(result), attempt_id, expected_state))
            _require(changed.rowcount == 1, "Concurrent wake transition; terminal records cannot change")
        return result

    def record_ready(self, attempt_id: str, ready_path: str | Path) -> dict:
        record = self.record(attempt_id)
        if record["result"]:
            return self.checked_result(attempt_id)
        _require(record["state"] == "STARTING", "Ready evidence requires an existing STARTING intent")
        permit, paths = self._permit(record), self.launch_paths(attempt_id)
        ready, raw = _record(Path(ready_path))
        _persist_record(paths["ready"], raw)
        startup, startup_raw = _record(paths["startup"])
        artifacts = [{"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                     for name, data in (("ready.json", raw), ("startup.json", startup_raw))]
        evidence = self._evidence(permit, state="started", artifacts=artifacts,
            launch_id=record["launch_id"], startup_hash=_digest(startup), ready_hash=_digest(ready))
        return self._save(attempt_id, permit, evidence, "STARTING")

    def not_started(self, attempt_id: str) -> dict:
        """Explicit cancellation only before this journal entered STARTING."""
        record, paths = self.record(attempt_id), self.launch_paths(attempt_id)
        if record["result"]:
            _require(json.loads(record["result"])["evidence"]["outcome"] == "not_started",
                     "A proven launch cannot be cancelled")
            return self.checked_result(attempt_id)
        _require(record["state"] in {"PREPARED", "ACCEPTED"} and record["launch_id"] is None,
                 "Unknown or started wake cannot be cancelled as not_started")
        permit = self._permit(record)
        tombstone = self.client.signer.sign(TOMBSTONE_DOMAIN, {
            "schema": TOMBSTONE_DOMAIN, "schema_version": 1, "mode": MODE,
            "permit_id": permit["permit_id"], "worker_id": self.client.actor_id,
            "target_actor_id": self.target_actor_id, "runtime_sha256": self.runtime_sha256,
            "state": "not_started"})
        _persist_record(paths["root"] / "tombstone.json", _bytes(tombstone))
        evidence = self._evidence(permit, state="not_started", artifacts=[], tombstone_hash=_digest(tombstone))
        return self._save(attempt_id, permit, evidence, record["state"])

    def checked_result(self, attempt_id: str) -> dict:
        record = self.record(attempt_id)
        if not record["result"]:
            raise ExecutionUncertain("No proven wake result; never infer completion from a PID")
        result, permit = json.loads(record["result"]), self._permit(record)
        sha = verify_wake_evidence(result["evidence"], permit, result["root"],
            authority_public_key=self.authority_public_key, bell_registration=self.registration,
            target_registration=self.target_registration)
        _require(sha == result["receipt"]["evidence_sha256"], "Wake outbox bytes changed")
        return result

    def completion_payload(self, attempt_id: str) -> dict:
        result = self.checked_result(attempt_id)
        names = ("startup", "ready") if result["evidence"]["outcome"] == "completed" else ("tombstone",)
        payload = {"attempt_id": attempt_id, "evidence": result["evidence"],
                   "records": {name: _record(Path(result["root"]) / (name + ".json"))[0] for name in names}}
        _require(len(_bytes(payload)) <= MAX_WAKE_BYTES, "Wake completion exceeds its bounded transport")
        return payload

    def report(self, attempt_id: str) -> dict:
        result, record = self.checked_result(attempt_id), self.record(attempt_id)
        gate = json.loads(record["assignment"])["control_authority"]
        return self.rpc.call(self.client, "worker_report", result["receipt"],
                             key="report:" + record["permit_id"], recovery_permit=gate["permit"])

    def resolved(self, attempt_id: str) -> bool:
        from shore_control.permits import verify_acceptance, verify_permit

        record, result = self.record(attempt_id), self.checked_result(attempt_id)
        permit = self._permit(record)
        status = self.rpc.fresh_read(self.client, "get_operation",
            {"epoch": permit["epoch"], "operation_id": permit["operation_id"]})
        gate = json.loads(record["assignment"])["control_authority"]
        _require(verify_permit(self.authority_public_key, status["permit"], gate["descriptor"]) == permit,
                 "Operation status refers to another wake permit")
        if status["status"] != "RESOLVED":
            return False
        if status.get("acceptance") is not None or result["receipt"]["outcome"] != "not_started":
            verify_acceptance(self.authority_public_key, status["acceptance"], permit)
        for name in ("worker_receipt", "controller_receipt"):
            peer = status.get(name)
            _require(type(peer) is dict and all(peer.get(k) == result["receipt"][k]
                     for k in ("outcome", "durable", "evidence_sha256")), "Wake dual receipts disagree")
        with self.transaction() as db:
            db.execute("UPDATE wake SET state='RESOLVED',resolved=? WHERE attempt_id=?",
                       (_json(status), attempt_id))
        return True

    def pending(self) -> list[dict]:
        with self.transaction() as db:
            return [dict(row) for row in db.execute("SELECT * FROM wake WHERE state!='RESOLVED' ORDER BY rowid")]


class FixedMaidLauncher:
    """Only the pinned Python and project synthetic Maid entrypoint are allowed."""

    def __init__(self, python_path: str | Path, source_root: str | Path, *,
                 popen_factory: Callable[..., Any] = subprocess.Popen) -> None:
        self.python_path = Path(python_path).absolute()
        self.source_root = Path(source_root).absolute()
        self.entrypoint = self.source_root / "scripts" / "simulation" / "shore_synthetic_maid.py"
        self.popen = popen_factory

    def __call__(self, runtime_path: Path, startup_path: Path, ready_path: Path) -> Any:
        from .control_failure import _safe_file

        for path in (self.python_path, self.entrypoint, runtime_path, startup_path):
            _safe_directory(path.parent)
            # Executables/source need not fit the small evidence byte limit.
            if path in (runtime_path, startup_path):
                _safe_file(path)
            else:
                _require(path.is_file() and not path.is_symlink(), "Pinned launch file is missing or redirected")
        _safe_directory(ready_path.parent)
        argv = [str(self.python_path), str(self.entrypoint), "--runtime-config", str(runtime_path),
                "--wake-startup", str(startup_path), "--ready-record", str(ready_path), "--run"]
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with (startup_path.parent / "maid.stdout.log").open("ab") as stdout, (
                startup_path.parent / "maid.stderr.log").open("ab") as stderr:
            return self.popen(argv, cwd=str(self.source_root), stdin=subprocess.DEVNULL,
                              stdout=stdout, stderr=stderr, shell=False, close_fds=True, creationflags=flags)


class BellRunner:
    """One bounded poll/reconcile iteration; no inbound Bell listener exists."""

    def __init__(self, coordinator: WakeCoordinator, collector: Any, launcher: Callable[..., Any], *,
                 ready_root: str | Path) -> None:
        self.coordinator, self.collector, self.launcher = coordinator, collector, launcher
        self.ready_root = Path(ready_root).absolute()

    def _ready_path(self, record: dict) -> Path:
        return self.ready_root / ("ready." + record["launch_id"] + ".json")

    def _deliver(self, attempt_id: str) -> dict:
        c = self.coordinator
        c.report(attempt_id)
        reply = self.collector.request_json("/v1/wake/complete", c.completion_payload(attempt_id))
        _require(type(reply) is dict and reply.get("permit_id") == c.record(attempt_id)["permit_id"]
                 and reply.get("status") in {"RESOLVED", "AWAITING_PEER_RECEIPT"},
                 "Collector did not acknowledge this exact wake")
        resolved = c.resolved(attempt_id)
        if resolved:
            c.reconcile_terminal_accept(attempt_id)
        return {"state": "RESOLVED" if resolved else "RESULT", "attempt_id": attempt_id}

    def report_idle(self) -> dict:
        """Sign this one job's durable empty-outbox proof, not a PID snapshot."""
        c = self.coordinator
        with c.transaction() as db:
            rows = db.execute("SELECT attempt_id,permit_id,state FROM wake ORDER BY permit_id").fetchall()
        _require(bool(rows) and all(row["state"] == "RESOLVED" for row in rows),
                 "Bell idle requires at least one wake and every durable row RESOLVED")
        for row in rows:
            c.reconcile_terminal_accept(row["attempt_id"])
        _require(not c.rpc.pending(), "Bell idle requires no pending signed authority RPC")
        record = c.client.signer.sign(IDLE_DOMAIN, {
            "schema": IDLE_DOMAIN, "schema_version": 1, "mode": MODE,
            "actor_id": c.client.actor_id, "fleet_id": c.client.fleet_id,
            "job_id": c.binding["job_id"], "epoch": c.binding["epoch"], "run_id": c.run_id,
            "runtime_sha256": c.runtime_sha256, "permit_ids": [row["permit_id"] for row in rows],
            "outboxes_pending": 0})
        reply = self.collector.request_json("/v1/control/idle", {"record": record})
        _require(reply == {"status": "RECORDED", "actor_id": c.client.actor_id},
                 "Collector did not record this actor's exact idle proof")
        return reply

    def step(self) -> dict:
        c = self.coordinator
        for record in c.pending():
            attempt_id = record["attempt_id"]
            if record["result"]:
                return self._deliver(attempt_id)
            if record["state"] == "STARTING":
                ready = self._ready_path(record)
                if ready.exists():
                    c.record_ready(attempt_id, ready)
                    return self._deliver(attempt_id)
                # No launch retry, no automatic cancellation and no PID probe.
                return {"state": "STARTING", "attempt_id": attempt_id}
            if c.draining():
                c.not_started(attempt_id)
                return self._deliver(attempt_id)
            if record["state"] == "PREPARED":
                c.accept(json.loads(record["assignment"]))
            return self._launch(attempt_id)
        reply = self.collector.request_json("/v1/wake/poll", {})
        _require(type(reply) is dict and set(reply) == {"assignment"}, "Invalid wake poll response")
        if reply["assignment"] is None:
            with c.transaction() as db:
                has_history = db.execute("SELECT 1 FROM wake LIMIT 1").fetchone() is not None
            if has_history:
                self.report_idle()
                return {"state": "IDLE_REPORTED"}
            return {"state": "IDLE"}
        permit = c.prepare(reply["assignment"])
        record = c.record(permit["attempt_id"])
        if record["result"]:
            return self._deliver(permit["attempt_id"])
        if c.draining():
            c.not_started(permit["attempt_id"])
            return self._deliver(permit["attempt_id"])
        c.accept(reply["assignment"])
        return self._launch(permit["attempt_id"])

    def _launch(self, attempt_id: str) -> dict:
        c = self.coordinator
        if c.draining():
            c.not_started(attempt_id)
            return self._deliver(attempt_id)
        self.ready_root.mkdir(parents=True, exist_ok=True)
        _safe_directory(self.ready_root)

        def launch(runtime, startup, _unused):
            # Launch id exists only after the durable STARTING commit.
            return self.launcher(runtime, startup, self._ready_path(c.record(attempt_id)))

        c.launch(attempt_id, launch, ready_path=self.ready_root / "unused.json")
        return {"state": "STARTING", "attempt_id": attempt_id}


def preflight_bell(config_path: str | Path) -> tuple[dict, Any, dict]:
    """Offline public-byte audit; no socket, daemon or private-key operation."""
    from .control_remote_worker import RemoteSyntheticMaidConfig, _json_bytes, checked_path, preflight_worker, read_json

    config, _ = read_json(config_path)
    fields = {"schema", "schema_version", "mode", "runtime_config_path", "runtime_sha256",
              "state_root", "poll_seconds"}
    _require(set(config) == fields and config["schema"] == "msabp.shore-remote-synthetic-bell"
             and type(config["schema_version"]) is int and config["schema_version"] == 1
             and config["mode"] == MODE, "Invalid explicit synthetic Bell configuration")
    _require(type(config["poll_seconds"]) is int and 1 <= config["poll_seconds"] <= 300,
             "Bell poll interval must be integer seconds in [1, 300]")
    runtime = RemoteSyntheticMaidConfig.load(config["runtime_config_path"], expected_sha256=config["runtime_sha256"])
    _require(hashlib.sha256(_json_bytes(dict(runtime.raw))).hexdigest() == runtime.runtime_sha256,
             "Bell runtime must use exact canonical bytes before signing its wake hash")
    root = checked_path(config["state_root"])
    _require(root != Path(root.anchor), "Bell state root cannot be a filesystem root")
    protected = [Path(runtime[name]) for name in ("source_root", "shore_source_root", "state_root", "output_root")]
    protected.append(Path(runtime["worker_config_path"]).parent)
    for path in protected:
        _require(root != path and root not in path.parents and path not in root.parents,
                 "Bell journal requires a separate controlled state root")
    for path in (Path(config_path), Path(config["runtime_config_path"]),
                 Path(runtime["bootstrap_path"]), Path(runtime["csv_path"])):
        _require(root != path and root not in path.parents, "Bell state must not contain its input configuration")
    report = preflight_worker(runtime)
    bootstrap = report["bootstrap"]
    from .control_remote_worker import BELL_ACTOR, MAID_ACTOR

    _same_device(bootstrap["workers"][BELL_ACTOR], bootstrap["workers"][MAID_ACTOR])
    return config, runtime, bootstrap


def run_bell(config_path: str | Path) -> None:
    """Explicit operational entry for a separately reviewed SYSTEM launcher.

    Installation, scheduled tasks and privilege changes are intentionally absent.
    This process serves one pinned job, retaining all uncertain/outbox state.
    """
    import sys
    import time
    from .control_remote_worker import BELL_ACTOR, LOCAL_NODE_ID, MAID_ACTOR, instance_lock

    config, runtime, bootstrap = preflight_bell(config_path)
    from shore_control.remote import DEFAULT_TAILSCALE, client_from_bootstrap
    from shore_control.tailnet import TailnetVerifier
    from .control_remote_http import RemotePrincessClient
    with instance_lock(config["state_root"]):
        client = client_from_bootstrap(bootstrap, BELL_ACTOR, local_node_id=LOCAL_NODE_ID,
                                      worker_config=runtime["worker_config_path"])
        network = {**bootstrap["network"], "local_node_id": LOCAL_NODE_ID,
                   "local_address": bootstrap["network"]["nodes"][LOCAL_NODE_ID]["address"]}
        verifier = TailnetVerifier(network, DEFAULT_TAILSCALE)
        collector = RemotePrincessClient(bootstrap, BELL_ACTOR, client.signer, verifier, run_id=runtime["run_id"])
        coordinator = WakeCoordinator(config["state_root"], client, bootstrap["authority_public_key"],
            binding=runtime["binding"], registration=bootstrap["workers"][BELL_ACTOR],
            target_actor_id=MAID_ACTOR, target_registration=bootstrap["workers"][MAID_ACTOR],
            runtime=runtime.raw, run_id=runtime["run_id"])
        runner = BellRunner(coordinator, collector, FixedMaidLauncher(sys.executable, runtime["source_root"]),
                            ready_root=runtime.state_root)
        while True:
            try:
                status = runner.step()
                print(json.dumps({"mode": MODE, **status}, sort_keys=True), flush=True)
            except Exception as exc:
                # Continue reconciliation under the same pinned job/journal;
                # no new operation, identity fallback or automatic relaunch.
                print(json.dumps({"mode": MODE, "state": "BLOCKED", "error": type(exc).__name__}), flush=True)
            time.sleep(config["poll_seconds"])
