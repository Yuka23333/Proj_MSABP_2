"""Project-side single-job orchestration for remote synthetic acceptance.

No SSH, deployment, service installation or solver entry exists here. Offline
preparation is separate from explicitly authorized bind/serve operations. An
uncertain wake is retained; neither a timeout nor a dead PID creates a new one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

from .control_failure import _persist_record, _safe_directory
from .control_remote_worker import (
    BELL_ACTOR, LOCAL_NODE_ID, MAID_ACTOR, MODE, PROJECT_SOURCE_FILES,
    SHORE_SOURCE_FILES, RemoteSyntheticMaidConfig, _json_bytes, checked_path,
    read_json, sha256_file,
)
from .princess import PrincessCoordinator
from .protocol import validate_message


CONTROLLER_ACTOR = "princess-24b88f2d-6ff7-415b-ae07-b8981a3418aa"
HOST_NODE = "n8BsWasrMb11CNTRL"
IDLE_DOMAIN = "msabp.synthetic-worker-idle"


def immutable_json(path, value):
    path = checked_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_directory(path.parent)
    _persist_record(path, _json_bytes(value))


class CollectorWakeQueue:
    """One fixed Bell/Maid wake, persisted before dispatch and never replaced."""

    def __init__(self, root, gate, runtime, *, terminal_only=False):
        self.root = checked_path(root)
        self.gate = gate
        self.terminal_only = terminal_only
        if terminal_only and any(not (self.root / name).is_file() for name in ("intent.json", "runtime.json")):
            raise ValueError("Receipt recovery requires the original wake journal")
        self.runtime = RemoteSyntheticMaidConfig.from_mapping(runtime)
        self.root.mkdir(parents=True, exist_ok=True)
        _safe_directory(self.root)
        self._lock = threading.RLock()
        binding = {**gate.binding, **gate.job["scope"], "fleet_id": gate.controller.fleet_id}
        if self.runtime["binding"] != binding:
            raise ValueError("Wake runtime differs from the acquired full job")
        if (gate.registrations[BELL_ACTOR]["tailnet_node_id"] != LOCAL_NODE_ID
                or gate.registrations[MAID_ACTOR]["tailnet_node_id"] != LOCAL_NODE_ID):
            raise ValueError("Only the pinned CoconutG2 actors are supported")
        self.intent = {"operation_id": self.runtime["run_id"] + ".wake-001",
            "run_id": self.runtime["run_id"], "worker_id": BELL_ACTOR,
            "attempt_id": self.runtime["run_id"] + ".wake-001", "action": "wake",
            "input_sha256": self.runtime.runtime_sha256,
            "action_context": {"target_actor_id": MAID_ACTOR,
                               "runtime_sha256": self.runtime.runtime_sha256}}
        intent_path = self.root / "intent.json"
        if intent_path.is_file():
            saved, _ = read_json(intent_path)
            if {k: v for k, v in saved.items() if k != "descriptor"} != self.intent:
                raise ValueError("Wake intent changed on collector restart")
            self.intent = saved
        descriptor = gate.descriptor_for_attempt(self.intent)
        if "descriptor" in self.intent and self.intent["descriptor"] != descriptor:
            raise ValueError("Wake persisted descriptor changed")
        self.intent["descriptor"] = descriptor
        immutable_json(self.root / "intent.json", self.intent)
        immutable_json(self.root / "runtime.json", dict(self.runtime.raw))

    def _terminal_operation(self, descriptor):
        """Reuse only signed terminal truth, rechecking actual evidence bytes.

        A slow first HTTP request may outlive its socket. Its durable proof is
        reusable by the identical retry/restarted collector, without serially
        repeating all authority TLS round trips inside the five-second limit.
        AWAITING or unsigned local summaries are never cached as terminal.
        """
        from .control_rpc import _verified_response

        path = self.root / "terminal" / (hashlib.sha256(
            descriptor["operation_id"].encode()).hexdigest() + ".json")
        params = {"epoch": descriptor["epoch"], "operation_id": descriptor["operation_id"]}
        if path.is_file():
            proof, _ = read_json(path)
        else:
            if self.terminal_only:
                raise ValueError("Receipt recovery requires the original signed terminal proof")
            # Reconcile uncertain old reads before making this new read.
            for pending in self.gate.journal.pending():
                if (pending["method"] == "get_operation" and pending["params"] == params
                        and pending["request"]["payload"]["actor_id"] == self.gate.session.actor_id):
                    self.gate.rpc("session", "get_operation", params, key=pending["key"])
            key = "terminal-read:" + uuid4().hex
            self.gate.rpc("session", "get_operation", params, key=key)
            proof = self.gate.journal.proof(key)
        request = proof["request"]["payload"]
        if (request["method"] != "get_operation"
                or request["params"] != {"fleet_id": self.gate.session.fleet_id, **params}):
            raise ValueError("Terminal proof differs from the requested operation")
        response = _verified_response(self.gate.session, proof["request"], proof["response"])
        if not response["ok"]:
            raise ValueError("Terminal proof is not an authority success")
        operation = response["result"]
        if operation["status"] != "RESOLVED":
            raise ValueError("Idle actor still has an unresolved authority operation")
        self.gate._verify_operation_evidence(descriptor, operation)
        immutable_json(path, proof)
        return operation

    def _drain_assignment(self):
        """Recover an issued wake, or prove this fixed wake was never issued."""
        from .control_authority import ControlError
        from .control_rpc import VerifiedRpcError

        path = self.root / "unissued.json"
        if path.is_file():
            proof, _ = read_json(path)
            self.gate.verify_unissued_intent(self.intent, proof)
            return None
        if self.terminal_only:
            saved, _ = read_json(self.root / "assignment.json")
            if (saved["runtime"] != self.runtime.raw
                    or saved["control_authority"]["descriptor"] != self.gate.descriptor_for_attempt(self.intent)):
                raise ValueError("Receipt recovery assignment differs from original journal")
            return saved
        with self.gate.journal.transaction() as db:
            row = db.execute("SELECT state,unissued_proof_json FROM control_operations WHERE operation_id=?",
                             (self.intent["operation_id"],)).fetchone()
        if row is not None and row["state"] == "cancelled_unissued":
            proof = json.loads(row["unissued_proof_json"])
            self.gate.verify_unissued_intent(self.intent, proof)
            immutable_json(path, proof)
            return None
        try:
            assignment = {"control_authority": self.gate.issue_operation(self.intent),
                          "runtime": dict(self.runtime.raw)}
        except (VerifiedRpcError, ControlError):
            # Only a fenced, fresh signed UNKNOWN_PERMIT proves absence. A
            # network error, unknown launch or local missing file never does.
            proof = self.gate.reconcile_unissued_intent(self.intent)
            immutable_json(path, proof)
            return None
        immutable_json(self.root / "assignment.json", assignment)
        return assignment

    def _no_maid_work(self):
        with self.gate.journal.transaction() as db:
            rows = db.execute("SELECT descriptor FROM control_operations").fetchall()
        if any(json.loads(row[0])["operation_id"] != self.intent["operation_id"] for row in rows):
            raise ValueError("Never-started wake conflicts with another operation")
        snapshot = self.gate._run_snapshot(self.runtime["run_id"])
        if (snapshot["attempts"] or snapshot["dispatch_intents"] or snapshot["collections"]
                or snapshot["uploads"] or any(t["status"] != "pending" for t in snapshot["tasks"])):
            raise ValueError("Never-started wake conflicts with Maid execution history")

    def _assignment(self):
        path = self.root / "assignment.json"
        if path.is_file():
            assignment, _ = read_json(path)
            return assignment
        # A lost response replays the SAME persisted operation/signature.
        with self.gate.journal.transaction() as db:
            existing = db.execute("SELECT 1 FROM control_operations WHERE operation_id=?",
                                  (self.intent["operation_id"],)).fetchone()
        if existing is None:
            self.gate.require_new_work()
        assignment = {"control_authority": self.gate.issue_operation(self.intent),
                      "runtime": dict(self.runtime.raw)}
        immutable_json(path, assignment)
        return assignment

    def _complete(self, body):
        from .control_remote_bell import MAX_WAKE_BYTES, verify_wake_evidence
        from shore_control.permits import verify_permit

        if (set(body) != {"attempt_id", "evidence", "records"}
                or body["attempt_id"] != self.intent["attempt_id"]
                or len(_json_bytes(body)) > MAX_WAKE_BYTES):
            raise ValueError("Invalid fixed wake completion")
        path = self.root / "assignment.json"
        if not path.is_file():
            raise ValueError("No dispatched wake to complete")
        assignment, _ = read_json(path)
        control = assignment["control_authority"]
        descriptor = self.gate.descriptor_for_attempt(self.intent)
        if control["descriptor"] != descriptor or assignment["runtime"] != self.runtime.raw:
            raise ValueError("Persisted wake differs from independent intent")
        resolved_path = self.root / "resolved.json"
        if resolved_path.is_file():
            saved, _ = read_json(resolved_path)
            if saved["completion"] != body:
                raise ValueError("Repeated wake completion changed its terminal evidence")
            current = self._terminal_operation(descriptor)
            return {"permit_id": current["permit"]["payload"]["permit_id"], "status": "RESOLVED"}
        permit = verify_permit(self.gate.authority_public_key, control["permit"], descriptor)
        names = ({"startup", "ready"} if body["evidence"].get("outcome") == "completed"
                 else {"tombstone"})
        if type(body["records"]) is not dict or set(body["records"]) != names:
            raise ValueError("Wake requires exact actual signed records")
        # A content-addressed evidence directory preserves rejected bytes while
        # preventing an invalid first request from poisoning later valid records.
        digest = hashlib.sha256(_json_bytes(body)).hexdigest()
        evidence_root = self.root / "evidence" / digest
        for name, record in body["records"].items():
            immutable_json(evidence_root / (name + ".json"), record)
        verify_wake_evidence(body["evidence"], permit, evidence_root,
            authority_public_key=self.gate.authority_public_key,
            bell_registration=self.gate.registrations[BELL_ACTOR],
            target_registration=self.gate.registrations[MAID_ACTOR])
        reply = self.gate.ack_evidence(control["permit"], body["evidence"], evidence_root)
        if reply["status"] == "RESOLVED":
            immutable_json(self.root / "resolved.json", {"completion": body,
                "evidence_root": str(evidence_root), "reply": reply})
        return reply

    def ready(self):
        from .control_remote_bell import verify_wake_evidence
        from shore_control.permits import verify_permit

        with self._lock:
            path = self.root / "resolved.json"
            if not path.is_file():
                return False
            saved, _ = read_json(path)
            assignment, _ = read_json(self.root / "assignment.json")
            descriptor = self.gate.descriptor_for_attempt(self.intent)
            permit = verify_permit(self.gate.authority_public_key,
                                    assignment["control_authority"]["permit"], descriptor)
            body = saved["completion"]
            evidence_hash = verify_wake_evidence(body["evidence"], permit, saved["evidence_root"],
                authority_public_key=self.gate.authority_public_key,
                bell_registration=self.gate.registrations[BELL_ACTOR],
                target_registration=self.gate.registrations[MAID_ACTOR])
            # A local summary is not an authority receipt. Ask for a freshly
            # verified operation and check BOTH receipts before new dispatch.
            current = self._terminal_operation(descriptor)
            receipt = {"outcome": body["evidence"]["outcome"], "durable": True,
                       "evidence_sha256": evidence_hash}
            self.gate._check_operation_receipts(current, assignment["control_authority"]["permit"],
                                                 descriptor, receipt)
            return current["status"] == "RESOLVED" and body["evidence"]["outcome"] == "completed"

    def handle(self, path, body, *, actor_id, node_id):
        if (actor_id not in {BELL_ACTOR, MAID_ACTOR} or node_id != LOCAL_NODE_ID
                or type(body) is not dict):
            raise ValueError("Wake actor/source mismatch")
        with self._lock:
            if self.terminal_only:
                if path != "/v1/control/idle":
                    raise ValueError("Receipt recovery accepts only existing idle records")
                saved, _ = read_json(self.root / "idle" / (actor_id + ".json"))
                if _json_bytes(saved) != _json_bytes(body):
                    raise ValueError("Receipt recovery cannot replace the original idle record")
                return self.record_idle(body, actor_id=actor_id)
            if path == "/v1/control/idle":
                return self.record_idle(body, actor_id=actor_id)
            if actor_id != BELL_ACTOR:
                raise ValueError("Only Bell may operate wake lifecycle")
            if path == "/v1/wake/complete":
                return self._complete(body)
            if path != "/v1/wake/poll" or body:
                raise ValueError("Unsupported wake request")
            if (self.root / "resolved.json").is_file():
                return {"assignment": None}
            # Collection remains possible in DRAINING; a fresh wake does not.
            if self.gate._get("local_drain", False) or self.gate.job["phase"] != "ACTIVE":
                return {"assignment": self._drain_assignment()}
            return {"assignment": self._assignment()}

    def record_idle(self, body, *, actor_id):
        """Do not infer a cleared device outbox from controller-side receipts."""
        from shore_control.crypto import verify_object

        if set(body) != {"record"} or actor_id not in {BELL_ACTOR, MAID_ACTOR}:
            raise ValueError("Expected exact registered device idle record")
        value = verify_object(self.gate.registrations[actor_id]["public_key"], IDLE_DOMAIN, body["record"])
        expected = {"schema": IDLE_DOMAIN, "schema_version": 1, "mode": MODE,
            "actor_id": actor_id, "fleet_id": self.gate.controller.fleet_id,
            "job_id": self.gate.binding["job_id"], "epoch": self.gate.binding["epoch"],
            "run_id": self.runtime["run_id"], "runtime_sha256": self.runtime.runtime_sha256,
            "outboxes_pending": 0}
        if (set(value) != set(expected) | {"permit_ids"}
                or any(_json_bytes(value[k]) != _json_bytes(v) for k, v in expected.items())
                or type(value["permit_ids"]) is not list):
            raise ValueError("Idle record differs from the frozen actor/job/runtime")
        if actor_id == MAID_ACTOR and not self.gate._get("local_drain", False):
            snapshot = self.gate._run_snapshot(self.runtime["run_id"])
            if any(row["status"] not in {"completed", "failed"} for row in snapshot["tasks"]):
                raise ValueError("Maid cannot declare final idle while new work is possible")
        with self.gate.journal.transaction() as db:
            rows = db.execute("SELECT descriptor FROM control_operations WHERE state!='cancelled_unissued'").fetchall()
        descriptors = [json.loads(row[0]) for row in rows
                       if json.loads(row[0])["worker_id"] == actor_id]
        permit_ids = []
        for descriptor in descriptors:
            operation = self._terminal_operation(descriptor)
            permit_ids.append(operation["permit"]["payload"]["permit_id"])
        if value["permit_ids"] != sorted(permit_ids) or (actor_id == BELL_ACTOR and not permit_ids):
            raise ValueError("Idle record omits or invents issued permits")
        immutable_json(self.root / "idle" / (actor_id + ".json"), body)
        return {"status": "RECORDED", "actor_id": actor_id}

    def idle_pending(self):
        actors = (BELL_ACTOR, MAID_ACTOR)
        if self.gate._get("local_drain", False):
            assignment = self._drain_assignment()
            if assignment is None:
                self._no_maid_work()
                return 0  # Permanently fenced absence, not a fictitious device idle.
            resolved = self.root / "resolved.json"
            if resolved.is_file():
                saved, _ = read_json(resolved)
                if saved["completion"]["evidence"]["outcome"] == "not_started":
                    self._terminal_operation(self.gate.descriptor_for_attempt(self.intent))
                    self._no_maid_work()
                    actors = (BELL_ACTOR,)  # Tombstone proves this Maid never existed.
        pending = 0
        for actor in actors:
            path = self.root / "idle" / (actor + ".json")
            if not path.is_file():
                pending += 1
                continue
            body, _ = read_json(path)
            self.record_idle(body, actor_id=actor)
        return pending


class WakeGatedCoordinator(PrincessCoordinator):
    """Maid readiness alone is insufficient: require resolved wake receipts."""

    def __init__(self, *args, wake_queue, **kwargs):
        self.wake_queue = wake_queue
        super().__init__(*args, **kwargs)

    def handle_message(self, request):
        request = validate_message(request)
        if (request["run_id"] != self.run_id or request.get("worker_id") != MAID_ACTOR
                or request["protocol_version"] != 2):
            raise ValueError("Explicit remote Maid/run/protocol required")
        if request["type"] == "request_task":
            if self.wake_queue.gate._get("local_drain", False):
                return self._reply(request, "stop", {"reason": "controller draining; preserve pending budget"})
            if not self.wake_queue.ready():
                return self._reply(request, "wait", {"retry_after_seconds": 2,
                                   "reason": "waiting for verified wake dual receipts"})
        return super().handle_message(request)


def open_clients(preparation, *, usb_manifest):
    """Operational only: F stays on USB, only limited session uses DPAPI."""
    from shore_control.remote import DEFAULT_TAILSCALE, load_bootstrap
    from shore_control.client import Client, MemorySigner
    from shore_control.crypto import fingerprint, public_key_text
    from shore_control.tailnet import TailnetVerifier
    from shore_control.tailnet_rpc import TailnetRpcTransport
    from .control_authority import ControlAuthority
    from .control_remote_session import load_or_create_session
    from .control_usb import LockedUsbIdentity, controller_client

    bootstrap = load_bootstrap(preparation["paths"]["bootstrap_path"])
    root = Path(preparation["output_root"])
    # Lazy public binding: receipt-only recovery must work while F is absent.
    signer = LockedUsbIdentity(actor=CONTROLLER_ACTOR,
        expected_fingerprint=fingerprint(bootstrap["controllers"][CONTROLLER_ACTOR]),
        selection=usb_manifest, binding_path=root / "controller-usb.json",
        context={"job_id": preparation["job_id"], "host_id": HOST_NODE,
                 "bootstrap_sha256": preparation["bootstrap_sha256"]},
        allow_new_binding=not any((root / name).exists() for name in
            ("session.dpapi.json", "control.sqlite", "deployment", "wake", "simulations")))
    controller = controller_client(bootstrap, actor=CONTROLLER_ACTOR,
                                   local_node_id=HOST_NODE, signer=signer)
    if signer.allow_new_binding:
        # Public-only pin BEFORE creating session/journal. A failed first
        # challenge must not strand a partial job with no recoverable USB pin.
        signer.inspect()
    if not (root / "session.dpapi.json").is_file() and any(
            (root / name).exists() for name in ("control.sqlite", "deployment", "wake", "simulations")):
        raise ValueError("Session material is missing from existing state; recovery must not create a replacement")
    if not (root / "control.sqlite").is_file() and any(
            (root / name).exists() for name in ("deployment", "wake")):
        raise ValueError("Bound control journal is missing; refusing reinitialization")
    session_id, key = load_or_create_session(root / "session.dpapi.json", context={
        "mode": MODE, "job_id": preparation["job_id"], "host_id": HOST_NODE,
        "bootstrap_sha256": preparation["bootstrap_sha256"],
        "job_spec": preparation["job_spec"]})
    verifier = TailnetVerifier(bootstrap["network"], DEFAULT_TAILSCALE)
    transport = TailnetRpcTransport(bootstrap["endpoint"], bootstrap["tls_sha256"],
                                   verifier, HOST_NODE, timeout=5.0)
    session = Client(transport, "session:" + session_id, MemorySigner(key),
                      bootstrap["authority_public_key"], bootstrap["fleet_id"])
    gate = ControlAuthority(root / "control.sqlite", controller_client=controller,
                            session_client=session, worker_registrations=bootstrap["workers"])
    return gate, verifier, session_id, public_key_text(key.public_key()), bootstrap


def prepare_project_run(preparation, bootstrap):
    """Initialize once; restart must not invoke unbound legacy initialization."""
    from .config import DeviceRegistry
    from .runtime import (PrincessRunPaths, PreparedWorklist, RunPreparation, prepare_run)
    from .state import freeze_csv

    endpoint = urlsplit(bootstrap["collector"]["endpoint"])
    registry = DeviceRegistry(bind_host=endpoint.hostname, port=endpoint.port,
                               advertise_url=bootstrap["collector"]["endpoint"], devices=())
    root, run_id = Path(preparation["output_root"]), preparation["run_id"]
    paths = PrincessRunPaths.for_run(run_id, repository_root=root)
    if not paths.database.exists():
        if (root / "deployment").exists() or (root / "wake").exists():
            raise ValueError("Bound task database is missing; refusing reinitialization")
        return prepare_run(source_csv=preparation["paths"]["samples_csv"], run_id=run_id,
                             registry=registry, devices=(), repository_root=root)
    # A partial first preparation can be recovered before binding, but no new
    # database or input is invented for a missing/corrupt bound runtime.
    run = json.loads(paths.runtime_json.read_text(encoding="utf-8"))
    source = freeze_csv(paths.source_csv, paths.source_csv)
    worklist = freeze_csv(paths.worklist_csv, paths.worklist_csv)
    if (source.sha256 != sha256_file(preparation["paths"]["samples_csv"])
            or worklist.sha256 != source.sha256 or run["run_id"] != run_id
            or run["worklist_csv_sha256"] != worklist.sha256
            or run["advertise_url"] != bootstrap["collector"]["endpoint"]):
        raise ValueError("Persisted project runtime differs from preparation")
    return RunPreparation(paths, PreparedWorklist(source, worklist, ()), run["api_token"], False)


def bind_job(preparation, gate, bootstrap, session_id, session_public_key):
    """Acquire full immutable job, register run, stage PUBLIC remote manifests."""
    root = Path(preparation["output_root"])
    prepared = prepare_project_run(preparation, bootstrap)
    gate.acquire(job_id=preparation["job_id"], host_id=HOST_NODE, session_id=session_id,
        session_public_key=session_public_key, project_id="MSABP_2",
        collector_url=bootstrap["collector"]["endpoint"], data_root=str(root),
        job_spec=preparation["job_spec"])
    gate.bind_job_documents(preparation["paths"]["job_manifest"],
                            preparation["paths"]["accounting_policy"])
    gate.register_run(preparation["run_id"], sha256_file(prepared.paths.runtime_json),
                       prepared.worklist.worklist.sha256)
    gate.bind_run_plan(preparation["run_id"], prepared.paths.runtime_json)
    remote = preparation["remote_paths"]
    runtime = {"schema": "msabp.shore-remote-synthetic-maid", "schema_version": 1,
        "mode": MODE, "run_id": preparation["run_id"], "worker_id": MAID_ACTOR,
        "local_node_id": LOCAL_NODE_ID, "bootstrap_path": remote["bootstrap_path"],
        "bootstrap_sha256": preparation["bootstrap_sha256"],
        "source_root": remote["source_root"],
        "source_hashes": {k: preparation["source_hashes"][k] for k in PROJECT_SOURCE_FILES},
        "shore_source_root": remote["shore_source_root"],
        "shore_source_hashes": {k: preparation["shore_source_hashes"][k] for k in SHORE_SOURCE_FILES},
        "worker_config_path": remote["worker_config_path"], "csv_path": remote["csv_path"],
        "csv_sha256": prepared.worklist.worklist.sha256, "api_token": prepared.api_token,
        "binding": {**gate.binding, **gate.job["scope"], "fleet_id": bootstrap["fleet_id"]},
        "state_root": remote["maid_state_root"], "output_root": remote["maid_output_root"],
        "poll_seconds": 2, "heartbeat_seconds": 5}
    if "release_profile" in preparation:
        from .control_job_material import build_runtime
        expected = build_runtime(preparation["release_profile"],
            Path(preparation["paths"]["bootstrap_path"]).read_bytes(), runtime["binding"],
            run_id=preparation["run_id"], api_token=prepared.api_token)
        if runtime != expected:
            raise ValueError("Supervisor runtime differs from the pre-signing approved release")
    config = RemoteSyntheticMaidConfig.from_mapping(runtime)
    stage = root / "deployment"
    immutable_json(stage / "maid-runtime.json", runtime)
    for name, key in ((Path(remote["bootstrap_path"]).name, "bootstrap_path"),
                      (Path(remote["csv_path"]).name, "samples_csv")):
        _persist_record(stage / name, Path(preparation["paths"][key]).read_bytes())
    immutable_json(stage / "bell-config.json", {
        "schema": "msabp.shore-remote-synthetic-bell", "schema_version": 1, "mode": MODE,
        "runtime_config_path": remote["runtime_config_path"], "runtime_sha256": config.runtime_sha256,
        "state_root": remote["bell_state_root"], "poll_seconds": 2})
    return prepared, runtime


def collector_material(path, bootstrap):
    """Check metadata; only TLS server creation later reads the collector key."""
    material, _ = read_json(path)
    if (set(material) != {"schema", "schema_version", "cert_path", "key_path", "endpoint", "tls_sha256"}
            or material["schema"] != "shore.collector-local" or material["schema_version"] != 1
            or any(material[k] != bootstrap["collector"][k] for k in ("endpoint", "tls_sha256"))):
        raise ValueError("Collector local material differs from the reviewed public bootstrap")
    for name in ("cert_path", "key_path"):
        checked_path(material[name])
    return material


def close_job(preparation, gate, *, reason):
    """Explicit operator abandonment only, after proved quiescent PAUSED_BOUND."""
    root = Path(preparation["output_root"])
    if gate.job["phase"] == "RELEASED":
        report = released_job(gate, disposition="closed")
        immutable_json(root / "explicit-close-report.json", report)
        return report
    if gate.job["phase"] != "PAUSED_BOUND" or not reason.strip():
        raise ValueError("Close requires a reconciled PAUSED_BOUND job and an explicit reason")
    runtime, _ = read_json(root / "deployment" / "maid-runtime.json")
    queue = CollectorWakeQueue(root / "wake", gate, runtime)
    pending = queue.idle_pending()
    if pending:
        raise ValueError("Close requires actual idle or proved never-started work")
    gate.build_checkpoint(worker_outboxes_pending=pending)
    report = gate.release(disposition="closed", reason=reason, explicit_close=True)
    immutable_json(root / "explicit-close-report.json", report)
    return report


def released_job(gate, *, disposition=None):
    """Reconcile the original signed release, including a lost report write."""
    with gate.journal.transaction() as db:
        rows = db.execute("SELECT key,params FROM rpc_journal WHERE method='release_job' AND identity=?",
                          (gate.journal._identity(gate.controller),)).fetchall()
    for row in rows:
        params = json.loads(row["params"])
        if (any(params.get(k) != v for k, v in gate.binding.items())
                or (disposition is not None and params.get("disposition") != disposition)):
            continue
        result = gate.rpc("controller", "release_job", params, key=row["key"])
        if result["phase"] != "RELEASED":
            raise ValueError("Original release has not been authenticated")
        return result
    raise ValueError("Released state lacks the original matching release journal")


def verified_quiescent_job(gate):
    """Authenticate a retained finish/release response without any RPC/signing."""
    from .control_rpc import _verified_response

    phase = gate.job["phase"]
    if phase not in {"PAUSED_BOUND", "RELEASED"}:
        raise ValueError("Receipt-only service requires PAUSED_BOUND or RELEASED")
    method = "release_job" if phase == "RELEASED" else "finish_drain"
    client = gate.controller if phase == "RELEASED" else gate.session
    with gate.journal.transaction() as db:
        rows = db.execute("SELECT key,params FROM rpc_journal WHERE method=? AND identity=? ORDER BY rowid DESC",
                          (method, gate.journal._identity(client))).fetchall()
    for row in rows:
        params = json.loads(row["params"])
        if any(params.get(k) != v for k, v in gate.binding.items()):
            continue
        proof = gate.journal.proof(row["key"])
        if proof is None:
            continue
        request = proof["request"]["payload"]
        if (request["method"] != method
                or request["params"] != {"fleet_id": client.fleet_id, **params}):
            raise ValueError("Retained terminal job proof differs from its request")
        response = _verified_response(client, proof["request"], proof["response"])
        if response["ok"] and response["result"] == gate.job:
            return response["result"]
    raise ValueError("Receipt-only service lacks the original signed terminal job proof")


def serve_receipts(preparation, gate, bootstrap, verifier, *, collector_config, seconds=300):
    """Reopen the pinned listener for existing idle acknowledgements only.

    No acquire, renew, authority RPC, task message, upload or wake is allowed.
    A supervisor preparation also exposes original job material/release proofs.
    Ctrl-C merely closes this bounded receipt window; it changes no job state.
    """
    from .control_remote_http import create_remote_server
    from .state import PrincessState

    if type(seconds) is not int or not 1 <= seconds <= 3600:
        raise ValueError("Receipt window must be integer seconds in [1, 3600]")
    job = verified_quiescent_job(gate)
    root = Path(preparation["output_root"])
    prepared = prepare_project_run(preparation, bootstrap)
    runtime, _ = read_json(root / "deployment" / "maid-runtime.json")
    queue = CollectorWakeQueue(root / "wake", gate, runtime, terminal_only=True)
    if queue.idle_pending():
        raise ValueError("Receipt-only service requires already retained terminal idle evidence")
    material = collector_material(collector_config, bootstrap)
    state = PrincessState(prepared.paths.database, control_authority=gate)
    server = thread = None
    try:
        coordinator = WakeGatedCoordinator(state, wake_queue=queue, run_id=preparation["run_id"],
            csv_sha256=prepared.worklist.worklist.sha256, results_dir=prepared.paths.results_root,
            incoming_dir=prepared.paths.incoming_root, lease_seconds=90, max_attempts=1)
        job_options = {}
        if "release_profile" in preparation:
            from .control_supervisor_controller import make_job_service
            job_options = {"job_service": make_job_service(preparation, gate, queue, bootstrap),
                           "allow_job_reads": True}
        server = create_remote_server(coordinator, token=prepared.api_token,
            upload_dir=prepared.paths.upload_root, bootstrap=bootstrap, verifier=verifier,
            cert_path=material["cert_path"], key_path=material["key_path"],
            wake_handler=queue.handle, receipt_only=True, **job_options)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print("[Shore synthetic] receipt-only listener active; no work or renewal permitted", flush=True)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        print("[Shore synthetic] receipt window closed; job unchanged", flush=True)
    finally:
        if server is not None:
            if thread is not None:
                server.shutdown()
                thread.join(timeout=10)
            server.server_close()
        state.close()
    return {"mode": MODE, "phase": job["phase"], "receipt_only": True}


def serve_job(preparation, gate, bootstrap, verifier, *, collector_config):
    """Keep collection alive on lost F; never auto-resume or abandon budget."""
    from .control_remote_http import create_remote_server
    from .state import PrincessState

    root = Path(preparation["output_root"])
    if gate.job["phase"] == "RELEASED":
        job = released_job(gate)
        if job["disposition"] == "closed":
            return close_job(preparation, gate, reason="Replay original explicit close")
        path = root / "completion-report.json"
        if path.is_file():
            report, _ = read_json(path)
            if report.get("phase") != "RELEASED":
                raise ValueError("Completion report disagrees with signed release")
            return report
        report = {"mode": MODE, "phase": job["phase"], "synthetic": True, "real_cst": False,
                  "checkpoint": gate._get("reconciliation"), "recovered_after_release": True}
        immutable_json(path, report)
        return report
    prepared = prepare_project_run(preparation, bootstrap)
    runtime, _ = read_json(root / "deployment" / "maid-runtime.json")
    queue = CollectorWakeQueue(root / "wake", gate, runtime)
    material = collector_material(collector_config, bootstrap)
    state = PrincessState(prepared.paths.database, control_authority=gate)
    server = thread = None
    try:
        state.bind_control_run(preparation["run_id"])
        coordinator = WakeGatedCoordinator(state, wake_queue=queue, run_id=preparation["run_id"],
            csv_sha256=prepared.worklist.worklist.sha256, results_dir=prepared.paths.results_root,
            incoming_dir=prepared.paths.incoming_root, lease_seconds=90, max_attempts=1)
        gate.bind_run_state(preparation["run_id"], prepared.paths.database)
        job_options = {}
        if "release_profile" in preparation:
            from .control_supervisor_controller import make_job_service
            job_options = {"job_service": make_job_service(preparation, gate, queue, bootstrap)}
        server = create_remote_server(coordinator, token=prepared.api_token,
            upload_dir=prepared.paths.upload_root, bootstrap=bootstrap, verifier=verifier,
            cert_path=material["cert_path"], key_path=material["key_path"], wake_handler=queue.handle,
            **job_options)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print("[Shore synthetic] collector active; awaiting independently deployed SYSTEM Bell", flush=True)
        while True:
            try:
                gate.poll_control()
                if "release_profile" in preparation:
                    from .control_supervisor_controller import publish_original_wake
                    publish_original_wake(gate, queue)
            except Exception as exc:
                print("[Shore synthetic] new dispatch stopped; collection retained: " + type(exc).__name__, flush=True)
            if "release_profile" in preparation:
                # An idle supervisor waits for a knock rather than polling. Repeat
                # it every pass: a datagram may be lost or the worker restarted.
                from .control_supervisor_controller import knock
                knock(bootstrap)
            progress = state.progress(preparation["run_id"])
            print("[Shore synthetic] " + json.dumps(progress.as_dict()), flush=True)
            if progress.is_terminal and gate.job["phase"] == "ACTIVE":
                gate.begin_drain("All synthetic evaluations reached terminal state")
            if gate.job["phase"] in {"DRAINING", "PAUSED_BOUND"}:
                try:
                    if gate.job["phase"] != "PAUSED_BOUND":
                        pending = queue.idle_pending()
                        if pending:
                            raise ValueError("Awaiting actual Bell/Maid signed idle records")
                        gate.finish_drain(worker_outboxes_pending=pending)
                    if progress.completed == preparation["points"]:
                        gate.release(disposition="completed", reason="Authorized synthetic acceptance completed")
                    # With missing F, release fails and collection stays alive.
                    # An incomplete drained job remains bound for explicit review.
                    report = {"mode": MODE, "phase": gate.job["phase"],
                        "progress": progress.as_dict(), "synthetic": True, "real_cst": False,
                        "checkpoint": gate._get("reconciliation")}
                    immutable_json(root / "completion-report.json", report)
                    time.sleep(3)  # Short terminal grace for the Maid's final poll.
                    return report
                except Exception as exc:
                    print("[Shore synthetic] drain/release pending: " + type(exc).__name__, flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        # Ctrl-C is an explicit stop of this collector, never a force-close.
        try:
            gate.begin_drain("Operator interrupted collector; retain journals and resume collection")
        finally:
            print("[Shore synthetic] stopped; job NOT released; use same preparation to collect", flush=True)
        raise
    finally:
        if server is not None:
            if thread is not None:
                server.shutdown()
                thread.join(timeout=10)
            server.server_close()
        state.close()
