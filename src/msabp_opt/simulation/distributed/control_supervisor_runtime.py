"""Fixed F-only supervisor lifecycle; no controller identity or permit issuance.

Only the installer initializes the global ledger. Each newly discovered job
gets its own immutable material journal; a previous job is never removed or
overwritten. Missing ledgers, uncertain process identity, and STARTING without
signed ready remain blocking. This module does not install or activate code.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import sqlite3
import time
from uuid import uuid4

from .control_job_material import _binding
from .control_release import runtime_paths, safe_job_id, validate_profile, verify_release
from .control_remote_worker import BELL_ACTOR, LOCAL_NODE_ID, MAID_ACTOR, _json_bytes, checked_path
from .control_supervisor import (
    STATE_SCHEMA, STATE_VERSION, MaintenanceRequired, MaterialJournal, _maintenance_database, _regular,
    _write_new, initialize_state,
)

MAINTENANCE_SCHEMA = "msabp.supervisor-maintenance-snapshot"
MAINTENANCE_VERSION = 1

# Pacing of the supervisor loop. A step re-verifies the whole installation and
# asks the authority for a fresh signed status, so it is not free. While a job
# is in progress the loop steps every STEP_SECONDS. With nothing to do it waits
# for a knock, and steps on its own only every IDLE_SECONDS as a fallback for a
# lost knock or a changed local policy.
KNOCK_PORT = 18768
STEP_SECONDS = 2.0
IDLE_SECONDS = 300.0
BLOCKED_SECONDS = 30.0
IDLE_STATES = frozenset({"IDLE", "MAINTENANCE"})


def pause_seconds(state):
    """How long the loop may rest after a step that returned this state."""
    if state in IDLE_STATES:
        return IDLE_SECONDS
    if state == "BLOCKED":
        return BLOCKED_SECONDS
    return STEP_SECONDS


class KnockListener:
    """A datagram on this port means "look now". It is a hint and nothing more.

    The datagram is never parsed and carries no authority: a woken supervisor
    does exactly what it would have done at its next scheduled step, including
    asking the authority for a fresh signed status. The loop keeps a minimum
    gap between steps, so knocking cannot make it work harder than plain
    STEP_SECONDS polling would. If the port cannot be bound the supervisor still
    works, on the fallback timer alone.
    """

    def __init__(self, address, port=KNOCK_PORT):
        self.socket = None
        try:
            listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                listener.bind((address, port))
                listener.setblocking(False)
            except BaseException:
                listener.close()
                raise
            self.socket = listener
        except (OSError, TypeError, ValueError, OverflowError):
            self.socket = None

    @property
    def listening(self):
        return self.socket is not None

    def drain(self):
        """Discard knocks that arrived before the step about to run."""
        while self.socket is not None:
            try:
                self.socket.recvfrom(64)
            except OSError:
                return

    def wait(self, seconds):
        """Rest up to ``seconds``; return True if a knock ended the rest early."""
        if seconds <= 0:
            return False
        if self.socket is None:
            time.sleep(seconds)
            return False
        try:
            readable, _, _ = select.select([self.socket], [], [], seconds)
        except (OSError, ValueError):
            time.sleep(seconds)
            return False
        return bool(readable)

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None


def maintenance_snapshot(verified_installation):
    """Reverify this supervisor's local history within one maintenance transaction.

    The caller must hold ``host.lock -> runtime.lock -> maintenance.lock``
    (offset 1, length 1), have observed the host exit, and have just completed
    strict ``verify_installation()`` in that same lock interval. This function
    does not acquire locks, contact a service, open a signing identity, launch
    work, initialize a ledger, or repair state. Every database is opened read
    only and every historical job is checked again on every call.

    The result is a transaction-local observation, NOT a reusable authorization
    ticket or a full-pool idle proof. Zero checked jobs/processes proves only
    that this supervisor's new ledger contains no records; it says nothing
    about legacy per-job entry points (including 003/004). The installer must
    establish their separate migration/exit evidence before activation.
    """
    try:
        runtime = SupervisorRuntime(verified_installation, maintenance_readonly=True)
        marker = runtime.ledger.installation()
        counts = runtime._quiescent(readonly=True)
        return {"schema": MAINTENANCE_SCHEMA, "schema_version": MAINTENANCE_VERSION,
                "installation_id": marker["installation_id"],
                "release_id": runtime.profile["release_id"], **counts, "quiescent": True}
    except MaintenanceRequired:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, sqlite3.Error) as exc:
        raise MaintenanceRequired("Maintenance snapshot requires intact original local history") from exc


def initialize_runtime_state(root, config_parent, profile, *, installation_id=None):
    """Explicit first-install only; never used by startup or recovery.

    Partial installation is deliberately preserved for administrative review.
    The installer owns ACL/owner, fixed-path and initial-idle validation.
    """
    root, config = checked_path(root), checked_path(config_parent)
    profile = validate_profile(profile)
    if not config.is_dir() or root == config or root in config.parents or config in root.parents:
        raise ValueError("Existing disjoint trusted configuration directory required")
    marker = {"schema": STATE_SCHEMA, "schema_version": STATE_VERSION,
              "installation_id": installation_id or uuid4().hex,
              "config_parent": str(config), "profile": profile}
    if type(marker["installation_id"]) is not str or not marker["installation_id"]:
        raise ValueError("Installation identity required")
    root.mkdir(exist_ok=False)
    (root / "jobs").mkdir()
    for name in ("host.lock", "maintenance.lock", "runtime.lock"):
        _write_new(root / name, b"0")
    with closing(sqlite3.connect(root / "runtime.sqlite")) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE install (id INTEGER PRIMARY KEY CHECK(id=1), record TEXT NOT NULL)")
        db.execute("INSERT INTO install VALUES(1,?)", (_json_bytes(marker).decode(),))
        db.execute("""CREATE TABLE jobs (job_id TEXT PRIMARY KEY COLLATE NOCASE,
            binding TEXT NOT NULL, snapshot TEXT NOT NULL, profile TEXT NOT NULL,
            stage TEXT NOT NULL, offer TEXT, release_proof TEXT, terminal TEXT)""")
        db.execute("""CREATE TABLE processes (process_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL, purpose TEXT NOT NULL, state TEXT NOT NULL,
            identity TEXT, exit_record TEXT)""")
        db.commit()
    _write_new(root / "runtime-installed.json", _json_bytes(marker))


class RuntimeLedger:
    def __init__(self, root, *, profile, maintenance_readonly=False):
        self.root, self.profile = checked_path(root), validate_profile(profile)
        self.maintenance_readonly = maintenance_readonly
        self.installation()

    @contextmanager
    def db(self, *, readonly=False):
        path = _regular(self.root / "runtime.sqlite")
        if self.maintenance_readonly:
            if not readonly:
                raise MaintenanceRequired("Maintenance cannot open a writable runtime ledger")
            with _maintenance_database(path) as db:
                db.row_factory = sqlite3.Row
                yield db
            return
        db = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"), uri=True)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            if not readonly:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if not readonly:
                db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def installation(self):
        try:
            raw = _regular(self.root / "runtime-installed.json").read_bytes()
            marker = json.loads(raw)
            if (set(marker) != {"schema", "schema_version", "installation_id", "config_parent", "profile"}
                    or marker["schema"] != STATE_SCHEMA or type(marker["schema_version"]) is not int
                    or marker["schema_version"] != STATE_VERSION
                    or _json_bytes(marker) != raw):
                raise MaintenanceRequired("Original runtime installation/profile changed")
            initial = validate_profile(marker["profile"])
            if initial["bootstrap_sha256"] != self.profile["bootstrap_sha256"]:
                raise MaintenanceRequired("Installation bootstrap cannot change")
            with self.db(readonly=True) as db:
                rows = db.execute("SELECT id,record FROM install").fetchall()
                if [tuple(row) for row in rows] != [(1, raw.decode())]:
                    raise MaintenanceRequired("Runtime installation ledger differs from marker")
                jobs = db.execute("SELECT job_id,binding,snapshot,profile,stage,offer,release_proof,terminal FROM jobs").fetchall()
                processes = db.execute("SELECT process_id,job_id,purpose,state,identity,exit_record FROM processes").fetchall()
                if any(row["stage"] != "RELEASED" and json.loads(row["profile"]) != self.profile for row in jobs):
                    raise MaintenanceRequired("Old active job requires its original release")
                job_profiles = {row["job_id"]: json.loads(row["profile"]) for row in jobs}
                if any(row["job_id"] not in job_profiles for row in processes):
                    raise MaintenanceRequired("Process refers to a missing original job")
                if any(row["state"] != "EXITED" and job_profiles[row["job_id"]] != self.profile
                       for row in processes):
                    raise MaintenanceRequired("Release change requires every old subtree exited")
            for name in ("host.lock", "maintenance.lock", "runtime.lock"):
                lock_path = _regular(self.root / name)
                # Exact unbuffered one-byte read avoids a speculative read
                # crossing the independently locked byte just beyond EOF.
                with lock_path.open("rb", buffering=0) as sentinel:
                    if lock_path.stat().st_size != 1 or sentinel.read(1) != b"0":
                        raise MaintenanceRequired("Original runtime lock changed")
            if not checked_path(self.root / "jobs").is_dir():
                raise MaintenanceRequired("Missing original per-job journal directory")
            if not checked_path(marker["config_parent"]).is_dir():
                raise MaintenanceRequired("Missing original configuration parent")
            return marker
        except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
            raise MaintenanceRequired("Runtime installation/ledger missing or damaged") from exc

    @contextmanager
    def exclusive(self):
        self.installation()
        with _regular(self.root / "runtime.lock").open("r+b") as stream:
            # Lock the byte after the immutable sentinel, not its contents.
            # Windows supports byte-range locks beyond EOF. This preserves
            # strict b'0' integrity verification while the lock is owned.
            stream.seek(1)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise MaintenanceRequired("Another runtime owns the lifecycle lock") from exc
            try:
                yield
            finally:
                stream.seek(1)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def active(self):
        with self.db(readonly=True) as db:
            rows = [dict(row) for row in db.execute("SELECT * FROM jobs WHERE stage!='RELEASED'")]
        if len(rows) > 1:
            raise MaintenanceRequired("More than one unreconciled original job")
        return rows[0] if rows else None

    def job(self, job_id):
        with self.db(readonly=True) as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (safe_job_id(job_id),)).fetchone()
        if row is None:
            raise MaintenanceRequired("Missing original job")
        return dict(row)

    def advance(self, job_id, stage, *, field=None, value=None):
        if stage not in {"DOWNLOADING", "COMMITTED", "SESSION_INIT", "SESSION", "RELEASED"}:
            raise ValueError("Unknown runtime stage")
        if field not in {None, "offer", "release_proof", "terminal"}:
            raise ValueError("Invalid lifecycle record")
        with self.db() as db:
            if field:
                db.execute(f"UPDATE jobs SET stage=?,{field}=? WHERE job_id=?",
                           (stage, _json_bytes(value).decode(), job_id))
            else:
                db.execute("UPDATE jobs SET stage=? WHERE job_id=?", (stage, job_id))

    def processes(self, job_id):
        with self.db(readonly=True) as db:
            return [dict(row) for row in db.execute("SELECT * FROM processes WHERE job_id=? ORDER BY rowid", (job_id,))]


def discovery_binding(status, bootstrap):
    """Filter a FRESH verified Bell status; never accept an unsigned hint."""
    if type(status) is not dict or status.get("fleet_id") != bootstrap["fleet_id"]:
        raise ValueError("Fresh authority status differs from local fleet")
    job = status.get("bound_job")
    if job is None:
        return None
    if type(job) is not dict or job.get("phase") != "ACTIVE":
        return None
    scope = job.get("scope")
    if type(scope) is not dict:
        raise ValueError("Missing original job scope")
    value = {"fleet_id": status["fleet_id"], **{k: job[k] for k in
             ("job_id", "epoch", "host_id", "session_id")}, **scope}
    return _binding(value, bootstrap)


class ManagedMaidLauncher:
    """Own original handles and persist every launch intent before spawning."""
    def __init__(self, installation, ledger, job_id, *, job_factory=None):
        from .control_winjob import ManagedJob
        self.installation, self.ledger, self.job_id = installation, ledger, job_id
        self.job_factory = job_factory or ManagedJob
        self.live = {}

    def spawn(self, args, *, purpose):
        process_id = uuid4().hex
        with self.ledger.db() as db:
            db.execute("INSERT INTO processes VALUES(?,?,?,'INTENT',NULL,NULL)",
                       (process_id, self.job_id, purpose))
        paths = self.installation["paths"]
        argv = [str(paths["python"]), "-I", "-B", "-u",
                str(paths["app"] / "scripts/simulation/shore_supervisor.py"),
                "--policy", str(paths["policy"]), *args]
        job = self.job_factory()
        try:
            process = job.spawn(argv, cwd=paths["app"])
        except BaseException:
            job.close()
            raise
        self.live[process_id] = (job, process)
        with self.ledger.db() as db:
            db.execute("UPDATE processes SET state='RUNNING',identity=? WHERE process_id=?",
                       (_json_bytes({"pid": process.pid, "creation_time": process.creation_time}).decode(), process_id))
        return process

    def __call__(self, runtime_path, startup_path, ready_path):
        # The Bell outbox runtime is byte-identical to the committed runtime.
        original = runtime_paths(self.installation["profile"], self.job_id)
        committed = _regular(Path(original["runtime_config_path"]))
        if _regular(Path(runtime_path)).read_bytes() != committed.read_bytes():
            raise MaintenanceRequired("Bell launch runtime differs from original committed bytes")
        return self.spawn(["--maid-runtime", str(committed), "--wake-startup", str(startup_path),
                           "--ready-record", str(ready_path), "--maid-run"], purpose="execute")

    def observe(self):
        for identifier, (job, process) in list(self.live.items()):
            if process.poll() is None or job.active_processes() != 0:
                continue
            record = process.exit_record.to_dict()
            if record["pid"] != process.pid or record["creation_time"] != process.creation_time:
                raise MaintenanceRequired("Original process identity changed")
            with self.ledger.db() as db:
                db.execute("UPDATE processes SET state='EXITED',exit_record=? WHERE process_id=?",
                           (_json_bytes(record).decode(), identifier))
            # Tree is already observed empty. close() is not used as a drain.
            job.close()
            del self.live[identifier]
        records = self.ledger.processes(self.job_id)
        if any(row["state"] != "EXITED" and row["process_id"] not in self.live for row in records):
            return "UNKNOWN"
        return "RUNNING" if self.live else "EXITED"


class OriginalJobSession:
    """Bridge one committed original wake into the existing strict Bell runner."""
    def __init__(self, installation, ledger, material, client, verifier, *, create):
        from .control_remote_bell import BellRunner, WakeCoordinator, preflight_bell
        from .control_remote_http import RemotePrincessClient
        self.installation, self.ledger, self.material = installation, ledger, material
        self.paths = runtime_paths(installation["profile"], material.job_id)
        config, runtime, bootstrap = preflight_bell(self.paths["bell_config_path"])
        self.runtime, self.bootstrap = runtime, bootstrap
        root = checked_path(config["state_root"])
        if create:
            if root.exists() or runtime.state_root.exists() or Path(runtime["output_root"]).exists():
                raise MaintenanceRequired("Unjournaled original runner directories cannot be adopted")
        else:
            for name in ("wake.sqlite", "rpc.sqlite"):
                _regular(root / name)
            # Existing but empty/replaced journals must not be initialized by
            # the legacy coordinator's CREATE TABLE IF NOT EXISTS convenience.
            with closing(sqlite3.connect((root / "wake.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
                rows = db.execute("SELECT permit_id FROM wake").fetchall()
                if rows != [(material.record["permit_id"],)]:
                    raise MaintenanceRequired("Missing or unexpected original wake inbox")
                if db.execute("SELECT COUNT(*) FROM pins").fetchone()[0] != 1:
                    raise MaintenanceRequired("Missing original Bell pins")
            with closing(sqlite3.connect((root / "rpc.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
                db.execute("SELECT key,request FROM rpc_journal").fetchall()
        self.coordinator = WakeCoordinator(root, client, bootstrap["authority_public_key"],
            binding=runtime["binding"], registration=bootstrap["workers"][BELL_ACTOR],
            target_actor_id=MAID_ACTOR, target_registration=bootstrap["workers"][MAID_ACTOR],
            runtime=runtime.raw, run_id=runtime["run_id"])
        self.collector = RemotePrincessClient(bootstrap, BELL_ACTOR, client.signer, verifier,
                                             run_id=runtime["run_id"])
        self.launcher = ManagedMaidLauncher(installation, ledger, material.job_id)
        self.runner = BellRunner(self.coordinator, self.collector, self.launcher, ready_root=runtime.state_root)
        assignment = {"runtime": json.loads(material.files["maid-runtime.json"]),
                      "control_authority": material.record["control_authority"]}
        self.permit = self.coordinator.prepare(assignment)

    def _idle(self):
        from shore_control.crypto import verify_object
        from .control_remote_bell import IDLE_DOMAIN
        from .control_remote_worker import _idle_payload, _read_idle_permits, read_json
        c = self.coordinator
        if c.pending() or c.rpc.pending():
            return None
        c.checked_result(self.permit["attempt_id"])
        idle_path = self.runtime.state_root / "idle.json"
        if not idle_path.exists():
            return None
        envelope, _ = read_json(idle_path)
        maid = verify_object(self.bootstrap["workers"][MAID_ACTOR]["public_key"], IDLE_DOMAIN, envelope)
        if _json_bytes(maid) != _json_bytes(_idle_payload(self.runtime, _read_idle_permits(self.runtime))):
            raise MaintenanceRequired("Maid idle differs from original runtime and durable journals")
        payload = {"schema": IDLE_DOMAIN, "schema_version": 1,
            "mode": self.runtime["mode"], "actor_id": BELL_ACTOR, "fleet_id": c.client.fleet_id,
            "job_id": c.binding["job_id"], "epoch": c.binding["epoch"], "run_id": c.run_id,
            "runtime_sha256": c.runtime_sha256, "permit_ids": [self.permit["permit_id"]], "outboxes_pending": 0}
        bell_path = c.root / "idle.json"
        if not bell_path.exists():
            _write_new(bell_path, _json_bytes(c.client.signer.sign(IDLE_DOMAIN, payload)))
        bell_envelope, _ = read_json(bell_path)
        bell = verify_object(self.bootstrap["workers"][BELL_ACTOR]["public_key"], IDLE_DOMAIN, bell_envelope)
        if _json_bytes(bell) != _json_bytes(payload):
            raise MaintenanceRequired("Bell idle differs from original wake journal")
        reply = self.collector.request_json("/v1/control/idle", {"record": bell_envelope})
        if reply != {"status": "RECORDED", "actor_id": BELL_ACTOR}:
            raise ValueError("Missing exact Bell idle acknowledgement")
        return {"bell_idle": bell_envelope, "maid_idle": envelope,
                "outboxes_pending": 0, "unknown": False}

    def step(self):
        tree = self.launcher.observe()
        if tree == "UNKNOWN":
            return {"state": "UNKNOWN", "terminal": None}
        if self.coordinator.pending():
            result = self.runner.step()  # Only original inbox; no wake poll.
            return {**result, "terminal": None}
        if tree == "RUNNING":
            return {"state": "RUNNING", "terminal": None}
        outcome = self.coordinator.checked_result(self.permit["attempt_id"])["receipt"]["outcome"]
        if outcome == "not_started":
            # There cannot be a real Maid idle signature when it never ran.
            # Preserve this correct not_started result without inventing one.
            return {"state": "MAINTENANCE_NOT_STARTED", "terminal": None}
        processes = self.ledger.processes(self.material.job_id)
        if not processes:
            raise MaintenanceRequired("Started wake has no original process exit observation")
        latest = processes[-1]
        if (self.runtime.state_root / "idle.json").exists() and json.loads(latest["exit_record"])["exit_code"] != 0:
            self.launcher.spawn(["--maid-runtime", self.paths["runtime_config_path"], "--report-idle"],
                                purpose="report-idle")
            return {"state": "IDLE_RECOVERY", "terminal": None}
        idle = self._idle()
        if idle is not None:
            return {"state": "IDLE", "terminal": {**idle, "subtree_exited": True,
                    "processes": self.ledger.processes(self.material.job_id)}}
        # A dead child is not permission to execute. Only the original terminal
        # recovery entrypoint may run, after validating all original journals.
        from .control_remote_worker import _terminal_records, _verify_recovery_launch
        _verify_recovery_launch(self.runtime, self.bootstrap)
        _terminal_records(self.runtime)
        self.launcher.spawn(["--maid-runtime", self.paths["runtime_config_path"], "--recover-terminal"],
                            purpose="recover-terminal")
        return {"state": "TERMINAL_RECOVERY", "terminal": None}


class SupervisorRuntime:
    def __init__(self, installation, *, bell_client=None, verifier=None,
                 material_client_factory=None, session_factory=None, refresh_installation=None,
                 maintenance_readonly=False):
        self.installation = installation
        self.profile = validate_profile(installation["profile"])
        self.bootstrap_data, self.bootstrap = installation["bootstrap_data"], installation["bootstrap"]
        if hashlib.sha256(self.bootstrap_data).hexdigest() != self.profile["bootstrap_sha256"]:
            raise MaintenanceRequired("Local bootstrap bytes changed")
        self.ledger = RuntimeLedger(installation["paths"]["state"], profile=self.profile,
                                    maintenance_readonly=maintenance_readonly)
        marker = self.ledger.installation()
        if checked_path(marker["config_parent"]) != checked_path(installation["paths"]["config"]):
            raise MaintenanceRequired("Original configuration parent differs from local policy")
        self.client, self.verifier = bell_client, verifier
        self.material_client_factory, self.session_factory = material_client_factory, session_factory
        self.refresh_installation = refresh_installation
        self.sessions = {}

    def _clients(self):
        if self.client is not None:
            if self.client.actor_id != BELL_ACTOR or self.client.fleet_id != self.bootstrap["fleet_id"]:
                raise MaintenanceRequired("Runtime requires the fixed Bell identity")
            return
        from shore_control.remote import DEFAULT_TAILSCALE, client_from_bootstrap
        from shore_control.tailnet import TailnetVerifier
        network = {**self.bootstrap["network"], "local_node_id": LOCAL_NODE_ID,
                   "local_address": self.bootstrap["network"]["nodes"][LOCAL_NODE_ID]["address"]}
        self.verifier = TailnetVerifier(network, DEFAULT_TAILSCALE)
        self.verifier.verify_local()
        self.client = client_from_bootstrap(self.bootstrap, BELL_ACTOR, local_node_id=LOCAL_NODE_ID,
                                            worker_config=str(self.installation["paths"]["worker_config"]))

    def _material_client(self, binding):
        from .control_job_http import JobMaterialClient
        return (self.material_client_factory or JobMaterialClient)(profile=self.profile,
            bootstrap_data=self.bootstrap_data, expected_binding=binding,
            signer=self.client.signer, verifier=self.verifier, timeout=5)

    def _journal(self, job_id):
        return MaterialJournal(self.ledger.root / "jobs" / safe_job_id(job_id) / "material",
                               profile=self.profile, bootstrap_data=self.bootstrap_data)

    def _verify_retired(self, row, *, readonly=False):
        """Recheck original bytes and signed terminal evidence after restart."""
        from shore_control.crypto import verify_object
        from .control_job_http import verify_release_proof
        from .control_remote_bell import IDLE_DOMAIN, verify_wake_evidence
        from .control_remote_worker import RemoteSyntheticMaidConfig, _idle_payload, _read_idle_permits
        job_id, profile = row["job_id"], validate_profile(json.loads(row["profile"]))
        original_paths = runtime_paths(profile, job_id)
        verify_release(profile, original_paths["source_root"], original_paths["shore_source_root"])
        binding, terminal = json.loads(row["binding"]), json.loads(row["terminal"])
        verify_release_proof(json.loads(row["release_proof"]), self.bootstrap, binding)
        journal = MaterialJournal(self.ledger.root / "jobs" / job_id / "material",
                                  profile=profile, bootstrap_data=self.bootstrap_data,
                                  maintenance_readonly=readonly)
        material = journal.original_readonly(job_id) if readonly else journal.original(job_id)
        config = RemoteSyntheticMaidConfig.from_mapping(json.loads(material.files["maid-runtime.json"]))
        if (terminal.get("unknown") is not False or terminal.get("outboxes_pending") != 0
                or terminal.get("subtree_exited") is not True):
            raise MaintenanceRequired("Original terminal summary is incomplete")
        maid = verify_object(self.bootstrap["workers"][MAID_ACTOR]["public_key"],
                             IDLE_DOMAIN, terminal["maid_idle"])
        # The worker helper uses mode=ro but only checks is_file(). Retain the
        # supervisor's non-aliased regular-file requirement transitively.
        for name in ("worker.sqlite", "rpc.sqlite"):
            _regular(config.state_root / "control" / name)
        opener = (_maintenance_database if readonly else
                  lambda path: closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)))
        permits = _read_idle_permits(config, database_opener=opener) if readonly else _read_idle_permits(config)
        if _json_bytes(maid) != _json_bytes(_idle_payload(config, permits)):
            raise MaintenanceRequired("Original Maid terminal journal or idle changed")
        bell = verify_object(self.bootstrap["workers"][BELL_ACTOR]["public_key"],
                             IDLE_DOMAIN, terminal["bell_idle"])
        expected = {**_idle_payload(config, [material.record["permit_id"]]), "actor_id": BELL_ACTOR}
        if _json_bytes(bell) != _json_bytes(expected):
            raise MaintenanceRequired("Original Bell idle changed")
        root = checked_path(original_paths["bell_state_root"])
        with opener(_regular(root / "wake.sqlite")) as db:
            rows = db.execute("SELECT permit_id,state,resolved,assignment,result FROM wake").fetchall()
            if (len(rows) != 1 or rows[0][0] != material.record["permit_id"]
                    or rows[0][1] != "RESOLVED" or rows[0][2] is None or rows[0][4] is None):
                raise MaintenanceRequired("Original Bell inbox is not fully resolved")
            from shore_control.permits import verify_permit
            gate = json.loads(rows[0][3])["control_authority"]
            permit = verify_permit(self.bootstrap["authority_public_key"], gate["permit"], gate["descriptor"])
            if permit["permit_id"] != material.record["permit_id"] or any(permit[k] != binding[k] for k in binding):
                raise MaintenanceRequired("Original resolved wake changed its job binding")
            result = json.loads(rows[0][4])
            if checked_path(result["root"]) != root / "outbox" / permit["permit_id"]:
                raise MaintenanceRequired("Original wake outbox moved")
            digest = verify_wake_evidence(result["evidence"], permit, result["root"],
                authority_public_key=self.bootstrap["authority_public_key"],
                bell_registration=self.bootstrap["workers"][BELL_ACTOR],
                target_registration=self.bootstrap["workers"][MAID_ACTOR])
            if digest != result["receipt"]["evidence_sha256"]:
                raise MaintenanceRequired("Original signed readiness/outbox bytes changed")
        with opener(_regular(root / "rpc.sqlite")) as db:
            if db.execute("SELECT COUNT(*) FROM rpc_journal WHERE result IS NULL AND error IS NULL").fetchone()[0]:
                raise MaintenanceRequired("Original Bell RPC remains pending")
        processes = self.ledger.processes(job_id)
        if processes != terminal["processes"] or not processes:
            raise MaintenanceRequired("Original process exit history changed")

    def _quiescent(self, *, readonly=False):
        with self.ledger.db(readonly=True) as db:
            rows = [dict(row) for row in db.execute("SELECT * FROM jobs")]
            processes = [dict(row) for row in db.execute("SELECT * FROM processes")]
        for row in processes:
            if row["job_id"] not in {job["job_id"] for job in rows}:
                raise MaintenanceRequired("Process refers to a missing original job")
            if row["state"] != "EXITED" or not row["exit_record"] or not row["identity"]:
                raise MaintenanceRequired("An old process has no original confirmed exit")
            identity, exit_record = json.loads(row["identity"]), json.loads(row["exit_record"])
            if (set(identity) != {"pid", "creation_time"}
                    or set(exit_record) != {"pid", "creation_time", "exit_time", "exit_code"}
                    or any(type(exit_record[key]) is not int for key in exit_record)
                    or any(type(identity[key]) is not int for key in identity)
                    or any(identity[key] != exit_record[key] for key in identity)
                    or exit_record["pid"] <= 0 or exit_record["creation_time"] <= 0
                    or exit_record["exit_time"] < exit_record["creation_time"]):
                raise MaintenanceRequired("Original process exit identity is inconsistent")
        for row in rows:
            if row["stage"] != "RELEASED" or row["release_proof"] is None or row["terminal"] is None:
                raise MaintenanceRequired("Original job lacks durable release and idle proof")
            if readonly:
                self._verify_retired(row, readonly=True)
            else:
                self._verify_retired(row)
        return {"checked_jobs": len(rows), "checked_processes": len(processes)}

    def _discover(self):
        # Client.call performs a new challenge/signature verification each time.
        # Never route discovery through SignedRpcJournal's replay cache.
        self._quiescent()
        status = self.client.call("status")
        binding = discovery_binding(status, self.bootstrap)
        if binding is None:
            return {"state": "IDLE"}
        job_id = binding["job_id"]
        directory = checked_path(self.ledger.root / "jobs" / job_id)
        with self.ledger.db() as db:
            if db.execute("SELECT 1 FROM processes WHERE state!='EXITED' LIMIT 1").fetchone():
                raise MaintenanceRequired("An old process has no original confirmed exit")
            if db.execute("SELECT 1 FROM jobs WHERE stage='RELEASED' AND (release_proof IS NULL OR terminal IS NULL)").fetchone():
                raise MaintenanceRequired("Released job lacks original terminal proof")
            if db.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone() or directory.exists():
                raise MaintenanceRequired("Original job ID cannot be reused or silently adopted")
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,'RESERVED',NULL,NULL,NULL)",
                (job_id, _json_bytes(binding).decode(), _json_bytes(status).decode(), _json_bytes(self.profile).decode()))
        directory.mkdir()
        # Explicit one-time job creation follows a fresh signed bound status.
        # A crash before DOWNLOADING remains RESERVED and requires maintenance.
        initialize_state(directory / "material", self.installation["paths"]["config"], self.profile)
        self.ledger.advance(job_id, "DOWNLOADING")
        return {"state": "DISCOVERED", "job_id": job_id}

    def _old_job(self, row):
        job_id, stage = row["job_id"], row["stage"]
        if json.loads(row["profile"]) != self.profile:
            raise MaintenanceRequired("Old job must retain its original approved release")
        if stage in {"RESERVED", "SESSION_INIT"}:
            raise MaintenanceRequired("Interrupted first creation requires maintenance, not reinitialization")
        if stage not in {"DOWNLOADING", "COMMITTED", "SESSION"}:
            raise MaintenanceRequired("Unknown original job lifecycle stage")
        binding = json.loads(row["binding"])
        material_client = self._material_client(binding)
        journal = self._journal(job_id)
        if stage == "DOWNLOADING":
            offer = material_client.offer()
            # Pin the original offer excluding a later terminal proof.
            stable = {key: value for key, value in offer.items() if key != "terminal_proof"}
            if row["offer"] is not None and json.loads(row["offer"]) != stable:
                raise MaintenanceRequired("Original job offer changed")
            self.ledger.advance(job_id, stage, field="offer", value=stable)
            journal.download(job_id, material_client.material(offer), expected_binding=binding)
            journal.verify(job_id)
            journal.commit(job_id)
            self.ledger.advance(job_id, "COMMITTED")
            return {"state": "COMMITTED", "job_id": job_id}
        material = journal.original(job_id)
        if job_id not in self.sessions:
            create = stage == "COMMITTED"
            if create:
                self.ledger.advance(job_id, "SESSION_INIT")
            factory = self.session_factory or OriginalJobSession
            self.sessions[job_id] = factory(self.installation, self.ledger, material,
                                            self.client, self.verifier, create=create)
            if create:
                self.ledger.advance(job_id, "SESSION")
        result = self.sessions[job_id].step()
        terminal = result.get("terminal")
        if terminal is None:
            return {"state": result["state"], "job_id": job_id}
        if (terminal.get("outboxes_pending") != 0 or terminal.get("unknown") is not False
                or terminal.get("subtree_exited") is not True
                or not terminal.get("bell_idle") or not terminal.get("maid_idle")):
            raise MaintenanceRequired("Local terminal evidence is incomplete")
        # Empty status never clears a job. Require the original signed F request
        # and authority release response transported by the job-scoped endpoint.
        from .control_job_http import verify_release_proof
        offer = material_client.offer()
        proof = offer.get("terminal_proof")
        if proof is None:
            return {"state": "AWAITING_RELEASE", "job_id": job_id}
        verify_release_proof(proof, bootstrap=self.bootstrap, expected_binding=binding)
        self.ledger.advance(job_id, "SESSION", field="release_proof", value=proof)
        self.ledger.advance(job_id, "RELEASED", field="terminal", value=terminal)
        del self.sessions[job_id]
        return {"state": "RELEASED", "job_id": job_id}

    def step(self):
        """One bounded transition; maintenance still reconciles the old job."""
        with self.ledger.exclusive():
            if self.refresh_installation is not None:
                current = self.refresh_installation()
                if current["profile"] != self.profile:
                    raise MaintenanceRequired("Active release changed while runtime was open")
                self.installation["policy"] = current["policy"]
            old = self.ledger.active()
            if old is None and self.installation["policy"].get("discovery_enabled") is not True:
                self._quiescent()
                if self.installation["policy"].get("stop_when_idle") is True:
                    return {"state": "STOPPED", "discovery_enabled": False}
                return {"state": "MAINTENANCE", "discovery_enabled": False}
            self._clients()
            return self._old_job(old) if old is not None else self._discover()

    def _knock_listener(self):
        try:
            address = self.bootstrap["network"]["nodes"][LOCAL_NODE_ID]["address"]
        except (KeyError, TypeError):
            address = None
        return KnockListener(address)

    def run(self, *, knock=None):
        announced = False
        try:
            while True:
                # A knock that arrives from here on is about news this step may
                # not see yet, so it must survive to end the rest that follows.
                if knock is not None:
                    knock.drain()
                try:
                    result = self.step()
                except Exception as exc:
                    result = {"state": "BLOCKED", "error": type(exc).__name__}
                # No token, permit, private identity paths, or runtime bytes in logs.
                print(json.dumps(result, sort_keys=True), flush=True)
                if result["state"] == "STOPPED":
                    return 0
                # The listener is opened only once the loop is going to rest: a
                # supervisor told to stop when idle exits without any network use.
                if knock is None:
                    knock = self._knock_listener()
                if not announced:
                    print(json.dumps({"knock": "listening" if knock.listening else "timer-only"}), flush=True)
                    announced = True
                # The minimum gap is unconditional; only the remainder of an
                # idle or blocked rest can be cut short by a knock.
                time.sleep(STEP_SECONDS)
                knock.wait(pause_seconds(result["state"]) - STEP_SECONDS)
        finally:
            if knock is not None:
                knock.close()
