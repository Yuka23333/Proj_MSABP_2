"""Explicit independent Maid runtime for the remote synthetic-only profile.

Preflight is offline and never opens a private key. Only explicit live run or
recovery entrypoints construct identity/transport. They retain Shore journals;
no dynamic executor, command line, CST doctor or shell is accepted.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sqlite3
import sys
import time
from typing import Any, Mapping
from urllib.parse import urlsplit

MODE = "shore-remote-synthetic-v1"
MAID_ACTOR = "coconutg2-maid"
BELL_ACTOR = "coconutg2-bell"
DEVICE_ID = "coconutg2"
LOCAL_NODE_ID = "nJc1aWG7ZA11CNTRL"
SCHEMA = "msabp.shore-remote-synthetic-maid"
READY_DOMAIN = "msabp.synthetic-maid-ready"
STARTUP_DOMAIN = "msabp.synthetic-bell-startup"
IDLE_DOMAIN = "msabp.synthetic-worker-idle"
MAX_CONFIG_BYTES = 1024 * 1024
TERMINAL_RECOVERY_ATTEMPTS = 3
TERMINAL_COMMIT_SECONDS = 5
PROJECT_SOURCE_FILES = (
    "scripts/simulation/shore_synthetic_maid.py",
    "scripts/simulation/shore_synthetic_bell.py",
    "src/msabp_opt/__init__.py", "src/msabp_opt/simulation/__init__.py",
    *("src/msabp_opt/simulation/distributed/" + name + ".py" for name in (
        "__init__", "state", "protocol", "maid", "case_runner", "http_api",
        "control_policy", "control_worker", "control_authority", "control_rpc",
        "control_failure", "control_synthetic", "control_remote_worker", "control_remote_http",
        "control_remote_bell")),
)
SHORE_SOURCE_FILES = tuple("shore_control/" + name + ".py" for name in (
    "__init__", "authority", "client", "crypto", "evidence", "job_contract",
    "permits", "registry", "remote", "remote_identity", "rpc", "tailnet", "tailnet_rpc", "usb_identity"))
BINDING_FIELDS = {"fleet_id", "job_id", "epoch", "host_id", "session_id",
                  "project_id", "collector_url", "data_root", "job_spec"}


def _absolute_path(value: str | Path) -> Path:
    path = Path(value)
    if (not path.is_absolute() or ".." in path.parts
            or path.anchor.startswith(("\\\\", "//"))):
        raise ValueError("Expected an absolute local non-traversing path")
    return path


def checked_path(value: str | Path) -> Path:
    path = _absolute_path(value)
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Redirected runtime paths are forbidden")
    return path


def sha256_file(path: str | Path) -> str:
    with checked_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate configuration field")
        result[key] = value
    return result


def _json_bytes(value) -> bytes:
    # Matches Shore's canonical integer-only subset without importing Shore.
    def check(item):
        if item is None or type(item) in (str, int, bool):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("Configuration accepts integer-only canonical JSON")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def read_json(path: str | Path) -> tuple[dict, bytes]:
    path = checked_path(path)
    if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("Missing or oversized configuration")
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique)
    if type(value) is not dict:
        raise ValueError("Configuration must be one object")
    _json_bytes(value)
    return value, data


def _hash(value, label):
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None or value == "0" * 64:
        raise ValueError(label + " requires a non-placeholder lowercase SHA256")
    return value


def _under(path, root):
    return path == root or root in path.parents


@dataclass(frozen=True)
class RemoteSyntheticMaidConfig:
    raw: Mapping[str, Any]
    runtime_sha256: str
    runtime_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, runtime_sha256: str | None = None,
                     runtime_path: Path | None = None):
        value = json.loads(_json_bytes(dict(raw)))
        required = {"schema", "schema_version", "mode", "run_id", "worker_id", "local_node_id",
                    "bootstrap_path", "bootstrap_sha256", "source_root", "source_hashes",
                    "shore_source_root", "shore_source_hashes", "worker_config_path",
                    "csv_path", "csv_sha256", "api_token", "binding", "state_root", "output_root"}
        optional = {"poll_seconds", "heartbeat_seconds"}
        if set(value) - required - optional or required - set(value):
            raise ValueError("Unexpected or missing remote Maid runtime fields")
        if (value["schema"] != SCHEMA or type(value["schema_version"]) is not int
                or value["schema_version"] != 1 or value["mode"] != MODE
                or value["worker_id"] != MAID_ACTOR or value["local_node_id"] != LOCAL_NODE_ID):
            raise ValueError("Explicit fixed remote synthetic Maid mode and actor required")
        for name in ("run_id", "api_token"):
            if type(value[name]) is not str or not value[name] or len(value[name]) > 256:
                raise ValueError("Invalid " + name)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value["run_id"]) is None:
            raise ValueError("Invalid run ID")
        for name in ("bootstrap_sha256", "csv_sha256"):
            _hash(value[name], name)
        for name in ("poll_seconds", "heartbeat_seconds"):
            if name in value and (type(value[name]) is not int or not 1 <= value[name] <= 300):
                raise ValueError("Timings must be integer seconds in [1, 300]")
        for name in ("bootstrap_path", "source_root", "shore_source_root", "worker_config_path",
                     "csv_path", "state_root", "output_root"):
            if type(value[name]) is not str:
                raise ValueError("Invalid path: " + name)
            # from_mapping also stages remote-host metadata locally. Filesystem
            # identity checks belong to that host's preflight, not this parser.
            _absolute_path(value[name])
        binding = value["binding"]
        if type(binding) is not dict or set(binding) != BINDING_FIELDS:
            raise ValueError("Independently pinned complete job binding required")
        if type(binding["epoch"]) is not int or binding["epoch"] < 1:
            raise ValueError("Invalid job epoch")
        for name in BINDING_FIELDS - {"epoch", "job_spec"}:
            if type(binding[name]) is not str or not binding[name]:
                raise ValueError("Invalid job binding")
        paths = {name: Path(value[name]) for name in (
            "state_root", "output_root", "source_root", "shore_source_root",
            "worker_config_path", "bootstrap_path", "csv_path")}
        for writable in (paths["state_root"], paths["output_root"]):
            if writable == Path(writable.anchor):
                raise ValueError("Runtime root cannot be a filesystem root")
            for protected in (paths["source_root"], paths["shore_source_root"],
                              paths["worker_config_path"].parent):
                if _under(writable, protected) or _under(protected, writable):
                    raise ValueError("Runtime state/results must be isolated from source and identity")
            for source in (paths["bootstrap_path"], paths["csv_path"], runtime_path):
                if source is not None and _under(Path(source), writable):
                    raise ValueError("Runtime state/results must not contain input configuration")
        if (_under(paths["state_root"], paths["output_root"])
                or _under(paths["output_root"], paths["state_root"])):
            raise ValueError("Results and durable state require disjoint roots")
        for key, names in (("source_hashes", PROJECT_SOURCE_FILES),
                           ("shore_source_hashes", SHORE_SOURCE_FILES)):
            if type(value[key]) is not dict or set(value[key]) != set(names):
                raise ValueError("Explicit complete fixed source manifest required: " + key)
            for expected in value[key].values():
                _hash(expected, key)
        digest = runtime_sha256 or hashlib.sha256(_json_bytes(value)).hexdigest()
        return cls(value, _hash(digest, "runtime_sha256"), runtime_path)

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str | None = None):
        source = checked_path(path)
        value, data = read_json(source)
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != _hash(expected_sha256, "expected runtime SHA256"):
            raise ValueError("Runtime SHA256 mismatch")
        return cls.from_mapping(value, runtime_sha256=digest, runtime_path=source)

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def state_root(self):
        return Path(self["state_root"])


def _check_sources(config):
    for root_key, hashes_key in (("source_root", "source_hashes"),
                                 ("shore_source_root", "shore_source_hashes")):
        for relative, expected in config[hashes_key].items():
            if sha256_file(Path(config[root_key]) / relative) != expected:
                raise ValueError("Pinned source changed: " + relative)
    actual = Path(__file__).absolute()
    if actual != Path(config["source_root"]) / "src/msabp_opt/simulation/distributed/control_remote_worker.py":
        raise ValueError("Maid module is not loaded from the pinned project source")
    for name, module in tuple(sys.modules.items()):
        if name == "msabp_opt" or name.startswith("msabp_opt."):
            filename = getattr(module, "__file__", None)
            if filename is None or not _under(Path(filename).absolute(), Path(config["source_root"]) / "src"):
                raise ValueError("Project dependency loaded from an unpinned location")
        if name == "shore_control" or name.startswith("shore_control."):
            filename = getattr(module, "__file__", None)
            if filename is None or not _under(Path(filename).absolute(), Path(config["shore_source_root"])):
                raise ValueError("Shore dependency loaded from an unpinned location")


def preflight_worker(config: RemoteSyntheticMaidConfig) -> dict:
    """Validate public bytes only; no daemon, sockets or private identity reads."""
    # Revalidate even a manually constructed dataclass.
    config = RemoteSyntheticMaidConfig.from_mapping(config.raw,
        runtime_sha256=config.runtime_sha256, runtime_path=config.runtime_path)
    for name in ("bootstrap_path", "source_root", "shore_source_root", "worker_config_path",
                 "csv_path", "state_root", "output_root"):
        checked_path(config[name])
    _check_sources(config)
    if sha256_file(config["bootstrap_path"]) != config["bootstrap_sha256"]:
        raise ValueError("Bootstrap SHA256 mismatch")
    if sha256_file(config["csv_path"]) != config["csv_sha256"]:
        raise ValueError("Input CSV SHA256 mismatch")
    if config["shore_source_root"] not in sys.path:
        sys.path.insert(0, config["shore_source_root"])
    from shore_control.remote import load_bootstrap
    from shore_control.job_contract import validate_job_spec

    bootstrap = load_bootstrap(config["bootstrap_path"])
    registration = bootstrap["workers"].get(MAID_ACTOR)
    if (not registration or registration["device_id"] != DEVICE_ID
            or registration["tailnet_node_id"] != LOCAL_NODE_ID
            or registration["role"] != "maid" or registration["allowed_actions"] != ["attempt"]):
        raise ValueError("Bootstrap differs from the fixed device-local Maid identity")
    binding = config["binding"]
    validate_job_spec(binding["job_spec"])
    if (binding["fleet_id"] != bootstrap["fleet_id"]
            or binding["collector_url"] != bootstrap["collector"]["endpoint"]
            or binding["host_id"] != bootstrap["network"]["local_node_id"]):
        raise ValueError("Frozen job differs from independent bootstrap")
    worker_path = Path(config["worker_config_path"])
    if worker_path.name != "worker-config.json":
        raise ValueError("Expected device-local worker-config.json")
    worker, _ = read_json(worker_path)  # Only paths/public metadata; never load_worker_config here.
    fields = {"schema", "schema_version", "device_id", "tailnet_node_id",
              "address", "dns_name", "workers", "key_paths"}
    node = bootstrap["network"]["nodes"][LOCAL_NODE_ID]
    if (set(worker) != fields or worker["schema"] != "shore.remote-worker"
            or type(worker["schema_version"]) is not int or worker["schema_version"] != 1
            or worker["device_id"] != DEVICE_ID or worker["tailnet_node_id"] != LOCAL_NODE_ID
            or worker["address"] != node["address"] or worker["dns_name"] != node["dns_name"]
            or worker["workers"] != bootstrap["workers"]
            or set(worker["workers"]) != {BELL_ACTOR, MAID_ACTOR}
            or worker["key_paths"] != {actor: str(worker_path.parent / "keys" / (actor + ".pem"))
                                       for actor in (BELL_ACTOR, MAID_ACTOR)}):
        raise ValueError("Device-local public registration differs from bootstrap")
    _check_sources(config)
    return {"schema": SCHEMA + "-preflight", "schema_version": 1, "mode": MODE,
            "runtime_sha256": config.runtime_sha256, "synthetic": True,
            "network_accessed": False, "private_keys_read": False, "cst": False,
            "gpu": False, "bootstrap": bootstrap}


def verify_startup_record(startup: dict, runtime: RemoteSyntheticMaidConfig, bootstrap: dict) -> dict:
    """Verify Bell authorization, accepted wake and frozen runtime independently."""
    from shore_control.crypto import verify_object
    from shore_control.permits import verify_acceptance, verify_permit
    from shore_control.registry import registration_sha256

    registration = bootstrap["workers"][BELL_ACTOR]
    record = verify_object(registration["public_key"], STARTUP_DOMAIN, startup)
    fields = {"schema", "schema_version", "mode", "launch_id", "runtime_sha256",
              "control_authority", "acceptance", "worker_id", "target_actor_id"}
    if (set(record) != fields or record["schema"] != STARTUP_DOMAIN
            or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["mode"] != MODE or record["worker_id"] != BELL_ACTOR
            or record["target_actor_id"] != MAID_ACTOR
            or record["runtime_sha256"] != runtime.runtime_sha256
            or type(record["launch_id"]) is not str
            or re.fullmatch(r"[a-f0-9]{32}", record["launch_id"]) is None):
        raise ValueError("Wake startup differs from the pinned runtime")
    gate = record["control_authority"]
    if type(gate) is not dict or set(gate) != {"descriptor", "permit"}:
        raise ValueError("Startup requires a complete wake gate")
    descriptor = gate["descriptor"]
    if type(descriptor) is not dict:
        raise ValueError("Invalid startup descriptor")
    expected = {**runtime["binding"], "operation_id": descriptor.get("operation_id"),
        "run_id": runtime["run_id"], "worker_id": BELL_ACTOR,
        "attempt_id": descriptor.get("attempt_id"), "action": "wake",
        "input_sha256": runtime.runtime_sha256, "device_id": DEVICE_ID, "worker_role": "bell",
        "worker_registration_sha256": registration_sha256(registration),
        "action_context": {"target_actor_id": MAID_ACTOR, "runtime_sha256": runtime.runtime_sha256}}
    if _json_bytes(descriptor) != _json_bytes(expected):
        raise ValueError("Wake descriptor differs from frozen job/runtime")
    permit = verify_permit(bootstrap["authority_public_key"], gate["permit"], expected)
    verify_acceptance(bootstrap["authority_public_key"], record["acceptance"], permit)
    return {**record, "verified_permit": permit}


@contextmanager
def instance_lock(state_root: str | Path):
    root = checked_path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    path = checked_path(root / "instance.lock")
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"1")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _persist_new(path, value):
    path = checked_path(path)
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _synthetic_execution_mode():
    previous = os.environ.get("MSABP_CONTROL_MODE")
    if previous not in (None, "", MODE):
        raise ValueError("Worker refuses to replace a different controlled mode")
    os.environ["MSABP_CONTROL_MODE"] = MODE
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MSABP_CONTROL_MODE", None)
        else:
            os.environ["MSABP_CONTROL_MODE"] = previous


def _clients(config, bootstrap, worker_client_factory, collector_client_factory, verifier_factory):
    from shore_control.remote import DEFAULT_TAILSCALE, client_from_bootstrap
    from shore_control.tailnet import TailnetVerifier
    from .control_remote_http import RemotePrincessClient

    network = {**bootstrap["network"], "local_node_id": LOCAL_NODE_ID,
        "local_address": bootstrap["network"]["nodes"][LOCAL_NODE_ID]["address"]}
    verifier = (verifier_factory(network) if verifier_factory is not None
                else TailnetVerifier(network, DEFAULT_TAILSCALE))
    verifier.verify_local()
    peer = bootstrap["network"]["local_node_id"]
    endpoint = urlsplit(bootstrap["endpoint"])
    verifier.verify_peer(endpoint.hostname, endpoint.port, peer)
    worker = (worker_client_factory or client_from_bootstrap)(bootstrap, MAID_ACTOR,
        local_node_id=LOCAL_NODE_ID, worker_config=config["worker_config_path"])
    if worker.actor_id != MAID_ACTOR or worker.fleet_id != bootstrap["fleet_id"]:
        raise ValueError("Constructed worker differs from fixed identity")
    collector = (collector_client_factory or RemotePrincessClient)(
        bootstrap, MAID_ACTOR, worker.signer, verifier, run_id=config["run_id"])
    return worker, collector


def _idle_payload(config, permit_ids):
    return {"schema": IDLE_DOMAIN, "schema_version": 1, "mode": MODE,
        "actor_id": MAID_ACTOR, "fleet_id": config["binding"]["fleet_id"],
        "job_id": config["binding"]["job_id"], "epoch": config["binding"]["epoch"],
        "run_id": config["run_id"], "runtime_sha256": config.runtime_sha256,
        "permit_ids": permit_ids, "outboxes_pending": 0}


def _read_idle_permits(config, *, database_opener=None):
    """Read-only final-state proof; never initialize or repair either journal."""
    database = checked_path(config.state_root / "control" / "worker.sqlite")
    rpc_database = checked_path(config.state_root / "control" / "rpc.sqlite")
    if not database.is_file() or not rpc_database.is_file():
        raise ValueError("Idle recovery requires both existing original journals")
    opener = database_opener or (lambda path: closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)))
    with opener(database) as db:
        rows = db.execute("SELECT permit_id,state,resolved FROM inbox ORDER BY permit_id").fetchall()
        if any(state != "RESOLVED" or resolved is None for _, state, resolved in rows):
            raise ValueError("Worker inbox/outbox is not fully resolved")
    with opener(rpc_database) as db:
        pending = db.execute("SELECT COUNT(*) FROM rpc_journal WHERE result IS NULL AND error IS NULL").fetchone()[0]
        if pending:
            raise ValueError("Worker RPC journal still has pending requests")
    return [row[0] for row in rows]


def _send_idle(collector, envelope):
    response = collector.request_json("/v1/control/idle", {"record": envelope})
    if response != {"status": "RECORDED", "actor_id": MAID_ACTOR}:
        raise ValueError("Collector did not acknowledge the exact worker idle record")


def _persist_worker_idle(config, gate, worker, collector):
    if gate.pending_assignments() or gate.rpc.pending():
        raise ValueError("Worker cannot claim idle with pending assignments or signed RPC")
    permit_ids = _read_idle_permits(config)
    envelope = worker.signer.sign(IDLE_DOMAIN, _idle_payload(config, permit_ids))
    _persist_new(config.state_root / "idle.json", envelope)
    _send_idle(collector, envelope)


def report_worker_idle(config: RemoteSyntheticMaidConfig, *, worker_client_factory=None,
                       collector_client_factory=None, verifier_factory=None) -> int:
    """Replay an existing immutable signed idle record; never start or recreate work."""
    bootstrap = preflight_worker(config)["bootstrap"]
    from shore_control.crypto import verify_object
    with _synthetic_execution_mode(), instance_lock(config.state_root):
        envelope, _ = read_json(config.state_root / "idle.json")
        value = verify_object(bootstrap["workers"][MAID_ACTOR]["public_key"], IDLE_DOMAIN, envelope)
        expected = _idle_payload(config, _read_idle_permits(config))
        if _json_bytes(value) != _json_bytes(expected):
            raise ValueError("Existing signed idle record differs from this runtime/journal")
        _, collector = _clients(config, bootstrap, worker_client_factory,
                                 collector_client_factory, verifier_factory)
        _send_idle(collector, envelope)
    return 0


def _terminal_records(config):
    """Require existing terminal journals before opening identity or retrying work."""
    database = checked_path(config.state_root / "control" / "worker.sqlite")
    rpc_database = checked_path(config.state_root / "control" / "rpc.sqlite")
    if not database.is_file() or not rpc_database.is_file():
        raise ValueError("Terminal recovery requires both existing original journals")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute("SELECT * FROM inbox ORDER BY rowid")]
    if not rows:
        raise ValueError("Terminal recovery requires existing durable terminal work")
    for row in rows:
        if row["state"] not in {"RESULT", "RESOLVED"} or not row["result"] or not row["assignment"]:
            raise ValueError("Terminal recovery refuses nonterminal or uncertain execution")
        descriptor = json.loads(row["descriptor"])
        expected = {**config["binding"], "run_id": config["run_id"], "worker_id": MAID_ACTOR}
        if any(descriptor.get(key) != value for key, value in expected.items()):
            raise ValueError("Terminal journal differs from the original runtime binding")
    return rows


def _verify_recovery_launch(config, bootstrap):
    """Recovery consumes the original signed launch, never a replacement wake."""
    from shore_control.crypto import verify_object

    launches = list(checked_path(config.state_root).glob("launch.*.json"))
    if len(launches) != 1:
        raise ValueError("Terminal recovery requires exactly one original launch record")
    envelope, _ = read_json(launches[0])
    startup = verify_startup_record(envelope, config, bootstrap)
    launch_id = startup["launch_id"]
    if launches[0].name != "launch." + launch_id + ".json":
        raise ValueError("Original launch filename differs from signed launch identity")
    ready, _ = read_json(config.state_root / ("ready." + launch_id + ".json"))
    value = verify_object(bootstrap["workers"][MAID_ACTOR]["public_key"], READY_DOMAIN, ready)
    permit = startup["verified_permit"]
    expected = {"schema": READY_DOMAIN, "schema_version": 1, "launch_id": launch_id,
        "permit_id": permit["permit_id"], "runtime_sha256": config.runtime_sha256,
        "actor_id": MAID_ACTOR, "device_id": DEVICE_ID, "fleet_id": bootstrap["fleet_id"],
        "job_id": permit["job_id"], "epoch": permit["epoch"], "run_id": config["run_id"]}
    if (set(value) != set(expected) | {"pid", "process_started_utc"}
            or any(value.get(key) != expected_value for key, expected_value in expected.items())
            or type(value["pid"]) is not int or value["pid"] < 1
            or type(value["process_started_utc"]) is not str):
        raise ValueError("Original signed readiness differs from this launch/runtime")


def _maid_config(config, bootstrap):
    from .maid import MaidRuntimeConfig

    return MaidRuntimeConfig.from_mapping({"schema_version": 1, "protocol_mode": MODE,
        "run_id": config["run_id"], "worker_id": MAID_ACTOR,
        "princess_url": bootstrap["collector"]["endpoint"], "api_token": config["api_token"],
        "csv_path": config["csv_path"], "csv_sha256": config["csv_sha256"],
        "project_path": str(config.state_root / "NEVER-OPEN-SYNTHETIC.cst"),
        "output_root": config["output_root"], "dry_run": True,
        "poll_seconds": config.raw.get("poll_seconds", 5),
        "heartbeat_seconds": config.raw.get("heartbeat_seconds", 15)})


def _no_recovery_execution(*_args, **_kwargs):
    raise RuntimeError("Terminal-only recovery cannot execute a sample")


def _recover_terminal_outboxes(config, maid, gate) -> bool:
    """Bounded durable-result replay, with no hello, task request or executor.

    The original Maid.run is deliberately not restarted: it could poll for new
    assignments. Its assignment replay is used only after every inbox entry is
    terminal and every unresolved artifact has passed checked_result.
    """
    original_config, original_executor = maid.config, maid._run_csv_row
    maid.config = replace(maid.config, artifact_commit_deadline_seconds=TERMINAL_COMMIT_SECONDS)
    maid._run_csv_row = _no_recovery_execution
    try:
        for retry in range(TERMINAL_RECOVERY_ATTEMPTS):
            records = _terminal_records(config)
            # Validate the entire pending batch before any signed RPC/delivery.
            for record in records:
                if record["state"] != "RESOLVED":
                    gate.checked_result(record["attempt_id"])
                    delivery = gate.delivery(record["attempt_id"])
                    if delivery is not None and "archive" in delivery:
                        if sha256_file(delivery["archive"]) != delivery["payload"]["archive_sha256"]:
                            raise ValueError("Durable Shore archive bytes changed")
            try:
                for record in records:
                    if record["state"] == "RESOLVED":
                        continue
                    attempt_id = record["attempt_id"]
                    gate.report(attempt_id)
                    # Most completed_pending failures need only this read, not
                    # a duplicate completion or an already-stopped collector.
                    if not gate.resolved(attempt_id):
                        maid._run_assignment(json.loads(record["assignment"]))
                if not gate.pending_assignments() and not gate.rpc.pending():
                    _read_idle_permits(config)
                    return True
            except Exception as exc:
                if not maid._is_control_network_error(exc):
                    raise
                maid._log("Terminal recovery unavailable; retaining original journals and result bytes")
            if retry + 1 < TERMINAL_RECOVERY_ATTEMPTS:
                time.sleep(1)
        return False
    finally:
        maid.config, maid._run_csv_row = original_config, original_executor


def recover_worker_terminal(config: RemoteSyntheticMaidConfig, *, worker_client_factory=None,
                            collector_client_factory=None, verifier_factory=None) -> int:
    """Settle only the original wake's durable terminal outboxes after process loss.

    Return 0 only after idle acknowledgement; 2 retains unresolved state for a
    later invocation. An existing idle.json instead belongs to --report-idle.
    """
    bootstrap = preflight_worker(config)["bootstrap"]
    from .control_synthetic import SyntheticSession
    from .control_worker import ShoreWorker
    from .maid import Maid

    with _synthetic_execution_mode(), instance_lock(config.state_root):
        if (config.state_root / "idle.json").exists():
            raise FileExistsError("Runtime already declared idle; use --report-idle")
        _verify_recovery_launch(config, bootstrap)
        _terminal_records(config)
        worker, collector = _clients(config, bootstrap, worker_client_factory,
                                     collector_client_factory, verifier_factory)
        gate = ShoreWorker(config.state_root / "control", worker, bootstrap["authority_public_key"],
            binding=config["binding"], registration=bootstrap["workers"][MAID_ACTOR],
            csv_sha256=config["csv_sha256"])
        maid = Maid(_maid_config(config, bootstrap), client=collector, artifact_client=collector,
                    control_worker=gate, session=SyntheticSession(), run_csv_row=_no_recovery_execution)
        try:
            _check_sources(config)
            if not _recover_terminal_outboxes(config, maid, gate):
                return 2
            _persist_worker_idle(config, gate, worker, collector)
            return 0
        finally:
            maid.session.close()


def run_worker(config: RemoteSyntheticMaidConfig, *, startup_record: dict,
               ready_record: str | Path, worker_client_factory=None,
               collector_client_factory=None, verifier_factory=None, maid_factory=None) -> int:
    """Run only the fixed synthetic executor; injected dependencies are for offline tests."""
    report = preflight_worker(config)
    bootstrap = report["bootstrap"]
    startup = verify_startup_record(startup_record, config, bootstrap)
    ready = checked_path(ready_record)
    if ready != config.state_root / ("ready." + startup["launch_id"] + ".json"):
        raise ValueError("Readiness must use the dedicated state-root launch path")
    from .control_synthetic import SyntheticSession, synthetic_case
    from .control_worker import ShoreWorker
    from .maid import Maid

    with _synthetic_execution_mode(), instance_lock(config.state_root):
        if (config.state_root / "idle.json").exists():
            raise FileExistsError("Runtime already declared idle; only --report-idle recovery is permitted")
        # This immutable per-wake marker prevents replay even after normal exit.
        _persist_new(config.state_root / ("launch." + startup["launch_id"] + ".json"), startup_record)
        worker_client, client = _clients(config, bootstrap, worker_client_factory,
                                         collector_client_factory, verifier_factory)
        gate = ShoreWorker(config.state_root / "control", worker_client, bootstrap["authority_public_key"],
            binding=config["binding"], registration=bootstrap["workers"][MAID_ACTOR],
            csv_sha256=config["csv_sha256"])
        maid = (maid_factory or Maid)(_maid_config(config, bootstrap), client=client, artifact_client=client,
            control_worker=gate, session=SyntheticSession(), run_csv_row=synthetic_case)
        _check_sources(config)
        permit = startup["verified_permit"]
        payload = {"schema": READY_DOMAIN, "schema_version": 1,
            "launch_id": startup["launch_id"], "permit_id": permit["permit_id"],
            "runtime_sha256": config.runtime_sha256, "actor_id": MAID_ACTOR,
            "device_id": DEVICE_ID, "fleet_id": bootstrap["fleet_id"],
            "job_id": permit["job_id"], "epoch": permit["epoch"], "run_id": config["run_id"],
            "pid": os.getpid(), "process_started_utc": datetime.now(timezone.utc).isoformat()}
        _persist_new(ready, worker_client.signer.sign(READY_DOMAIN, payload))
        result = maid.run()
        if result == 2:
            # A normal failure/paused run without terminal work is not recovery
            # authority. STARTING and accepted-only work remain fail closed.
            try:
                _terminal_records(config)
            except ValueError:
                return result
            if _recover_terminal_outboxes(config, maid, gate):
                result = 0
        if result == 0:
            _persist_worker_idle(config, gate, worker_client, client)
        return result
