"""Offline remote worker contract checks: generated keys and in-memory RPC only."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("shore_control")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.authority import Authority, digest
from shore_control.client import Client, MemorySigner
from shore_control.crypto import canonical_json, fingerprint, public_key_text, sign_object, verify_object
from shore_control.permits import DESCRIPTOR_FIELDS

from msabp_opt.simulation.distributed.control_remote_worker import (
    BELL_ACTOR, DEVICE_ID, LOCAL_NODE_ID, MAID_ACTOR, MODE, PROJECT_SOURCE_FILES,
    IDLE_DOMAIN, READY_DOMAIN, SCHEMA, SHORE_SOURCE_FILES, STARTUP_DOMAIN,
    RemoteSyntheticMaidConfig, instance_lock, preflight_worker, report_worker_idle, run_worker,
    verify_startup_record,
)
from msabp_opt.simulation.distributed.control_synthetic import SyntheticSession, synthetic_case
from msabp_opt.simulation.distributed.protocol import make_message


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def remote_worker(tmp_path, monkeypatch):
    import shore_control

    monkeypatch.delenv("MSABP_CONTROL_MODE", raising=False)
    root = Path(__file__).resolve().parents[1]
    shore_root = Path(shore_control.__file__).resolve().parent.parent
    keys = {actor: Ed25519PrivateKey.generate() for actor in (
        "authority", "controller", "session", BELL_ACTOR, MAID_ACTOR)}
    controller_id = "princess-11111111-1111-4111-8111-111111111111"
    host = "n8BsWasrMb11CNTRL"
    registrations = {actor: {"public_key": public_key_text(keys[actor].public_key()),
        "device_id": DEVICE_ID, "tailnet_node_id": LOCAL_NODE_ID, "role": role,
        "allowed_actions": [action]}
        for actor, role, action in ((BELL_ACTOR, "bell", "wake"), (MAID_ACTOR, "maid", "attempt"))}
    fleet = "shore-princess-" + "1234" * 6
    bootstrap = {"schema": "shore.remote-bootstrap", "schema_version": 1, "mode": MODE,
        "fleet_id": fleet, "authority_id": fingerprint(keys["authority"].public_key()),
        "authority_public_key": public_key_text(keys["authority"].public_key()),
        "endpoint": "https://100.64.0.1:18766", "tls_sha256": "a" * 64,
        "network": {"tailnet_name": "offline-test", "magic_dns_suffix": "example.ts.net",
            "local_node_id": host, "local_address": "100.64.0.1", "nodes": {
                host: {"address": "100.64.0.1", "dns_name": "host.example.ts.net."},
                LOCAL_NODE_ID: {"address": "100.64.0.2", "dns_name": "maid.example.ts.net."}}},
        "controllers": {controller_id: public_key_text(keys["controller"].public_key())},
        "workers": registrations, "controller_nodes": [host, LOCAL_NODE_ID],
        "collector": {"endpoint": "https://100.64.0.1:18767", "tls_sha256": "b" * 64}}
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_bytes(canonical_json(bootstrap))
    worker_dir = tmp_path / "device"
    worker_dir.mkdir()
    worker_path = worker_dir / "worker-config.json"
    worker_path.write_bytes(canonical_json({"schema": "shore.remote-worker", "schema_version": 1,
        "device_id": DEVICE_ID, "tailnet_node_id": LOCAL_NODE_ID,
        **bootstrap["network"]["nodes"][LOCAL_NODE_ID], "workers": registrations,
        "key_paths": {actor: str(worker_dir / "keys" / (actor + ".pem")) for actor in registrations}}))
    # No private key file is created or accessed: identities exist only in memory.
    authority = Authority(tmp_path / "authority.sqlite", keys["authority"], fleet_id=fleet,
        controllers=bootstrap["controllers"], workers=registrations)
    def client(actor, key):
        return Client(authority, actor, MemorySigner(keys[key]), keys["authority"].public_key(), fleet)
    controller = client(controller_id, "controller")
    worker_client = client(MAID_ACTOR, MAID_ACTOR)
    job_spec = {"schema": "shore.job-spec", "schema_version": 1,
        "manifest_sha256": digest({"synthetic": True}), "approved_units": 4,
        "budget_unit": "objective_evaluation", "accounting_policy_sha256": digest({"completed": True})}
    job = controller.call("acquire", job_id="job-offline", host_id=host, session_id="session-offline",
        session_public_key=public_key_text(keys["session"].public_key()), project_id="MSABP_2",
        collector_url=bootstrap["collector"]["endpoint"], data_root=str(tmp_path), job_spec=job_spec)
    binding = {key: job[key] for key in ("job_id", "epoch", "host_id", "session_id")}
    binding.update(job["scope"], fleet_id=fleet)
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("sample_id,value\ncase-1,1\n", encoding="utf-8")
    raw = {"schema": SCHEMA, "schema_version": 1, "mode": MODE,
        "run_id": "run-offline", "worker_id": MAID_ACTOR, "local_node_id": LOCAL_NODE_ID,
        "bootstrap_path": str(bootstrap_path), "bootstrap_sha256": sha(bootstrap_path),
        "source_root": str(root), "source_hashes": {name: sha(root / name) for name in PROJECT_SOURCE_FILES},
        "shore_source_root": str(shore_root),
        "shore_source_hashes": {name: sha(shore_root / name) for name in SHORE_SOURCE_FILES},
        "worker_config_path": str(worker_path), "csv_path": str(csv_path), "csv_sha256": sha(csv_path),
        "api_token": "offline-unused-token", "binding": binding,
        "state_root": str(tmp_path / "maid-state"), "output_root": str(tmp_path / "maid-output")}
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_bytes(canonical_json(raw))
    config = RemoteSyntheticMaidConfig.load(runtime_path)
    bound = {key: binding[key] for key in ("job_id", "epoch", "host_id", "session_id")}
    controller.call("register_run", **bound, run_id=config["run_id"],
        original_plan_sha256=digest({"plan": "offline"}), input_manifest_sha256=config["csv_sha256"])
    permit = controller.call("validate_dispatch", **bound, operation={
        "operation_id": "wake-offline", "run_id": config["run_id"], "worker_id": BELL_ACTOR,
        "attempt_id": "wake-offline", "action": "wake", "input_sha256": config.runtime_sha256,
        "action_context": {"target_actor_id": MAID_ACTOR, "runtime_sha256": config.runtime_sha256}})
    descriptor = {key: permit["payload"][key] for key in DESCRIPTOR_FIELDS}
    accepted = client(BELL_ACTOR, BELL_ACTOR).call("accept_permit", **{
        key: permit["payload"][key] for key in ("permit_id", "operation_id", "run_id", "attempt_id",
                                               "action", "input_sha256", "device_id")})
    startup = sign_object(keys[BELL_ACTOR], STARTUP_DOMAIN, {
        "schema": STARTUP_DOMAIN, "schema_version": 1, "mode": MODE, "launch_id": "1" * 32,
        "runtime_sha256": config.runtime_sha256,
        "control_authority": {"descriptor": descriptor, "permit": permit}, "acceptance": accepted,
        "worker_id": BELL_ACTOR, "target_actor_id": MAID_ACTOR})
    return SimpleNamespace(root=tmp_path, config=config, raw=raw, bootstrap=bootstrap, keys=keys,
        worker_client=worker_client, startup=startup, controller=controller, authority=authority)


def test_offline_preflight_never_opens_key_daemon_or_socket(remote_worker, monkeypatch):
    monkeypatch.setattr("socket.socket", lambda *_a, **_k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("Subprocess forbidden"))
    from shore_control import remote
    monkeypatch.setattr(remote, "load_worker_config", lambda *_a: pytest.fail("Private identity read forbidden"))
    report = preflight_worker(remote_worker.config)
    assert report["network_accessed"] is report["private_keys_read"] is False
    assert report["synthetic"] is True
    assert not remote_worker.config.state_root.exists()


@pytest.mark.parametrize("field,value", [
    ("mode", "shore-fake-v1"), ("worker_id", BELL_ACTOR),
    ("local_node_id", "coconutg2"), ("poll_seconds", 0.5),
    ("schema_version", True), ("command", "arbitrary.exe"),
    ("private_key", "secret"), ("bootstrap_sha256", "0" * 64),
])
def test_strict_runtime_rejects_downgrade_dynamic_code_and_unknown_fields(remote_worker, field, value):
    raw = {**remote_worker.raw, field: value}
    with pytest.raises(ValueError):
        RemoteSyntheticMaidConfig.from_mapping(raw)


@pytest.mark.parametrize("target", ["source_root", "shore_source_root", "state_root"])
def test_results_cannot_overlap_source_or_durable_identity(remote_worker, target):
    raw = {**remote_worker.raw, "output_root": remote_worker.raw[target]}
    with pytest.raises(ValueError, match="isolated|disjoint"):
        RemoteSyntheticMaidConfig.from_mapping(raw)


def test_source_and_bootstrap_pins_fail_closed(remote_worker):
    raw = copy.deepcopy(remote_worker.raw)
    raw["source_hashes"][PROJECT_SOURCE_FILES[0]] = "f" * 64
    with pytest.raises(ValueError, match="Pinned source"):
        preflight_worker(RemoteSyntheticMaidConfig.from_mapping(raw))
    raw = {**remote_worker.raw, "bootstrap_sha256": "f" * 64}
    with pytest.raises(ValueError, match="Bootstrap SHA256"):
        preflight_worker(RemoteSyntheticMaidConfig.from_mapping(raw))


def test_duplicate_runtime_json_and_changed_raw_hash_refused(remote_worker):
    source = remote_worker.root / "duplicate.json"
    source.write_text('{"schema":1,"schema":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        RemoteSyntheticMaidConfig.load(source)
    with pytest.raises(ValueError, match="Runtime SHA256"):
        RemoteSyntheticMaidConfig.load(remote_worker.config.runtime_path, expected_sha256="a" * 64)


def test_signed_startup_requires_fixed_runtime_and_acceptance(remote_worker):
    value = verify_startup_record(remote_worker.startup, remote_worker.config, remote_worker.bootstrap)
    assert value["verified_permit"]["action"] == "wake"
    changed = copy.deepcopy(remote_worker.startup)
    changed["payload"]["target_actor_id"] = BELL_ACTOR
    changed = sign_object(remote_worker.keys[BELL_ACTOR], STARTUP_DOMAIN, changed["payload"])
    with pytest.raises(ValueError, match="pinned runtime"):
        verify_startup_record(changed, remote_worker.config, remote_worker.bootstrap)
    changed = copy.deepcopy(remote_worker.startup)
    changed["payload"]["acceptance"]["payload"]["permit_id"] = "other"
    changed = sign_object(remote_worker.keys[BELL_ACTOR], STARTUP_DOMAIN, changed["payload"])
    with pytest.raises(Exception, match="acceptance"):
        verify_startup_record(changed, remote_worker.config, remote_worker.bootstrap)


def test_instance_lock_is_os_owned_and_survives_stale_file(tmp_path):
    with instance_lock(tmp_path):
        with pytest.raises(OSError):
            with instance_lock(tmp_path):
                pytest.fail("Concurrent instance admitted")
    with instance_lock(tmp_path):
        pass


def test_fixed_executor_small_allowlisted_and_clearly_nonphysical(tmp_path):
    result = synthetic_case({"private_key": "must-not-be-exported"}, output_root=tmp_path, case_id="one")
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["synthetic"] is True and manifest["physical_result"] is False
    assert manifest["dry_run"] is False
    assert set(manifest["artifacts"]) == {"s11", "rad_eff", "tot_eff", "farfield_source"}
    paths = list(result.case_directory.iterdir())
    assert len(paths) == 5 and sum(path.stat().st_size for path in paths) < 1024 * 1024
    assert all(b"must-not-be-exported" not in path.read_bytes() for path in paths)
    assert SyntheticSession().open() is None


def test_run_uses_real_maid_but_mock_transports_signs_ready_and_refuses_replay(remote_worker, monkeypatch):
    monkeypatch.setattr("socket.socket", lambda *_a, **_k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("Subprocess forbidden"))
    monkeypatch.setattr("platform.node", lambda: "offline-test")
    messages, checks, idles = [], [], []
    class Verifier:
        def verify_local(self):
            checks.append("local")
        def verify_peer(self, *_args):
            checks.append("peer")
    class Collector:
        def send(self, message):
            messages.append(message["type"])
            return make_message("welcome" if message["type"] == "hello" else "stop",
                run_id=message["run_id"], worker_id=message["worker_id"], protocol_version=2)
        def request_json(self, path, body):
            assert path == "/v1/control/idle"
            idles.append(body["record"])
            return {"status": "RECORDED", "actor_id": MAID_ACTOR}
    ready = remote_worker.config.state_root / ("ready." + "1" * 32 + ".json")
    kwargs = dict(startup_record=remote_worker.startup, ready_record=ready,
        worker_client_factory=lambda *_a, **_k: remote_worker.worker_client,
        collector_client_factory=lambda *_a, **_k: Collector(),
        verifier_factory=lambda _network: Verifier())
    assert run_worker(remote_worker.config, **kwargs) == 0
    assert messages == ["hello", "request_task"] and checks == ["local", "peer"]
    value = verify_object(remote_worker.bootstrap["workers"][MAID_ACTOR]["public_key"],
                          READY_DOMAIN, json.loads(ready.read_bytes()))
    assert value["pid"] == os.getpid() and value["runtime_sha256"] == remote_worker.config.runtime_sha256
    assert value["actor_id"] == MAID_ACTOR and value["launch_id"] == "1" * 32
    idle = verify_object(remote_worker.bootstrap["workers"][MAID_ACTOR]["public_key"],
                          IDLE_DOMAIN, idles[0])
    assert idle["permit_ids"] == [] and idle["outboxes_pending"] == 0
    replay_kwargs = {key: value for key, value in kwargs.items()
                     if key not in {"startup_record", "ready_record"}}
    assert report_worker_idle(remote_worker.config, **replay_kwargs) == 0
    assert idles[0] == idles[1] and messages == ["hello", "request_task"]
    with pytest.raises(FileExistsError):
        run_worker(remote_worker.config, **kwargs)


def test_no_startup_is_not_an_authorized_run(remote_worker):
    with pytest.raises(ValueError):
        run_worker(remote_worker.config, startup_record={},
            ready_record=remote_worker.config.state_root / "ready.json",
            worker_client_factory=lambda *_a, **_k: pytest.fail("Unauthenticated client construction"))


def test_idle_missing_or_pending_state_never_constructs_client(remote_worker):
    with pytest.raises(ValueError, match="Missing"):
        report_worker_idle(remote_worker.config,
            worker_client_factory=lambda *_a, **_k: pytest.fail("No idle means no identity access"))


def test_idle_upload_lost_reply_retains_immutable_record_for_replay(remote_worker, monkeypatch):
    monkeypatch.setattr("socket.socket", lambda *_a, **_k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("Subprocess forbidden"))
    sent = []
    class Verifier:
        def verify_local(self):
            pass
        def verify_peer(self, *_a):
            pass
    class Collector:
        fail = True
        def request_json(self, path, body):
            sent.append(body["record"])
            if self.fail:
                raise TimeoutError("Mock acknowledgement lost")
            return {"status": "RECORDED", "actor_id": MAID_ACTOR}
    collector = Collector()
    factories = dict(worker_client_factory=lambda *_a, **_k: remote_worker.worker_client,
        collector_client_factory=lambda *_a, **_k: collector,
        verifier_factory=lambda _network: Verifier())
    with pytest.raises(TimeoutError):
        run_worker(remote_worker.config, startup_record=remote_worker.startup,
            ready_record=remote_worker.config.state_root / ("ready." + "1" * 32 + ".json"),
            maid_factory=lambda *_a, **_k: SimpleNamespace(run=lambda: 0), **factories)
    original = (remote_worker.config.state_root / "idle.json").read_bytes()
    collector.fail = False
    assert report_worker_idle(remote_worker.config, **factories) == 0
    assert sent[0] == sent[1]
    assert (remote_worker.config.state_root / "idle.json").read_bytes() == original


def test_pending_rpc_refuses_idle_even_if_maid_returns_success(remote_worker, monkeypatch):
    from msabp_opt.simulation.distributed.control_rpc import SignedRpcJournal

    monkeypatch.setattr(SignedRpcJournal, "pending", lambda _self: [{"key": "unresolved"}])
    class Verifier:
        def verify_local(self):
            pass
        def verify_peer(self, *_a):
            pass
    with pytest.raises(ValueError, match="pending assignments or signed RPC"):
        run_worker(remote_worker.config, startup_record=remote_worker.startup,
            ready_record=remote_worker.config.state_root / ("ready." + "1" * 32 + ".json"),
            worker_client_factory=lambda *_a, **_k: remote_worker.worker_client,
            collector_client_factory=lambda *_a, **_k: SimpleNamespace(
                request_json=lambda *_a: pytest.fail("Pending RPC must not claim idle")),
            verifier_factory=lambda _network: Verifier(),
            maid_factory=lambda *_a, **_k: SimpleNamespace(run=lambda: 0))
    assert not (remote_worker.config.state_root / "idle.json").exists()
