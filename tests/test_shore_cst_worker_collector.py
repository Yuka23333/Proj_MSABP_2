"""Offline CST-mode worker-to-collector integration with a mock child process."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

pytest.importorskip("shore_control")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.authority import Authority, digest
from shore_control.client import Client, MemorySigner
from shore_control.crypto import canonical_json, fingerprint, public_key_text, sign_object
from shore_control.permits import DESCRIPTOR_FIELDS

from test_control_cst import SuccessfulRunner, policy

from msabp_opt.simulation.distributed.config import DeviceConfig, DeviceRegistry, LaunchMode
from msabp_opt.simulation.distributed.control_authority import ControlAuthority
from msabp_opt.simulation.distributed.control_remote_worker import (
    BELL_ACTOR,
    CST_MODE,
    CST_PROJECT_SOURCE_FILES,
    DEVICE_ID,
    LOCAL_NODE_ID,
    MAID_ACTOR,
    MODE,
    SHORE_SOURCE_FILES,
    STARTUP_DOMAIN,
    RemoteSyntheticMaidConfig,
    run_worker,
)
from msabp_opt.simulation.distributed.http_api import PrincessClient
from msabp_opt.simulation.distributed.runtime import PrincessRuntime, prepare_run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _PhysicalRunner(SuccessfulRunner):
    """Use the tested mock child, then make physical provenance explicit."""

    def __call__(self, command, **kwargs):
        completed = super().__call__(command, **kwargs)
        request_path = Path(command[command.index("--request") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        manifest_path = Path(request["output_root"]) / ("case_" + request["case_id"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(synthetic=False, physical_result=True)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return completed


def test_cst_run_worker_commits_physical_result_and_dual_receipts(tmp_path, monkeypatch):
    import shore_control
    from msabp_opt.simulation.distributed import control_remote_worker as worker_module

    monkeypatch.delenv("MSABP_CONTROL_MODE", raising=False)
    root = Path(__file__).resolve().parents[1]
    shore_root = Path(shore_control.__file__).resolve().parent.parent
    keys = {
        actor: Ed25519PrivateKey.generate()
        for actor in ("authority", "controller", "session", BELL_ACTOR, MAID_ACTOR)
    }
    controller_id = "princess-11111111-1111-4111-8111-111111111111"
    host = "n8BsWasrMb11CNTRL"
    fleet = "shore-princess-" + "1234" * 6
    registrations = {
        actor: {
            "public_key": public_key_text(keys[actor].public_key()),
            "device_id": DEVICE_ID,
            "tailnet_node_id": LOCAL_NODE_ID,
            "role": role,
            "allowed_actions": [action],
        }
        for actor, role, action in (
            (BELL_ACTOR, "bell", "wake"),
            (MAID_ACTOR, "maid", "attempt"),
        )
    }
    bootstrap = {
        "schema": "shore.remote-bootstrap",
        "schema_version": 1,
        "mode": MODE,
        "fleet_id": fleet,
        "authority_id": fingerprint(keys["authority"].public_key()),
        "authority_public_key": public_key_text(keys["authority"].public_key()),
        "endpoint": "https://100.64.0.1:18766",
        "tls_sha256": "a" * 64,
        "network": {
            "tailnet_name": "offline-test",
            "magic_dns_suffix": "example.ts.net",
            "local_node_id": host,
            "local_address": "100.64.0.1",
            "nodes": {
                host: {"address": "100.64.0.1", "dns_name": "host.example.ts.net."},
                LOCAL_NODE_ID: {"address": "100.64.0.2", "dns_name": "maid.example.ts.net."},
            },
        },
        "controllers": {controller_id: public_key_text(keys["controller"].public_key())},
        "workers": registrations,
        "controller_nodes": [host, LOCAL_NODE_ID],
        "collector": {"endpoint": "https://100.64.0.1:18767", "tls_sha256": "b" * 64},
    }
    bootstrap_path = tmp_path / "authority-bootstrap.json"
    bootstrap_path.write_bytes(canonical_json(bootstrap))
    worker_dir = tmp_path / "identity"
    worker_dir.mkdir()
    worker_config = worker_dir / "worker-config.json"
    worker_config.write_bytes(canonical_json({
        "schema": "shore.remote-worker",
        "schema_version": 1,
        "device_id": DEVICE_ID,
        "tailnet_node_id": LOCAL_NODE_ID,
        **bootstrap["network"]["nodes"][LOCAL_NODE_ID],
        "workers": registrations,
        "key_paths": {
            actor: str(worker_dir / "keys" / (actor + ".pem"))
            for actor in registrations
        },
    }))

    authority = Authority(
        tmp_path / "authority.sqlite",
        keys["authority"],
        fleet_id=fleet,
        controllers=bootstrap["controllers"],
        workers=registrations,
    )

    def client(actor, key):
        return Client(
            authority,
            actor,
            MemorySigner(keys[key]),
            keys["authority"].public_key(),
            fleet,
        )

    controller = client(controller_id, "controller")
    session = client("session:session-offline", "session")
    worker_client = client(MAID_ACTOR, MAID_ACTOR)
    gate = ControlAuthority(
        tmp_path / "control.sqlite",
        controller_client=controller,
        session_client=session,
        worker_clients={MAID_ACTOR: worker_client},
        worker_registrations=registrations,
    )

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    local_url = f"http://127.0.0.1:{port}"
    device = DeviceConfig(
        id=MAID_ACTOR,
        enabled=True,
        launch_mode=LaunchMode.LOCAL,
        ssh_target=None,
        repo_root=str(tmp_path),
        python_path=str(policy(tmp_path / "solver-policy").python_executable),
    )
    registry = DeviceRegistry(
        bind_host="127.0.0.1",
        advertise_url=local_url,
        port=port,
        devices=(device,),
    )
    source_csv = tmp_path / "samples.csv"
    source_csv.write_text("sample_id,value\ncase-1,1\n", encoding="utf-8")
    preparation = prepare_run(
        source_csv=source_csv,
        run_id="run-cst-offline",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    job_manifest = tmp_path / "job-manifest.json"
    accounting = tmp_path / "accounting-policy.json"
    job_manifest.write_bytes(canonical_json({"job": "cst-offline", "approved_units": 1}))
    accounting.write_bytes(canonical_json({
        "schema": "msabp.budget-policy",
        "schema_version": 1,
        "unit_key": "evaluation_id",
        "settled_outcomes": ["completed"],
    }))
    spec = {
        "schema": "shore.job-spec",
        "schema_version": 1,
        "manifest_sha256": _sha(job_manifest),
        "approved_units": 1,
        "budget_unit": "objective_evaluation",
        "accounting_policy_sha256": _sha(accounting),
    }
    gate.acquire(
        job_id="cst-offline",
        host_id=host,
        session_id="session-offline",
        session_public_key=public_key_text(keys["session"].public_key()),
        project_id="MSABP_2",
        collector_url=bootstrap["collector"]["endpoint"],
        data_root=str(preparation.paths.results_root),
        job_spec=spec,
    )
    gate.bind_job_documents(job_manifest, accounting)
    csv_sha256 = _sha(preparation.paths.worklist_csv)
    gate.register_run(
        "run-cst-offline",
        _sha(preparation.paths.runtime_json),
        csv_sha256,
    )
    gate.bind_run_plan("run-cst-offline", preparation.paths.runtime_json)
    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=tmp_path / "unused-controller-template.cst",
        control_authority=gate,
        heartbeat_seconds=1,
        lease_seconds=10,
        poll_seconds=0.001,
    )
    runtime.start_server()

    solver_policy = policy(tmp_path / "child")
    project_template = solver_policy.project_template
    raw_runtime = {
        "schema": "msabp.shore-remote-cst-maid",
        "schema_version": 1,
        "mode": CST_MODE,
        "run_id": "run-cst-offline",
        "worker_id": MAID_ACTOR,
        "local_node_id": LOCAL_NODE_ID,
        "bootstrap_path": str(bootstrap_path),
        "bootstrap_sha256": _sha(bootstrap_path),
        "source_root": str(root),
        "source_hashes": {name: _sha(root / name) for name in CST_PROJECT_SOURCE_FILES},
        "shore_source_root": str(shore_root),
        "shore_source_hashes": {name: _sha(shore_root / name) for name in SHORE_SOURCE_FILES},
        "worker_config_path": str(worker_config),
        "csv_path": str(preparation.paths.worklist_csv),
        "csv_sha256": csv_sha256,
        "api_token": preparation.api_token,
        "binding": {**gate.binding, **gate.job["scope"], "fleet_id": fleet},
        "state_root": str(tmp_path / "maid-state"),
        "output_root": str(tmp_path / "maid-output"),
        "project_template_path": str(project_template),
        "project_template_sha256": _sha(project_template),
        "coordinate_quantum_mm": "0.01",
        "allow_disconnected_conductor": False,
        "command_timeout_seconds": 15,
        "poll_seconds": 2,
        "heartbeat_seconds": 5,
    }
    runtime_path = tmp_path / "maid-runtime.json"
    runtime_path.write_bytes(canonical_json(raw_runtime))
    config = RemoteSyntheticMaidConfig.load(runtime_path)

    wake = gate.issue_operation({
        "operation_id": "run-cst-offline.wake-001",
        "run_id": config["run_id"],
        "worker_id": BELL_ACTOR,
        "attempt_id": "run-cst-offline.wake-001",
        "action": "wake",
        "input_sha256": config.runtime_sha256,
        "action_context": {"target_actor_id": MAID_ACTOR, "runtime_sha256": config.runtime_sha256},
    })
    permit = wake["permit"]
    descriptor = {key: permit["payload"][key] for key in DESCRIPTOR_FIELDS}
    accepted = client(BELL_ACTOR, BELL_ACTOR).call("accept_permit", **{
        key: permit["payload"][key]
        for key in (
            "permit_id", "operation_id", "run_id", "attempt_id", "action",
            "input_sha256", "device_id",
        )
    })
    startup = sign_object(keys[BELL_ACTOR], STARTUP_DOMAIN, {
        "schema": STARTUP_DOMAIN,
        "schema_version": 1,
        "mode": MODE,
        "launch_id": "1" * 32,
        "runtime_sha256": config.runtime_sha256,
        "control_authority": {"descriptor": descriptor, "permit": permit},
        "acceptance": accepted,
        "worker_id": BELL_ACTOR,
        "target_actor_id": MAID_ACTOR,
    })

    runner = _PhysicalRunner()
    monkeypatch.setattr(worker_module, "CST_PYTHON_EXECUTABLE", solver_policy.python_executable)
    original_dependencies = worker_module._execution_dependencies

    def dependencies(selected):
        session_object, executor = original_dependencies(selected)
        executor._process_runner = runner
        return session_object, executor

    monkeypatch.setattr(worker_module, "_execution_dependencies", dependencies)

    princess = PrincessClient(local_url, preparation.api_token, timeout=5)
    idle_records = []

    class Collector:
        def send(self, message):
            return princess.send(message)

        def upload_artifact(self, *args):
            return princess.upload_artifact(*args)

        def request_json(self, path, body):
            assert path == "/v1/control/idle"
            idle_records.append(body["record"])
            return {"status": "RECORDED", "actor_id": MAID_ACTOR}

    class Verifier:
        def verify_local(self):
            pass

        def verify_peer(self, *_args):
            pass

    ready = config.state_root / ("ready." + "1" * 32 + ".json")
    completed = None
    try:
        assert run_worker(
            config,
            startup_record=startup,
            ready_record=ready,
            worker_client_factory=lambda *_a, **_k: worker_client,
            collector_client_factory=lambda *_a, **_k: Collector(),
            verifier_factory=lambda _network: Verifier(),
        ) == 0
        completed = runtime.state.progress("run-cst-offline").completed
    finally:
        runtime.close()

    assert len(runner.calls) == 1
    assert completed == 1
    with gate.journal.transaction() as database:
        rows = database.execute("SELECT operation_id,descriptor FROM control_operations").fetchall()
    attempts = [
        row["operation_id"]
        for row in rows
        if json.loads(row["descriptor"])["action"] == "attempt"
    ]
    assert len(attempts) == 1
    attempt = gate.get_operation(attempts[0])
    assert attempt["status"] == "RESOLVED"
    assert attempt["worker_receipt"] and attempt["controller_receipt"]
    # The sole unresolved operation is the deliberately bypassed mock Bell wake;
    # the physical evaluation itself has both signed receipts above.
    assert gate.status()["unresolved_count"] == 1
    result = preparation.paths.results_root / "case_case-1"
    assert {path.name for path in result.iterdir()} == {
        "manifest.json",
        "S11.csv",
        "Rad_Eff.csv",
        "Tot_Eff.csv",
        "Farfield Source [1].ffs",
        "geometry_tree.json",
    }
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is False and manifest["physical_result"] is True
    assert manifest["simulation_mode"] == "antenna_tree"
    assert idle_records
