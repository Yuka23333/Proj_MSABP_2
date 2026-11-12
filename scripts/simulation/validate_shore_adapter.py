"""Exercise real Princess/HTTP/Maid entry points against the existing Shore lab.

Synthetic files only. No CST session, GPU, SSH, Bell or real USB key is used.
Use a new --output directory; recovery files are retained on any failure.
"""
from __future__ import annotations

import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import secrets
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SOURCE_FILES = (
    "scripts/simulation/validate_shore_adapter.py", "scripts/simulation/princess.py",
    "scripts/optimization/run_krvea_tree_14d.py",
    *("src/msabp_opt/simulation/distributed/" + name + ".py" for name in (
        "control_authority", "control_rpc", "control_worker", "control_collector", "control_failure", "control_policy",
        "state", "princess", "maid", "http_api", "protocol", "runtime", "transport", "bell")),
    *("src/msabp_opt/optimization/" + name + ".py" for name in (
        "proposal_relay", "krvea_relay", "phase2_krvea_relay", "phase2_krvea_tree_relay")),
    *("tests/" + name + ".py" for name in (
        "test_control_authority", "test_shore_maid", "test_shore_state", "test_shore_project_integration",
        "test_shore_execution_policy", "test_shore_failures", "test_shore_unissued_recovery",
        "test_shore_rpc_recovery", "test_shore_terminal_recovery", "test_shore_dispatch_recovery")),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


class SyntheticSession:
    def open(self):
        return None

    def close(self):
        pass


def synthetic_case(row, **kwargs):
    from msabp_opt.simulation.distributed.case_runner import CaseRunResult

    case_id = kwargs["case_id"]
    directory = Path(kwargs["output_root"]) / ("case_" + case_id)
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, filename in (("s11", "S11.csv"), ("rad_eff", "Rad_Eff.csv"),
                           ("tot_eff", "Tot_Eff.csv"), ("farfield_source", "Farfield Source [1].ffs")):
        path = directory / filename
        path.write_bytes(("SYNTHETIC ONLY " + case_id + " " + name + "\n").encode())
        artifacts[name] = {"path": filename, "sha256": sha(path), "size_bytes": path.stat().st_size}
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, {"schema_version": 1, "status": "completed", "dry_run": False,
                              "case_id": case_id, "simulation_mode": "antenna_characterization",
                              "synthetic": True, "artifacts": artifacts})
    return CaseRunResult(case_id=case_id, case_directory=directory,
                         manifest_path=manifest_path, s11_path=directory / "S11.csv",
                         farfield_source_path=directory / "Farfield Source [1].ffs",
                         dry_run=False, elapsed_seconds=0.0)


def run(args):
    source_hashes = {name: sha(ROOT / name) for name in SOURCE_FILES}
    public_report = Path(args.public_report).resolve() if args.public_report else None
    if public_report is not None and public_report.exists():
        raise ValueError("Use a new public report path; previous acceptance evidence is immutable")
    handoff = json.loads(Path(args.handoff).read_text(encoding="utf-8"))
    if handoff["mode"] != "synthetic-loopback-only":
        raise ValueError("Only the existing synthetic loopback handoff is accepted")
    sys.path.insert(0, handoff["source_root"])
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from shore_control.client import Client, MemorySigner
    from shore_control.crypto import load_public_key, public_key_text
    from shore_control.lab import load_config, load_lab_key
    from shore_control.rpc import RpcTransport
    from msabp_opt.simulation.distributed.control_authority import ControlAuthority
    from msabp_opt.simulation.distributed.control_worker import ShoreWorker
    from msabp_opt.simulation.distributed.config import DeviceRegistry
    from msabp_opt.simulation.distributed.maid import Maid, MaidRuntimeConfig
    from msabp_opt.simulation.distributed.runtime import PrincessRuntime, prepare_run

    config = load_config(handoff["existing_lab_config"])
    for name in ("fleet_id", "authority_public_key", "controllers", "workers", "deployment_id"):
        if handoff[name] != config[name]:
            raise ValueError("Lab configuration changed: " + name)
    if handoff["endpoint"] != f"http://127.0.0.1:{config['port']}":
        raise ValueError("Lab endpoint changed")
    public = load_public_key(config["authority_public_key"])
    transport = RpcTransport(handoff["endpoint"])
    def actor(name):
        expected = config["controllers"].get(name) or config["workers"][name]["public_key"]
        key = load_lab_key(config["test_keys"][name], expected)
        return Client(transport, name, MemorySigner(key), public, config["fleet_id"])
    controller, worker_client = actor("controller-a"), actor("maid-a")
    status = controller.call("status")
    if status["bound_job"] is not None:
        raise RuntimeError("Existing authority is busy; no takeover, reset or automatic close")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Only a freshly generated synthetic session key; never copies E/F keys.
    session_key = Ed25519PrivateKey.generate()
    (output / "synthetic-session.pem").write_bytes(session_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    session_id = "adapter-" + secrets.token_hex(8)
    session = Client(transport, "session:" + session_id, MemorySigner(session_key), public, config["fleet_id"])
    gate = ControlAuthority(output / "control.sqlite", controller_client=controller,
                            session_client=session, worker_registrations=config["workers"])
    # Pick a free loopback port. Failure to bind later fails closed before work.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    collector_url = f"http://127.0.0.1:{port}"
    # Shore's scope contract requires a Tailnet-form address. This synthetic
    # resource identifier is never contacted; only the loopback client below
    # transports bytes. This is NOT a live Tailnet/source-attestation test.
    scope_collector_url = f"http://100.64.0.254:{port}"
    job_id = "msabp-adapter-" + secrets.token_hex(10)
    run_id = job_id + "-batch01"
    source = output / "samples.csv"
    source.write_text("sample_id,value\n" + "".join(f"{i},{i}\n" for i in range(args.points)), encoding="utf-8")
    manifest = output / "job-manifest.json"
    policy = output / "accounting-policy.json"
    write_json(manifest, {"job_id": job_id, "approved_units": 128, "synthetic": True,
                          "initial_run": run_id, "initial_csv_sha256": sha(source)})
    write_json(policy, {"schema": "msabp.budget-policy", "schema_version": 1,
                        "unit_key": "evaluation_id", "settled_outcomes": ["completed"]})
    spec = {"schema": "shore.job-spec", "schema_version": 1, "manifest_sha256": sha(manifest),
            "approved_units": 128, "budget_unit": "objective_evaluation", "accounting_policy_sha256": sha(policy)}
    gate.acquire(job_id=job_id, host_id="jiangjiang-lab", session_id=session_id,
                 session_public_key=public_key_text(session_key.public_key()), project_id="MSABP_2",
                 collector_url=scope_collector_url, data_root=str(output), job_spec=spec)
    gate.bind_job_documents(manifest, policy)
    write_json(output / "recovery.json", {"job_id": job_id, "session_id": session_id,
               "collector_url": collector_url, "handoff": str(Path(args.handoff).resolve()),
               "binding": gate.binding, "note": "Retain this lab journal; never reset authority DB."})
    registry = DeviceRegistry(bind_host="127.0.0.1", advertise_url=collector_url, port=port, devices=())
    prepared = prepare_run(source_csv=source, run_id=run_id, registry=registry,
                            devices=(), repository_root=output)
    gate.register_run(run_id, sha(prepared.paths.runtime_json), prepared.worklist.worklist.sha256)
    gate.bind_run_plan(run_id, prepared.paths.runtime_json)
    executions = []
    def executor(row, **kwargs):
        executions.append(kwargs["case_id"])
        return synthetic_case(row, **kwargs)
    with PrincessRuntime(preparation=prepared, registry=registry, devices=(),
                          project_template=output / "never-open.cst", dry_run=True,
                          control_authority=gate) as runtime:
        runtime.start_server()
        binding = {**gate.binding, **gate.job["scope"], "fleet_id": config["fleet_id"]}
        worker = ShoreWorker(output / "worker-control", worker_client, public, binding=binding,
                              registration=config["workers"]["maid-a"], csv_sha256=prepared.worklist.worklist.sha256)
        maid_config = MaidRuntimeConfig.from_mapping({
            "schema_version": 1, "protocol_mode": "shore-fake-v1", "run_id": run_id,
            "worker_id": "maid-a", "princess_url": collector_url, "api_token": prepared.api_token,
            "csv_path": str(prepared.paths.worklist_csv), "csv_sha256": prepared.worklist.worklist.sha256,
            "project_path": str(output / "never-open.cst"), "output_root": str(output / "worker-output"),
            "dry_run": True, "heartbeat_seconds": 1, "poll_seconds": 0.1})
        code = Maid(maid_config, session=SyntheticSession(), run_csv_row=executor, control_worker=worker).run()
        if code != 0 or len(executions) != args.points or len(set(executions)) != args.points:
            raise RuntimeError("Synthetic execution count differs or Maid failed")
        gate.begin_drain("synthetic adapter acceptance completed")
        checkpoint, budget = gate.build_checkpoint()
        gate.finish_drain()
        assert budget["settled_units"] == args.points and budget["pending_units"] == 128 - args.points
        report = {"schema": "msabp.shore-adapter-lab", "schema_version": 1,
                  "observed_utc": datetime.now(timezone.utc).isoformat(), "job_id": job_id,
                  "executions": executions, "progress": runtime.state.progress(run_id).as_dict(),
                  "checkpoint": checkpoint, "budget": budget,
                  "real_cst": False, "ssh": False, "gpu": False, "phase": gate.job["phase"]}
        report["collector_transport"] = {"signed_synthetic_scope": scope_collector_url,
                                          "actual_loopback": collector_url, "tailnet_tested": False}
        if args.close_synthetic_job:
            gate.release(disposition="closed", reason="Explicitly close synthetic acceptance budget; no research budget abandoned",
                         explicit_close=True)
            report["phase"] = gate.job["phase"]
        report["source_hashes"] = source_hashes
        report["source_unchanged_during_run"] = all(sha(ROOT / name) == value for name, value in source_hashes.items())
        report["shore_source_hashes"] = {str(path.relative_to(handoff["source_root"])): sha(path)
            for path in sorted((Path(handoff["source_root"]) / "shore_control").glob("*.py"))}
        write_json(output / "report.json", report)
        if not report["source_unchanged_during_run"]:
            raise RuntimeError("Source changed during acceptance; report is retained but not a frozen baseline")
        if public_report is not None:
            public_report.parent.mkdir(parents=True, exist_ok=True)
            write_json(public_report, report)
    print(json.dumps({"report": str(output / "report.json"), "settled": args.points,
                      "pending": 128 - args.points, "phase": report["phase"]}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", default=r"D:\Shore_Protocal\artifacts\project-adapter-lab-connection-2026-10-14.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--public-report", help="Optional shareable JSON evidence; contains no private keys or API tokens")
    parser.add_argument("--points", type=int, default=32, choices=range(1, 129))
    parser.add_argument("--close-synthetic-job", action="store_true")
    args = parser.parse_args()
    old_import = builtins.__import__
    def guarded_import(name, *a, **kw):
        if name == "cst" or name.startswith("cst."):
            raise AssertionError("CST imports are forbidden in adapter acceptance")
        return old_import(name, *a, **kw)
    with patch.dict(os.environ, {"MSABP_CONTROL_MODE": "shore-fake-v1"}), \
         patch("platform.uname", return_value=platform.uname_result("Windows", "shore-lab", "10", "synthetic", "AMD64")), \
         patch("subprocess.Popen", side_effect=AssertionError("No subprocess allowed")), \
         patch("builtins.__import__", new=guarded_import):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
