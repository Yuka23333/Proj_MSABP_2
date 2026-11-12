"""Fixed isolated supervisor and Maid child entrypoint (installed use only).

No project or Shore import occurs before the fixed host revalidates local
policy, package closure, interpreter, ACLs, and original installation state.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

FIXED_SUPERVISOR = Path(r"C:\ProgramData\ShorePrincess\project-msabp\supervisor")
FIXED_POLICY = FIXED_SUPERVISOR / "policy.json"


def trusted_installation(policy_path):
    if (sys.platform != "win32" or not sys.flags.isolated or not sys.dont_write_bytecode
            or Path(policy_path) != FIXED_POLICY):
        raise RuntimeError("Installed worker requires Windows, -I -B, and the fixed policy")
    name = "_shore_fixed_supervisor_host"
    host = sys.modules.get(name)
    if host is None:
        spec = importlib.util.spec_from_file_location(name, FIXED_SUPERVISOR / "supervisor_host.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Fixed supervisor host is unavailable")
        host = importlib.util.module_from_spec(spec)
        sys.modules[name] = host
        spec.loader.exec_module(host)
    installation = host.preflight_worker_policy(FIXED_POLICY)
    expected = installation["paths"]["app"] / "scripts/simulation/shore_supervisor.py"
    if Path(__file__).absolute() != expected or Path(sys.executable) != installation["paths"]["python"]:
        raise RuntimeError("Worker entry or interpreter differs from approved installation")
    return installation


def _import_roots(installation):
    for root in (installation["paths"]["shore"], installation["paths"]["app"] / "src"):
        value = str(root)
        if value not in sys.path:
            sys.path.insert(0, value)
    # Explicit pinned Shore import: -I deliberately ignores PYTHONPATH.
    import shore_control
    expected = installation["paths"]["shore"] / "shore_control/__init__.py"
    if Path(shore_control.__file__).absolute() != expected:
        raise RuntimeError("Shore dependency was imported from an unapproved root")


def _maid(args, installation):
    from msabp_opt.simulation.distributed.control_release import runtime_paths
    from msabp_opt.simulation.distributed.control_remote_worker import (
        RemoteSyntheticMaidConfig, read_json, recover_worker_terminal,
        report_worker_idle, run_worker, verify_startup_record,
    )
    from msabp_opt.simulation.distributed.control_supervisor import MaterialJournal
    from msabp_opt.simulation.distributed.control_supervisor_runtime import RuntimeLedger
    config = RemoteSyntheticMaidConfig.load(args.maid_runtime)
    job_id = config["binding"]["job_id"]
    paths = runtime_paths(installation["profile"], job_id)
    if Path(args.maid_runtime) != Path(paths["runtime_config_path"]):
        raise ValueError("Child requires the original fixed committed runtime")
    ledger = RuntimeLedger(installation["paths"]["state"], profile=installation["profile"])
    row = ledger.job(job_id)
    if row["stage"] != "SESSION" or json.loads(row["binding"]) != config["binding"]:
        raise ValueError("Child cannot run outside its original active lifecycle")
    material = MaterialJournal(ledger.root / "jobs" / job_id / "material",
        profile=installation["profile"], bootstrap_data=installation["bootstrap_data"]).original(job_id)
    if material.record["runtime_sha256"] != config.runtime_sha256:
        raise ValueError("Child runtime differs from original committed material")
    if args.report_idle:
        return report_worker_idle(config)
    if args.recover_terminal:
        return recover_worker_terminal(config)
    startup, _ = read_json(args.wake_startup)
    verified = verify_startup_record(startup, config, installation["bootstrap"])
    expected_startup = Path(paths["bell_state_root"]) / "outbox" / material.record["permit_id"] / "startup.json"
    expected_ready = config.state_root / ("ready." + verified["launch_id"] + ".json")
    if Path(args.wake_startup) != expected_startup or Path(args.ready_record) != expected_ready:
        raise ValueError("Child launch paths differ from original signed wake")
    return run_worker(config, startup_record=startup, ready_record=args.ready_record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--maid-runtime")
    parser.add_argument("--wake-startup")
    parser.add_argument("--ready-record")
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("run", "maid-run", "recover-terminal", "report-idle"):
        mode.add_argument("--" + flag, action="store_true")
    args = parser.parse_args(argv)
    if args.run and any((args.maid_runtime, args.wake_startup, args.ready_record)):
        parser.error("Supervisor accepts no caller-supplied runtime or launch paths")
    if not args.run and not args.maid_runtime:
        parser.error("Maid mode requires its fixed original runtime")
    if args.maid_run != bool(args.wake_startup and args.ready_record):
        parser.error("Only a new Maid launch requires both signed startup and ready paths")
    if not args.maid_run and (args.wake_startup or args.ready_record):
        parser.error("Recovery cannot accept startup or ready paths")
    try:
        installation = trusted_installation(args.policy)
        _import_roots(installation)
        if not args.run:
            return _maid(args, installation)
        from msabp_opt.simulation.distributed.control_supervisor_runtime import SupervisorRuntime
        SupervisorRuntime(installation,
            refresh_installation=lambda: trusted_installation(args.policy)).run()
        return 0
    except Exception as exc:
        print(json.dumps({"state": "MAINTENANCE", "error": type(exc).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
