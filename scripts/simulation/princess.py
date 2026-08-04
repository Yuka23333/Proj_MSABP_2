"""Thin command-line entrypoint for distributed CST Princess control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed.config import (  # noqa: E402
    DEFAULT_DEVICE_CONFIG_PATH,
    load_device_registry,
)
from msabp_opt.simulation.distributed.runtime import (  # noqa: E402
    PrincessRuntime,
    PrincessRuntimeError,
    copy_registry_with_device,
    default_run_id,
    load_status,
    prepare_run,
    select_devices,
)
from msabp_opt.simulation.distributed.transport import doctor_device  # noqa: E402


DEFAULT_PROJECT = REPOSITORY_ROOT / "simulations" / "models" / "msa-bp.cst"


def _add_registry_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--devices-config",
        type=Path,
        default=DEFAULT_DEVICE_CONFIG_PATH,
        help="Princess/Maid device registry JSON",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distribute and supervise CST simulations with Princess/Maid."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="prepare and execute one CSV run")
    _add_registry_argument(start)
    start.add_argument("--csv", required=True, type=Path, help="sampler CSV")
    start.add_argument("--run-id", default=None, help="stable run identifier")
    start.add_argument(
        "--device",
        action="append",
        dest="device_ids",
        help="device id to use; repeat for multiple devices",
    )
    start.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    start.add_argument("--dry-run", action="store_true")
    start.add_argument("--allow-disconnected-conductor", action="store_true")
    start.add_argument("--coordinate-quantum-mm", type=float, default=0.01)
    start.add_argument("--command-timeout-seconds", type=float, default=15.0)
    start.add_argument("--heartbeat-seconds", type=float, default=15.0)
    start.add_argument("--poll-seconds", type=float, default=5.0)
    start.add_argument("--lease-seconds", type=float, default=90.0)
    start.add_argument("--max-consecutive-errors", type=int, default=5)
    start.add_argument("--max-attempts", type=int, default=3)
    start.add_argument("--max-recovery-launch-attempts", type=int, default=3)
    start.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    start.add_argument("--stale-idle-seconds", type=float, default=30.0)
    start.add_argument("--artifact-timeout-seconds", type=float, default=600.0)
    start.add_argument(
        "--artifact-commit-deadline-seconds",
        type=float,
        default=1800.0,
        help="total retry deadline shared by artifact upload and completion commit",
    )
    start.add_argument(
        "--resume-grace-seconds",
        type=float,
        default=None,
        help="wait for an existing Maid to refresh last_seen before relaunch",
    )
    start.add_argument("--save-project-after-case", action="store_true")

    status = subparsers.add_parser("status", help="read durable run progress")
    status.add_argument("--run-id", required=True)

    doctor = subparsers.add_parser("doctor", help="check Maid hosts read-only")
    _add_registry_argument(doctor)
    doctor.add_argument(
        "--device",
        action="append",
        dest="device_ids",
        help="device id to check; defaults to enabled devices",
    )

    add = subparsers.add_parser("add-device", help="append a validated Maid device")
    _add_registry_argument(add)
    add.add_argument("--id", required=True)
    add.add_argument(
        "--launch-mode",
        choices=("bell", "ssh_process", "scheduled_task", "local"),
        default="bell",
    )
    add.add_argument("--ssh-target")
    add.add_argument("--repo-root", required=True)
    add.add_argument("--python-path", required=True)
    add.add_argument("--scheduled-task-name")
    add.add_argument("--runtime-config-path")
    add.add_argument("--ssh-port", type=int, default=22)
    add.add_argument("--ssh-connect-timeout-seconds", type=float, default=10.0)
    add.add_argument("--identity-file")
    add.add_argument("--bell-host")
    add.add_argument("--bell-port", type=int, default=8766)
    add.add_argument("--bell-connect-timeout-seconds", type=float, default=10.0)
    add.add_argument("--enabled", action="store_true")
    return parser


def _run_start(args: argparse.Namespace) -> int:
    registry = load_device_registry(args.devices_config)
    devices = select_devices(registry, args.device_ids)
    run_id = args.run_id or default_run_id()
    preparation = prepare_run(
        source_csv=args.csv,
        run_id=run_id,
        registry=registry,
        devices=devices,
        repository_root=REPOSITORY_ROOT,
    )
    print(
        f"[Princess] run={run_id} valid={preparation.worklist.worklist.row_count} "
        f"excluded={len(preparation.worklist.excluded_rows)} "
        f"devices={','.join(device.id for device in devices)}",
        flush=True,
    )
    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=devices,
        project_template=args.project,
        dry_run=args.dry_run,
        coordinate_quantum_mm=args.coordinate_quantum_mm,
        allow_disconnected_conductor=args.allow_disconnected_conductor,
        command_timeout_seconds=args.command_timeout_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        poll_seconds=args.poll_seconds,
        lease_seconds=args.lease_seconds,
        max_consecutive_errors=args.max_consecutive_errors,
        max_attempts=args.max_attempts,
        save_project_after_case=args.save_project_after_case,
        startup_timeout_seconds=args.startup_timeout_seconds,
        stale_idle_seconds=args.stale_idle_seconds,
        artifact_timeout_seconds=args.artifact_timeout_seconds,
        artifact_commit_deadline_seconds=args.artifact_commit_deadline_seconds,
        max_recovery_launch_attempts=args.max_recovery_launch_attempts,
    )
    try:
        runtime.start_server()
        deployments = runtime.start_workers(
            resume_grace_seconds=args.resume_grace_seconds,
        )
        for item in deployments:
            print(
                f"[Princess] Maid {item.device.id} awake, pid={item.launch.pid}, "
                f"stdout={item.launch.stdout_path}",
                flush=True,
            )
        result = runtime.monitor()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if int(result["failed"]) == 0 else 2
    except KeyboardInterrupt:
        print("[Princess] interrupted; durable state is kept for inspection", flush=True)
        return 130
    except PrincessRuntimeError as exc:
        print(f"[Princess] stopped: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        runtime.close()


def _run_doctor(args: argparse.Namespace) -> int:
    registry = load_device_registry(args.devices_config)
    devices = select_devices(registry, args.device_ids)
    failed = False
    for device in devices:
        report = doctor_device(device)
        payload = {
            **report.__dict__,
            "launch_mode": device.launch_mode.value,
            "ok": report.ok,
            "missing_requirements": report.missing_requirements,
            "modules": [item.__dict__ for item in report.modules],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        failed = failed or not report.ok
    return int(failed)


def _run_add_device(args: argparse.Namespace) -> int:
    mapping: dict[str, Any] = {
        "id": args.id,
        "enabled": args.enabled,
        "launch_mode": args.launch_mode,
        "ssh_target": args.ssh_target,
        "repo_root": args.repo_root,
        "python_path": args.python_path,
        "scheduled_task_name": args.scheduled_task_name,
        "runtime_config_path": args.runtime_config_path,
        "ssh_port": args.ssh_port,
        "ssh_connect_timeout_seconds": args.ssh_connect_timeout_seconds,
        "identity_file": args.identity_file,
        "bell_host": args.bell_host,
        "bell_port": args.bell_port,
        "bell_connect_timeout_seconds": args.bell_connect_timeout_seconds,
    }
    registry = copy_registry_with_device(args.devices_config, mapping)
    print(f"added {args.id}; registry now has {len(registry.devices)} devices")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "start":
            return _run_start(args)
        if args.command == "status":
            print(
                json.dumps(
                    load_status(args.run_id, repository_root=REPOSITORY_ROOT),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "doctor":
            return _run_doctor(args)
        if args.command == "add-device":
            return _run_add_device(args)
        raise AssertionError(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Princess error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
