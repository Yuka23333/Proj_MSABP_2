"""Configure, diagnose, or run the persistent Maid Bell in foreground mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed.bell import (  # noqa: E402
    DEFAULT_BELL_PORT,
    MaidBellClient,
    MaidBellConfig,
    MaidBellController,
    MaidBellServer,
    default_bell_config_path,
    write_bell_config,
)


DEFAULT_MAID_PYTHON = Path(
    r"C:\Users\telecom\miniforge3\envs\maid\python.exe"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the persistent device-local Maid Bell."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser(
        "init-config",
        help="write the machine-local Bell configuration",
    )
    init.add_argument("--device-id", required=True)
    init.add_argument("--listen-host", required=True)
    init.add_argument("--port", type=int, default=DEFAULT_BELL_PORT)
    init.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    init.add_argument("--python-path", type=Path, default=DEFAULT_MAID_PYTHON)
    init.add_argument(
        "--maid-entrypoint",
        type=Path,
        default=REPOSITORY_ROOT / "scripts" / "simulation" / "maid.py",
    )
    init.add_argument("--output", type=Path, default=default_bell_config_path())

    for name in ("serve", "doctor"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--config",
            type=Path,
            default=default_bell_config_path(),
        )

    ping = subparsers.add_parser("ping", help="query a running Bell")
    ping.add_argument("--host", required=True)
    ping.add_argument("--port", type=int, default=DEFAULT_BELL_PORT)
    ping.add_argument("--timeout", type=float, default=10.0)
    return parser


def _init_config(args: argparse.Namespace) -> int:
    config = MaidBellConfig.from_mapping(
        {
            "schema_version": 1,
            "device_id": args.device_id,
            "listen_host": args.listen_host,
            "port": args.port,
            "repo_root": str(args.repo_root),
            "python_path": str(args.python_path),
            "maid_entrypoint": str(args.maid_entrypoint),
        }
    )
    destination = write_bell_config(args.output, config)
    print(destination)
    return 0


def _doctor(config_path: Path) -> int:
    config = MaidBellConfig.load(config_path)
    controller = MaidBellController(config)
    report = {
        "device_id": config.device_id,
        "listen_host": config.listen_host,
        "port": config.port,
        "repo_root": str(config.repo_root),
        "repo_exists": config.repo_root.is_dir(),
        "python_path": str(config.python_path),
        "python_exists": config.python_path.is_file(),
        "maid_entrypoint": str(config.maid_entrypoint),
        "maid_entrypoint_exists": config.maid_entrypoint.is_file(),
        "state": controller.status(),
    }
    report["ok"] = all(
        report[name]
        for name in ("repo_exists", "python_exists", "maid_entrypoint_exists")
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-config":
            return _init_config(args)
        if args.command == "doctor":
            return _doctor(args.config)
        if args.command == "ping":
            response = MaidBellClient(
                args.host,
                args.port,
                timeout=args.timeout,
            ).ping()
            print(json.dumps(response, ensure_ascii=False, indent=2))
            return 0
        if args.command == "serve":
            config = MaidBellConfig.load(args.config)
            server = MaidBellServer(config)
            print(
                f"[Maid Bell:{config.device_id}] listening on "
                f"{server.server_address[0]}:{server.server_address[1]}",
                flush=True,
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                return 130
            finally:
                server.shutdown()
            return 0
        raise AssertionError(f"unknown command: {args.command}")
    except Exception as exc:
        print(f"Maid Bell error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
