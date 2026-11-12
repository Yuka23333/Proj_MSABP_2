"""Independent signed remote synthetic Maid. Preflight is strictly offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from msabp_opt.simulation.distributed.control_remote_worker import (  # noqa: E402
    RemoteSyntheticMaidConfig, preflight_worker, read_json, recover_worker_terminal,
    report_worker_idle, run_worker,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--wake-startup")
    parser.add_argument("--ready-record")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--recover-terminal", action="store_true",
                      help="LIVE recovery: settle original durable terminal outboxes, never execute or claim work")
    mode.add_argument("--report-idle", action="store_true",
                      help="LIVE recovery: resend an existing signed idle record, never relaunch")
    args = parser.parse_args(argv)
    if not args.run and (args.wake_startup or args.ready_record):
        parser.error("Preflight/recovery does not accept launch or readiness paths")
    if args.run and (not args.wake_startup or not args.ready_record):
        parser.error("Run requires explicit signed wake startup and readiness paths")
    try:
        config = RemoteSyntheticMaidConfig.load(args.runtime_config)
        if args.preflight:
            report = preflight_worker(config)
            report.pop("bootstrap")
            print(json.dumps(report, sort_keys=True), flush=True)
            return 0
        if args.report_idle:
            return report_worker_idle(config)
        if args.recover_terminal:
            return recover_worker_terminal(config)
        startup, _ = read_json(args.wake_startup)
        return run_worker(config, startup_record=startup, ready_record=args.ready_record)
    except (ValueError, OSError, RuntimeError) as exc:
        # Never dump runtime tokens, key paths, signed requests or stack locals.
        print(json.dumps({"error": type(exc).__name__, "message": "Synthetic Maid refused startup or execution"}),
              flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
