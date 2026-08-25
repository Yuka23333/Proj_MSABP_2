"""One-click emergency dismissal for every currently running Maid Bell.

Press F5 in an IDE or run this file without arguments. Active leases belonging
to the exact runs reported by the Bells are returned to pending without
consuming an attempt, then each Maid/CST process tree is stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed.config import (  # noqa: E402
    DEFAULT_DEVICE_CONFIG_PATH,
    load_device_registry,
)
from msabp_opt.simulation.distributed.off_duty import (  # noqa: E402
    DEFAULT_OFF_DUTY_REASON,
    dismiss_all_maids,
)


DEFAULT_RUNS_ROOT = REPOSITORY_ROOT / "simulations" / "runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dismiss every currently running Maid Bell, stop its CST process "
            "tree, and return its active task to pending."
        )
    )
    parser.add_argument(
        "--devices-config",
        type=Path,
        default=DEFAULT_DEVICE_CONFIG_PATH,
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--device",
        dest="device_ids",
        action="append",
        default=[],
        help="Maid id; repeat to limit dismissal (default: every Bell device)",
    )
    parser.add_argument("--reason", default=DEFAULT_OFF_DUTY_REASON)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = dismiss_all_maids(
        load_device_registry(args.devices_config),
        runs_root=args.runs_root,
        device_ids=tuple(args.device_ids),
        reason=args.reason,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), flush=True)
    if report.ok:
        print(
            "[Princess] All Maids are off duty. Released cases remain pending "
            "and their attempt budgets were refunded.",
            flush=True,
        )
        return 0
    print(
        "[Princess] Off-duty request completed with errors; inspect the report above.",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
