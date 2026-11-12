"""Outbound-only synthetic Bell; installation and execution are separate acts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from msabp_opt.simulation.distributed.control_remote_bell import MODE, preflight_bell, run_bell  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            _config, runtime, _bootstrap = preflight_bell(args.config)
            print(json.dumps({"schema": "msabp.shore-remote-synthetic-bell-preflight", "schema_version": 1,
                              "mode": MODE, "runtime_sha256": runtime.runtime_sha256,
                              "network_accessed": False, "private_keys_read": False,
                              "synthetic": True, "cst": False, "gpu": False}, sort_keys=True), flush=True)
            return 0
        run_bell(args.config)
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": "Synthetic Bell refused configuration or startup"}),
              flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
