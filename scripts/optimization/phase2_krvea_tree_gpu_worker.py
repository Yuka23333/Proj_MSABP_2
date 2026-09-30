"""Execute a hash-addressed arbitrary-tree K-RVEA request (never a CST solve)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from msabp_opt.optimization import phase2_krvea_tree_relay as relay  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="Check the request without GP fitting")
    args = parser.parse_args(argv)
    if args.validate_only:
        import json

        relay.validate_request_payload(json.loads(args.request.read_text(encoding="utf-8-sig")))
        print("[Tree K-RVEA GPU] request validation passed; no fit performed", flush=True)
    else:
        reused = relay.execute_request_file(args.request, args.response)
        print(f"[Tree K-RVEA GPU] {'reused' if reused else 'completed'}: {args.response.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
