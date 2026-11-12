"""F-only task dispatch to an ALREADY installed and reviewed Shore supervisor.

prepare/preflight are offline. bind/serve require --live-approved and the
original F gate; this entry never uploads code or opens SSH. No CST/GPU mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shore_synthetic_princess import main  # noqa: E402


if __name__ == "__main__":
    try:
        raise SystemExit(main(supervisor=True))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"Supervisor Princess refused operation: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(2)
