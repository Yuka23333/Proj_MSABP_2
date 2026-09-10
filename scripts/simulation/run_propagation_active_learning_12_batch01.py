"""F5/CLI launcher for propagation active-learning batch 01."""

from __future__ import annotations

import sys
from pathlib import Path, PureWindowsPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.simulation import princess  # noqa: E402


WORKLIST_CSV = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_active_learning_12_batch01.csv"
)
RUN_ID = "propagation-active-learning-12-batch01-001"
DEVICE_IDS = ("convallariag5", "coconutg2")
DEVICE_PROJECT_RELATIVE_PATH = PureWindowsPath(
    "simulations",
    "models",
    "msa-bp-propagation.cst",
)


def build_princess_argv() -> list[str]:
    if not WORKLIST_CSV.is_file():
        raise FileNotFoundError(
            "generate the worklist first with "
            "scripts/simulation/prepare_propagation_active_learning_12_batch01.py"
        )
    argv = [
        "start",
        "--csv",
        str(WORKLIST_CSV),
        "--run-id",
        RUN_ID,
        "--devices-config",
        str(princess.DEFAULT_DEVICE_CONFIG_PATH),
        "--device-project-relative-path",
        str(DEVICE_PROJECT_RELATIVE_PATH),
    ]
    for device_id in DEVICE_IDS:
        argv.extend(("--device", device_id))
    return argv


def main() -> int:
    print("[Propagation AL-01] 8 calibration + 4 exploitation cases")
    print(
        "[Propagation AL-01] template: "
        f"{DEVICE_PROJECT_RELATIVE_PATH}; transfer=S21 only"
    )
    return princess.main(build_princess_argv())


if __name__ == "__main__":
    raise SystemExit(main())
