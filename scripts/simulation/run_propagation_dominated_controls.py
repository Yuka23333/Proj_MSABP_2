"""F5/CLI launcher for the eight dominated-control propagation cases."""

from __future__ import annotations

import sys
from pathlib import Path, PureWindowsPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.simulation import princess  # noqa: E402


WORKLIST_CSV = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_dominated_controls_8.csv"
)
RUN_ID = "propagation-dominated-controls-8-001"
DEVICE_IDS = ("convallariag5", "coconutg2")
DEVICE_PROJECT_RELATIVE_PATH = PureWindowsPath(
    "simulations", "models", "msa-bp-propagation.cst"
)


def build_princess_argv() -> list[str]:
    if not WORKLIST_CSV.is_file():
        raise FileNotFoundError(
            "generate the worklist first with "
            "scripts/simulation/prepare_propagation_dominated_controls.py"
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
    print("[Propagation] 8 dominated controls from the 512 + six-round archive")
    print(
        "[Propagation] each Maid uses its own manually provisioned template: "
        f"{DEVICE_PROJECT_RELATIVE_PATH}"
    )
    print("[Propagation] transfer=S21 only; native E-fields remain Maid-local")
    return princess.main(build_princess_argv())


if __name__ == "__main__":
    raise SystemExit(main())
