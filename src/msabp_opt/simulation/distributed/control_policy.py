"""Fail-closed execution boundary for explicitly selected Shore workloads.

Legacy workflows are not silently enrolled. This switch must be set before
starting an adapter process; it is NOT production authorization. Each selected
mode still requires its own signed runtime, source closure and execution gate.
"""
from __future__ import annotations

import os


LAB_MODE = "shore-fake-v1"
REMOTE_MODE = "shore-remote-synthetic-v1"
CST_MODE = "shore-remote-cst-v1"
CONTROL_MODES = frozenset((LAB_MODE, REMOTE_MODE, CST_MODE))
MODE_ENV = "MSABP_CONTROL_MODE"


def control_mode() -> str:
    mode = os.environ.get(MODE_ENV, "")
    if mode and mode not in CONTROL_MODES:
        raise RuntimeError(f"Unsupported {MODE_ENV}: {mode!r}")
    return mode


def lab_mode() -> bool:
    return control_mode() == LAB_MODE


def controlled_mode() -> bool:
    return bool(control_mode())


def reject_external_execution(action: str) -> None:
    mode = control_mode()
    if mode:
        label = "fake-worker lab" if mode == LAB_MODE else mode
        raise RuntimeError(
            f"Shore {label} refuses {action}; this legacy entry is not enrolled. "
            "No local/SSH/Bell/GPU or credential fallback is allowed."
        )
