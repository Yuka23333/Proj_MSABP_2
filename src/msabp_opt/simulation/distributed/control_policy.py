"""Fail-closed execution boundary for the explicitly selected Shore fake lab.

Legacy workflows are not silently enrolled. This switch must be set before
starting an experimental adapter process; it is NOT production authorization.
No key discovery, SSH-agent fallback or real solver is available in this mode.
"""
from __future__ import annotations

import os


LAB_MODE = "shore-fake-v1"
REMOTE_MODE = "shore-remote-synthetic-v1"
CONTROL_MODES = frozenset((LAB_MODE, REMOTE_MODE))
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
    if controlled_mode():
        label = "fake-worker lab" if lab_mode() else REMOTE_MODE
        raise RuntimeError(
            f"Shore {label} refuses {action}; this legacy entry is not enrolled. "
            "No local/SSH/Bell/GPU or credential fallback is allowed."
        )
