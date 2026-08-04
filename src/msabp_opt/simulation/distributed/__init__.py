"""Distributed CST simulation coordination primitives.

``PrincessState`` is the durable single-writer task ledger.  The protocol
helpers provide a dependency-free JSONL channel suitable for local subprocess
pipes and Windows OpenSSH alike.
"""

from .protocol import (
    FRAME_MAGIC,
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_frame,
    encode_frame,
    iter_frames,
    make_message,
    validate_message,
)
from .state import (
    ClaimedTask,
    EventRecord,
    FailureOutcome,
    FailureRecord,
    FrozenCsv,
    FrozenCsvRow,
    LeaseError,
    PrincessState,
    Progress,
    RestartDecision,
    RunMismatchError,
    StateError,
    StateStore,
    UnknownRunError,
    UnknownWorkerError,
    WorkerUnavailableError,
    freeze_csv,
    sha256_file,
    validate_frozen_csv,
)

__all__ = [
    "FRAME_MAGIC",
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "ClaimedTask",
    "EventRecord",
    "FailureOutcome",
    "FailureRecord",
    "FrozenCsv",
    "FrozenCsvRow",
    "LeaseError",
    "PrincessState",
    "Progress",
    "RestartDecision",
    "RunMismatchError",
    "StateError",
    "StateStore",
    "UnknownRunError",
    "UnknownWorkerError",
    "WorkerUnavailableError",
    "decode_frame",
    "encode_frame",
    "freeze_csv",
    "iter_frames",
    "make_message",
    "sha256_file",
    "validate_frozen_csv",
    "validate_message",
]
