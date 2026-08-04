"""Transport-neutral messages shared by Princess and Maid.

HTTP uses the validated JSON envelope directly.  The optional JSONL codec adds
a marker for local subprocess pipes or diagnostic SSH sessions, where banners
and unrelated process output may otherwise be mistaken for protocol traffic.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


PROTOCOL_VERSION = 1
FRAME_MAGIC = "MSABP/1\t"
MAX_FRAME_BYTES = 1024 * 1024


class ProtocolError(ValueError):
    """Raised when a Princess/Maid protocol frame is malformed."""


def utc_now_text() -> str:
    """Return an RFC 3339-compatible UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def make_message(
    message_type: str,
    *,
    run_id: str,
    worker_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
    message_id: str | None = None,
    sent_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a versioned protocol message with stable envelope fields."""

    if not isinstance(message_type, str) or not message_type.strip():
        raise ProtocolError("message_type must be a non-empty string")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ProtocolError("run_id must be a non-empty string")
    if worker_id is not None and (
        not isinstance(worker_id, str) or not worker_id.strip()
    ):
        raise ProtocolError("worker_id must be a non-empty string when provided")
    if payload is not None and not isinstance(payload, Mapping):
        raise ProtocolError("payload must be a mapping")

    message: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "type": message_type.strip(),
        "message_id": message_id or uuid4().hex,
        "run_id": run_id.strip(),
        "sent_at_utc": sent_at_utc or utc_now_text(),
        "payload": dict(payload or {}),
    }
    if worker_id is not None:
        message["worker_id"] = worker_id.strip()
    return message


def validate_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a decoded protocol message."""

    if not isinstance(message, Mapping):
        raise ProtocolError("protocol payload must be a JSON object")

    normalized = dict(message)
    version = normalized.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol_version {version!r}; expected {PROTOCOL_VERSION}"
        )

    for field in ("type", "message_id", "run_id", "sent_at_utc"):
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ProtocolError(f"{field} must be a non-empty string")

    worker_id = normalized.get("worker_id")
    if worker_id is not None and (
        not isinstance(worker_id, str) or not worker_id.strip()
    ):
        raise ProtocolError("worker_id must be a non-empty string when provided")

    payload = normalized.get("payload")
    if not isinstance(payload, Mapping):
        raise ProtocolError("payload must be a JSON object")
    normalized["payload"] = dict(payload)
    return normalized


def encode_frame(message: Mapping[str, Any]) -> str:
    """Encode one message as a magic-prefixed compact JSON line."""

    normalized = validate_message(message)
    try:
        body = json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"message is not JSON serializable: {exc}") from exc

    frame = f"{FRAME_MAGIC}{body}\n"
    if len(frame.encode("utf-8")) > MAX_FRAME_BYTES:
        raise ProtocolError(
            f"encoded frame exceeds {MAX_FRAME_BYTES} byte safety limit"
        )
    return frame


def decode_frame(frame: str | bytes) -> dict[str, Any]:
    """Decode and validate one complete protocol frame."""

    if isinstance(frame, bytes):
        if len(frame) > MAX_FRAME_BYTES:
            raise ProtocolError(
                f"frame exceeds {MAX_FRAME_BYTES} byte safety limit"
            )
        try:
            text = frame.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("frame is not valid UTF-8") from exc
    elif isinstance(frame, str):
        text = frame
        if len(text.encode("utf-8")) > MAX_FRAME_BYTES:
            raise ProtocolError(
                f"frame exceeds {MAX_FRAME_BYTES} byte safety limit"
            )
    else:
        raise ProtocolError("frame must be str or bytes")

    text = text.rstrip("\r\n")
    if not text.startswith(FRAME_MAGIC):
        raise ProtocolError("line does not start with the Princess/Maid frame marker")
    body = text[len(FRAME_MAGIC) :]
    if not body:
        raise ProtocolError("protocol frame has an empty JSON body")

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON protocol frame: {exc.msg}") from exc
    return validate_message(decoded)


def iter_frames(
    lines: Iterable[str | bytes],
    *,
    ignore_non_protocol: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield decoded frames from a text stream.

    ``ignore_non_protocol`` defaults to true because Windows OpenSSH and remote
    shells may emit a banner before Maid starts.  A line that carries the magic
    marker is always decoded strictly; malformed protocol traffic is never
    silently discarded.
    """

    magic_bytes = FRAME_MAGIC.encode("ascii")
    for line in lines:
        is_protocol = (
            line.startswith(magic_bytes)
            if isinstance(line, bytes)
            else isinstance(line, str) and line.startswith(FRAME_MAGIC)
        )
        if not is_protocol and ignore_non_protocol:
            continue
        yield decode_frame(line)
