"""Persist only restricted session keys with Windows CURRENT_USER DPAPI.

This is not an F identity or a device identity loader. No plaintext fallback,
key discovery, silent reset, machine-scope DPAPI or ACL modification exists.
Call only from the explicit operational entry; offline preparation uses no key.
"""
from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import secrets


MODE = "shore-remote-synthetic-v1"
SCHEMA = "msabp.shore-remote-session"
MAX_SESSION_FILE_BYTES = 64 * 1024
_FIELDS = {"schema", "schema_version", "protection", "context", "session_id", "protected_blob_b64"}
_INNER_FIELDS = {"schema", "schema_version", "context_sha256", "session_id", "private_key_b64"}


def _canonical(value) -> bytes:
    from shore_control.crypto import canonical_json

    return canonical_json(value)


def _context(value: dict) -> dict:
    if (type(value) is not dict or not {"mode", "bootstrap_sha256", "job_id"} <= set(value)
            or value["mode"] != MODE
            or type(value["bootstrap_sha256"]) is not str
            or re.fullmatch(r"[a-f0-9]{64}", value["bootstrap_sha256"]) is None
            or value["bootstrap_sha256"] == "0" * 64
            or type(value["job_id"]) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value["job_id"]) is None):
        raise ValueError("Session context requires explicit mode, bootstrap SHA256 and immutable job ID")
    data = _canonical(value)
    if len(data) > 16 * 1024:
        raise ValueError("Session context exceeds its bounded public metadata budget")
    return json.loads(data)


def _entropy(context: dict) -> bytes:
    return hashlib.sha256(b"msabp-shore-restricted-session-v1\0" + _canonical(context)).digest()


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data: bytes, entropy: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise RuntimeError("Restricted session persistence requires Windows CURRENT_USER DPAPI")
    if not data or len(data) > MAX_SESSION_FILE_BYTES or len(entropy) != 32:
        raise ValueError("Invalid bounded DPAPI input")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    source_buffer = ctypes.create_string_buffer(data)
    entropy_buffer = ctypes.create_string_buffer(entropy)
    source = _DataBlob(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    salt = _DataBlob(len(entropy), ctypes.cast(entropy_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = _DataBlob()
    if decrypt:
        function = crypt32.CryptUnprotectData
        function.argtypes = [ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.POINTER(_DataBlob),
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob)]
        second = None
    else:
        function = crypt32.CryptProtectData
        function.argtypes = [ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.POINTER(_DataBlob),
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob)]
        second = "MSABP restricted Shore session"
    function.restype = wintypes.BOOL
    # 0x1 forbids UI. Never use CRYPTPROTECT_LOCAL_MACHINE (0x4).
    if not function(ctypes.byref(source), second, ctypes.byref(salt), None, None, 0x1, ctypes.byref(output)):
        raise OSError(ctypes.get_last_error(), "Windows DPAPI refused restricted session protection")
    try:
        if not output.cbData or output.cbData > MAX_SESSION_FILE_BYTES:
            raise ValueError("DPAPI output exceeds its bounded session contract")
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(output.pbData, ctypes.c_void_p))


def _protect(data: bytes, entropy: bytes) -> bytes:
    return _dpapi(data, entropy, decrypt=False)


def _unprotect(data: bytes, entropy: bytes) -> bytes:
    return _dpapi(data, entropy, decrypt=True)


def _read(path: Path) -> dict:
    from shore_control.evidence import _open_artifact, load_evidence_json

    if not path.is_file() or path.stat().st_size > MAX_SESSION_FILE_BYTES:
        raise ValueError("Existing restricted session file is absent, invalid or oversized")
    with _open_artifact(path) as stream:
        before = os.fstat(stream.fileno())
        data = stream.read(MAX_SESSION_FILE_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (len(data) != before.st_size or len(data) > MAX_SESSION_FILE_BYTES
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError("Restricted session file changed during safe read")
    value = load_evidence_json(data)
    if _canonical(value) != data:
        raise ValueError("Restricted session file must retain its canonical protected bytes")
    return value


def load_or_create_session(path: str | Path, *, context: dict):
    """Return ``(session_id, Ed25519PrivateKey)`` bound to exact public context.

    Existing corruption, changed context or a different Windows user is a hard
    error. The file is never replaced. The caller owns its process instance lock.
    Creation writes ciphertext only; a partial write remains a fail-closed file.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from shore_control.evidence import load_evidence_json
    from .control_remote_worker import checked_path

    context = _context(context)
    path = checked_path(path)
    entropy = _entropy(context)
    if path.exists():
        outer = _read(path)
        if (set(outer) != _FIELDS or outer["schema"] != SCHEMA
                or type(outer["schema_version"]) is not int or outer["schema_version"] != 1
                or outer["protection"] != "windows-dpapi-current-user"
                or _canonical(outer["context"]) != _canonical(context)
                or type(outer["session_id"]) is not str
                or re.fullmatch(r"remote-[a-f0-9]{32}", outer["session_id"]) is None
                or type(outer["protected_blob_b64"]) is not str):
            raise ValueError("Existing restricted session context/protection differs; refusing replacement")
        blob = base64.b64decode(outer["protected_blob_b64"], validate=True)
        data = _unprotect(blob, entropy)
        inner = load_evidence_json(data)
        if (set(inner) != _INNER_FIELDS or _canonical(inner) != data
                or inner["schema"] != SCHEMA + "-secret"
                or type(inner["schema_version"]) is not int or inner["schema_version"] != 1
                or inner["context_sha256"] != hashlib.sha256(_canonical(context)).hexdigest()
                or inner["session_id"] != outer["session_id"]
                or type(inner["private_key_b64"]) is not str):
            raise ValueError("DPAPI session payload differs from its immutable public binding")
        private = base64.b64decode(inner["private_key_b64"], validate=True)
        if len(private) != 32:
            raise ValueError("Invalid restricted Ed25519 session key")
        return outer["session_id"], Ed25519PrivateKey.from_private_bytes(private)
    # Complete protection first, so unavailable DPAPI cannot leave a plaintext
    # or partially provisioned identity on disk.
    key = Ed25519PrivateKey.generate()
    session_id = "remote-" + secrets.token_hex(16)
    private = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                serialization.NoEncryption())
    inner = {"schema": SCHEMA + "-secret", "schema_version": 1,
             "context_sha256": hashlib.sha256(_canonical(context)).hexdigest(), "session_id": session_id,
             "private_key_b64": base64.b64encode(private).decode("ascii")}
    protected = _protect(_canonical(inner), entropy)
    outer = {"schema": SCHEMA, "schema_version": 1, "protection": "windows-dpapi-current-user",
             "context": context, "session_id": session_id,
             "protected_blob_b64": base64.b64encode(protected).decode("ascii")}
    encoded = _canonical(outer)
    if len(encoded) > MAX_SESSION_FILE_BYTES:
        raise ValueError("Protected restricted session exceeds its file budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    checked_path(path)
    with path.open("xb") as stream:
        if os.name == "nt":
            from shore_control.evidence import _verify_windows_handle

            _verify_windows_handle(stream.fileno(), path)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return session_id, key
