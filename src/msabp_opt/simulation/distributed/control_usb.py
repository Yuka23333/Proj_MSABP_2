"""Discover the enrolled Princess USB once, then pin its path and volume.

Only public manifests are scanned. Private signing remains Shore's UsbIdentity,
with a fresh key read on every signature. A volume GUID is a local continuity
guard, not hardware authentication or protection against copied USB secrets.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path, PureWindowsPath
import re
import threading

from .control_failure import _persist_record
from .control_remote_worker import _json_bytes, checked_path, read_json


SCHEMA = "msabp.princess-usb-binding"
RELATIVE_MANIFEST = PureWindowsPath("PrincessControl/identity.json")
_VOLUME = re.compile(r"\\\\\?\\Volume\{[0-9a-fA-F-]{36}\}\\", re.IGNORECASE)


def _manifest_path(value):
    path = PureWindowsPath(value)
    if (not path.is_absolute() or not re.fullmatch(r"[A-Za-z]:", path.drive)
            or path.parts[1:] != RELATIVE_MANIFEST.parts):
        raise ValueError("Expected X:\\PrincessControl\\identity.json on a local removable drive")
    return Path(str(path))


def removable_roots():
    """Enumerate only Windows drive-letter roots, never arbitrary directories."""
    if os.name != "nt":
        raise ValueError("Princess USB discovery requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetLogicalDrives.argtypes = []
    kernel.GetLogicalDrives.restype = ctypes.c_uint32
    kernel.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
    kernel.GetDriveTypeW.restype = ctypes.c_uint
    mask = kernel.GetLogicalDrives()
    if not mask:
        raise OSError(ctypes.get_last_error(), "Cannot enumerate local drives")
    return [f"{chr(65 + index)}:\\" for index in range(26)
            if mask & (1 << index) and kernel.GetDriveTypeW(f"{chr(65 + index)}:\\") == 2]


def volume_guid(manifest_path):
    from shore_control.usb_identity import _assert_removable

    path = _manifest_path(manifest_path)
    _assert_removable(path)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    call = kernel.GetVolumeNameForVolumeMountPointW
    call.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    buffer = ctypes.create_unicode_buffer(128)
    if not call(path.anchor, buffer, len(buffer)):
        raise OSError(ctypes.get_last_error(), "Cannot identify Princess USB volume")
    if not _VOLUME.fullmatch(buffer.value):
        raise ValueError("Invalid Windows volume identity")
    return buffer.value.lower()


def _public_identity(manifest_path, actor, expected_fingerprint):
    from shore_control.crypto import fingerprint
    from shore_control.usb_identity import (
        _assert_regular_path, _assert_removable, _read_bytes, _unique_object,
    )

    path = _manifest_path(manifest_path)
    _assert_removable(path)
    _assert_regular_path(path, path.parent)
    value = json.loads(_read_bytes(path, 65536).decode("utf-8-sig"),
                       object_pairs_hook=_unique_object)
    if (type(value) is not dict or value.get("Role") != "Princess"
            or not isinstance(value.get("IdentityId"), str)
            or "princess-" + value["IdentityId"] != actor
            or value.get("Fingerprint") != expected_fingerprint
            or fingerprint(value.get("PublicKey")) != expected_fingerprint):
        raise ValueError("USB public identity does not match the enrolled Princess")


def discover_usb(*, actor, expected_fingerprint, selection="auto"):
    paths = ([Path(root) / str(RELATIVE_MANIFEST) for root in removable_roots()]
             if selection == "auto" else [_manifest_path(selection)])
    matches = []
    for path in paths:
        try:
            before = volume_guid(path)
            _public_identity(path, actor, expected_fingerprint)
            if volume_guid(path) != before:
                raise ValueError("Princess USB changed during discovery")
        except (OSError, ValueError):
            if selection != "auto":
                raise
            continue
        matches.append({"manifest_path": str(path), "volume_guid": before})
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one enrolled Princess USB; found {len(matches)}. No fallback used.")
    return matches[0]


class LockedUsbIdentity:
    """Lazy signer: absent F must not prevent session-only drain/collection.

    Persist only public metadata. Restart uses the original binding even with
    F absent. No automatic relocation, rediscovery or rebinding of an old job.
    """

    def __init__(self, *, actor, expected_fingerprint, selection="auto",
                 binding_path=None, context=None, allow_new_binding=True):
        self.actor = actor
        self.expected_fingerprint = expected_fingerprint
        self.selection = selection
        if selection != "auto":
            _manifest_path(selection)
        self.binding_path = checked_path(binding_path) if binding_path is not None else None
        self.context = dict(context or {})
        self.allow_new_binding = allow_new_binding
        self._record = None
        self._lock = threading.RLock()

    def _resolve(self):
        expected = {"schema": SCHEMA, "schema_version": 1, "actor": self.actor,
                    "fingerprint": self.expected_fingerprint, "context": self.context}
        if self.binding_path is not None and self.binding_path.exists():
            record, _ = read_json(self.binding_path)
            if (set(record) != set(expected) | {"manifest_path", "volume_guid"}
                    or type(record["schema_version"]) is not int
                    or any(record[key] != value for key, value in expected.items())
                    or type(record["volume_guid"]) is not str
                    or not _VOLUME.fullmatch(record["volume_guid"])):
                raise ValueError("Original Princess USB binding is invalid; refusing replacement")
            _manifest_path(record["manifest_path"])
        elif self._record is not None:
            if self.binding_path is not None:
                raise ValueError("Original Princess USB binding is missing; refusing replacement")
            record = self._record
        else:
            if not self.allow_new_binding:
                raise ValueError("Existing job has no Princess USB binding; refusing rediscovery")
            record = {**expected, **discover_usb(actor=self.actor,
                expected_fingerprint=self.expected_fingerprint, selection=self.selection)}
            if self.binding_path is not None:
                _persist_record(self.binding_path, _json_bytes(record))
        if self.selection != "auto" and _manifest_path(self.selection) != _manifest_path(record["manifest_path"]):
            raise ValueError("Explicit USB path differs from the original job binding")
        if self._record is not None and record != self._record:
            raise ValueError("Original Princess USB binding changed")
        self._record = record
        return record

    def _assert_present(self, record):
        if volume_guid(record["manifest_path"]) != record["volume_guid"]:
            raise ValueError("Princess USB volume changed; no automatic rebinding")
        _public_identity(record["manifest_path"], self.actor, self.expected_fingerprint)
        if volume_guid(record["manifest_path"]) != record["volume_guid"]:
            raise ValueError("Princess USB volume changed during verification")

    def inspect(self):
        """Public-only discovery/check; no private key, signing or networking."""
        with self._lock:
            record = self._resolve()
            self._assert_present(record)
            return json.loads(_json_bytes(record))

    def sign(self, domain, payload):
        from shore_control.usb_identity import UsbIdentity

        with self._lock:
            record = self._resolve()
            self._assert_present(record)
            envelope = UsbIdentity(record["manifest_path"], self.expected_fingerprint).sign(domain, payload)
            self._assert_present(record)
            return envelope


def controller_client(bootstrap, *, actor, local_node_id, signer):
    """Same pinned Shore transport, with a project-side USB-presence wrapper."""
    from shore_control.client import Client
    from shore_control.remote import DEFAULT_TAILSCALE
    from shore_control.tailnet import TailnetVerifier
    from shore_control.tailnet_rpc import TailnetRpcTransport

    if actor not in bootstrap["controllers"] or local_node_id not in bootstrap["controller_nodes"]:
        raise ValueError("Controller actor/node is not enrolled")
    network = {**bootstrap["network"], "local_node_id": local_node_id,
               "local_address": bootstrap["network"]["nodes"][local_node_id]["address"]}
    verifier = TailnetVerifier(network, DEFAULT_TAILSCALE)
    transport = TailnetRpcTransport(bootstrap["endpoint"], bootstrap["tls_sha256"], verifier,
                                   bootstrap["network"]["local_node_id"], timeout=5.0)
    return Client(transport, actor, signer, bootstrap["authority_public_key"], bootstrap["fleet_id"])
