"""Run the legacy SSH/SCP transport through the Shore management USB key.

The legacy Princess transport builds OpenSSH ``ssh``/``scp`` command lines and
relies on a private key in the user profile.  That key was revoked on the
managed hosts on 2026-10-06; they now accept only the key on the Shore
management USB (``<drive>:\\ShoreControl``).  The USB is FAT32, so Windows
OpenSSH refuses its key file, and the key must not be copied off the drive.

With ``MSABP_SSH_BACKEND=shore-usb`` this module rewrites each OpenSSH command
line into the equivalent Plink/PSCP command line.  Plink and PSCP come from the
USB, read the PPK directly from the USB for every connection, and pin the host
key recorded in the USB's ``nodes\\<host>.json``.  Nothing is cached, copied or
handed to an agent, so removing the USB stops new connections.

Without that environment variable nothing changes.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

BACKEND_ENV = "MSABP_SSH_BACKEND"
USB_ROOT_ENV = "MSABP_SHORE_USB_ROOT"
BACKEND_NAME = "shore-usb"

_DRIVE_REMOVABLE = 2
_REMOTE_SPEC_RE = re.compile(r"^(?P<user>[^@:/\\]+)@(?P<host>[^@:/\\]+):(?P<path>.+)$")
_TARGET_RE = re.compile(r"^(?P<user>[^@:/\\\s]+)@(?P<host>[^@:/\\\s]+)$")
_HOST_KEY_RE = re.compile(r"^ssh-ed25519 (?P<blob>[A-Za-z0-9+/]+={0,2})(?: [^\r\n]*)?$")
_ADDRESS_RE = re.compile(r"^100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}$")
_USER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ShoreUsbError(OSError):
    """The USB backend is selected but cannot build a safe command line."""


@dataclass(frozen=True)
class ShoreNode:
    host_name: str
    user_name: str
    address: str
    dns_name: str
    host_key_fingerprint: str
    port: int


def backend_enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(BACKEND_ENV, "")
    value = value.strip().lower()
    if value in ("", "openssh"):
        return False
    if value == BACKEND_NAME:
        return True
    raise ShoreUsbError(
        f"{BACKEND_ENV} must be 'openssh' or '{BACKEND_NAME}', not {value!r}"
    )


def host_key_fingerprint(host_key: str) -> str:
    """Return the ``SHA256:...`` fingerprint Plink expects for ``-hostkey``."""

    match = _HOST_KEY_RE.fullmatch(host_key.strip()) if isinstance(host_key, str) else None
    if match is None:
        raise ShoreUsbError("node record does not hold one Ed25519 host key")
    try:
        blob = base64.b64decode(match["blob"], validate=True)
    except ValueError as exc:
        raise ShoreUsbError("node host key is not valid base64") from exc
    digest = hashlib.sha256(blob).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _is_removable(drive_root: str) -> bool:
    return ctypes.windll.kernel32.GetDriveTypeW(drive_root) == _DRIVE_REMOVABLE


def _has_identity(root: Path) -> bool:
    return (root / "identity" / "controller.ppk").is_file()


def find_usb_root(environ: Mapping[str, str] | None = None) -> Path:
    """Locate the one ``ShoreControl`` directory that holds the USB identity.

    ``MSABP_SHORE_USB_ROOT`` names it explicitly.  Otherwise every removable
    drive is checked and exactly one must match; the drive letter may change.
    """

    env = os.environ if environ is None else environ
    explicit = env.get(USB_ROOT_ENV, "").strip()
    if explicit:
        root = Path(explicit)
        if not _has_identity(root):
            raise ShoreUsbError(f"{USB_ROOT_ENV}={explicit!r} holds no USB identity")
        return root
    if os.name != "nt":
        raise ShoreUsbError("USB discovery needs Windows; set " + USB_ROOT_ENV)
    found = [
        Path(f"{letter}:\\ShoreControl")
        for letter in string.ascii_uppercase
        if _is_removable(f"{letter}:\\") and _has_identity(Path(f"{letter}:\\ShoreControl"))
    ]
    if not found:
        raise ShoreUsbError(
            "Shore management USB is not inserted; no other SSH identity is used"
        )
    if len(found) > 1:
        raise ShoreUsbError(
            "more than one Shore management USB is inserted; set " + USB_ROOT_ENV
        )
    return found[0]


def load_nodes(root: Path) -> tuple[ShoreNode, ...]:
    nodes = []
    for path in sorted((root / "nodes").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            node = ShoreNode(
                host_name=str(record["HostName"]),
                user_name=str(record["UserName"]),
                address=str(record["Address"]),
                dns_name=str(record["DNSName"]).rstrip("."),
                host_key_fingerprint=host_key_fingerprint(record["HostKey"]),
                port=int(record.get("Port", 22)),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ShoreUsbError(f"cannot read node record {path.name}: {exc}") from exc
        if not _ADDRESS_RE.fullmatch(node.address) or not _USER_RE.fullmatch(node.user_name):
            raise ShoreUsbError(f"node record {path.name} has an invalid address or user")
        nodes.append(node)
    if not nodes:
        raise ShoreUsbError("the Shore management USB has no enrolled nodes")
    return tuple(nodes)


def resolve_node(nodes: Sequence[ShoreNode], user: str, host: str) -> ShoreNode:
    """Match ``user@host`` to one enrolled node by name, DNS name or address."""

    wanted = host.rstrip(".").casefold()
    matches = [
        node
        for node in nodes
        if wanted
        in (
            node.host_name.casefold(),
            node.dns_name.casefold(),
            node.dns_name.split(".", 1)[0].casefold(),
            node.address,
        )
    ]
    if len(matches) != 1:
        raise ShoreUsbError(f"host {host!r} is not one enrolled Shore node")
    node = matches[0]
    if node.user_name != user:
        raise ShoreUsbError(
            f"{node.host_name} is enrolled for user {node.user_name!r}, not {user!r}"
        )
    return node


def _split_openssh_arguments(arguments: Sequence[str]) -> tuple[int | None, list[str]]:
    """Return the requested port and the positional arguments.

    Only the options the legacy transport emits are accepted.  ``BatchMode``
    and ``ConnectTimeout`` have no Plink equivalent and are dropped: Plink runs
    with ``-batch`` and the caller's subprocess timeout still applies.
    """

    port: int | None = None
    index = 0
    while index < len(arguments) and arguments[index].startswith("-"):
        option = arguments[index]
        if index + 1 >= len(arguments):
            raise ShoreUsbError(f"option {option} has no value")
        value = arguments[index + 1]
        if option == "-o":
            if value.split("=", 1)[0] not in ("BatchMode", "ConnectTimeout"):
                raise ShoreUsbError(f"unsupported SSH option {value!r}")
        elif option in ("-p", "-P"):
            if not value.isdigit() or not 1 <= int(value) <= 65535:
                raise ShoreUsbError(f"invalid SSH port {value!r}")
            port = int(value)
        elif option == "-i":
            raise ShoreUsbError(
                "identity_file cannot be combined with the USB backend; "
                "the USB key is the only identity"
            )
        else:
            raise ShoreUsbError(f"unsupported SSH option {option!r}")
        index += 2
    return port, list(arguments[index:])


def _check_port(node: ShoreNode, port: int | None) -> None:
    if port is not None and port != node.port:
        raise ShoreUsbError(
            f"{node.host_name} is enrolled on port {node.port}, not {port}"
        )


def _identity_arguments(root: Path, node: ShoreNode) -> list[str]:
    return [
        "-i",
        str(root / "identity" / "controller.ppk"),
        "-hostkey",
        node.host_key_fingerprint,
    ]


def _translate_ssh(root: Path, nodes: Sequence[ShoreNode], arguments: Sequence[str]) -> list[str]:
    port, positional = _split_openssh_arguments(arguments)
    if len(positional) < 2:
        raise ShoreUsbError("SSH command line has no target or no remote command")
    target = _TARGET_RE.fullmatch(positional[0])
    if target is None:
        raise ShoreUsbError(f"SSH target {positional[0]!r} is not user@host")
    node = resolve_node(nodes, target["user"], target["host"])
    # OpenSSH joins the remaining words with spaces to form the remote command;
    # passing the joined string as one argument makes Plink send the same text.
    remote_command = " ".join(positional[1:])
    _check_port(node, port)
    # Same options as Get-ShoreClientArgs in the USB's Common.ps1, which is the
    # form already exercised against the enrolled hosts.
    return [
        str(root / "bin" / "plink.exe"),
        "-batch",
        "-no-antispoof",
        "-T",
        "-ssh",
        "-4",
        "-P",
        str(node.port),
        "-noagent",
        "-a",
        "-x",
        "-noshare",
        "-no-trivial-auth",
        *_identity_arguments(root, node),
        "-l",
        node.user_name,
        node.address,
        remote_command,
    ]


def _translate_scp(root: Path, nodes: Sequence[ShoreNode], arguments: Sequence[str]) -> list[str]:
    port, positional = _split_openssh_arguments(arguments)
    if len(positional) != 2:
        raise ShoreUsbError("SCP command line needs exactly one source and one destination")
    specs = [_REMOTE_SPEC_RE.fullmatch(item) for item in positional]
    if sum(spec is not None for spec in specs) != 1:
        raise ShoreUsbError("exactly one side of an SCP transfer must be remote")
    node = None
    rewritten = []
    for item, spec in zip(positional, specs):
        if spec is None:
            rewritten.append(item)
            continue
        node = resolve_node(nodes, spec["user"], spec["host"])
        rewritten.append(f"{node.user_name}@{node.address}:{spec['path']}")
    assert node is not None
    _check_port(node, port)
    # Same options as the PSCP calls in Shore's Send-PublicPackage.ps1.
    return [
        str(root / "bin" / "pscp.exe"),
        "-batch",
        "-sftp",
        "-4",
        "-P",
        str(node.port),
        "-noagent",
        *_identity_arguments(root, node),
        *rewritten,
    ]


def translate_command(
    command: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Rewrite an OpenSSH ``ssh``/``scp`` argv for the USB backend.

    Commands for other programs, and every command while the backend is not
    selected, are returned unchanged.
    """

    command = [str(item) for item in command]
    if not command or not backend_enabled(environ):
        return command
    program = Path(command[0]).name.lower().removesuffix(".exe")
    if program not in ("ssh", "scp"):
        return command
    root = find_usb_root(environ)
    for name in ("plink.exe", "pscp.exe"):
        if not (root / "bin" / name).is_file():
            raise ShoreUsbError(f"{name} is missing from the Shore management USB")
    nodes = load_nodes(root)
    if program == "ssh":
        return _translate_ssh(root, nodes, command[1:])
    return _translate_scp(root, nodes, command[1:])
