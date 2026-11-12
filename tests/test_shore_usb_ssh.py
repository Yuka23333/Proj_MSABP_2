from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed import transport  # noqa: E402
from msabp_opt.simulation.distributed.config import device_from_mapping  # noqa: E402
from msabp_opt.simulation.distributed.shore_usb_ssh import (  # noqa: E402
    BACKEND_ENV,
    USB_ROOT_ENV,
    ShoreUsbError,
    backend_enabled,
    host_key_fingerprint,
    translate_command,
)

# Synthetic host key: 51 bytes shaped like an Ed25519 public-key blob.
_BLOB = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + bytes(range(32))
_HOST_KEY = "ssh-ed25519 " + base64.b64encode(_BLOB).decode("ascii") + " system@Test"
_FINGERPRINT = "SHA256:" + base64.b64encode(hashlib.sha256(_BLOB).digest()).decode().rstrip("=")
_ENCODED = "ZQBjAGgAbwA="


def _usb(tmp_path: Path, *, nodes: dict[str, dict] | None = None) -> Path:
    root = tmp_path / "ShoreControl"
    (root / "identity").mkdir(parents=True)
    (root / "identity" / "controller.ppk").write_text("synthetic", encoding="ascii")
    (root / "bin").mkdir()
    for name in ("plink.exe", "pscp.exe"):
        (root / "bin" / name).write_bytes(b"")
    (root / "nodes").mkdir()
    if nodes is None:
        nodes = {
            "coconutg2": {
                "HostName": "COCONUTG2",
                "UserName": "telecom",
                "Address": "100.108.0.45",
                "DNSName": "coconutg2.example.ts.net",
                "HostKey": _HOST_KEY,
                "Port": 22,
            }
        }
    for name, record in nodes.items():
        # Windows PowerShell 5.1 may write a BOM; the reader must accept it.
        (root / "nodes" / f"{name}.json").write_text(json.dumps(record), encoding="utf-8-sig")
    return root


def _env(root: Path) -> dict[str, str]:
    return {BACKEND_ENV: "shore-usb", USB_ROOT_ENV: str(root)}


def _ssh(target: str = "telecom@coconutg2") -> list[str]:
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target,
        "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
        "-EncodedCommand", _ENCODED,
    ]


def test_backend_is_off_by_default_and_leaves_commands_alone() -> None:
    assert backend_enabled({}) is False
    assert backend_enabled({BACKEND_ENV: "openssh"}) is False
    assert translate_command(_ssh(), environ={}) == _ssh()


def test_unknown_backend_name_is_rejected() -> None:
    with pytest.raises(ShoreUsbError):
        backend_enabled({BACKEND_ENV: "pageant"})


def test_fingerprint_matches_openssh_sha256_form() -> None:
    assert host_key_fingerprint(_HOST_KEY) == _FINGERPRINT


def test_ssh_becomes_plink_with_usb_key_and_pinned_host_key(tmp_path: Path) -> None:
    root = _usb(tmp_path)
    command = translate_command(_ssh(), environ=_env(root))
    assert command == [
        str(root / "bin" / "plink.exe"),
        "-batch", "-no-antispoof", "-T", "-ssh", "-4", "-P", "22",
        "-noagent", "-a", "-x", "-noshare", "-no-trivial-auth",
        "-i", str(root / "identity" / "controller.ppk"),
        "-hostkey", _FINGERPRINT,
        "-l", "telecom", "100.108.0.45",
        "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand " + _ENCODED,
    ]


@pytest.mark.parametrize(
    "host", ["COCONUTG2", "coconutg2.example.ts.net", "coconutg2.example.ts.net.", "100.108.0.45"]
)
def test_host_may_be_named_by_hostname_dns_or_address(tmp_path: Path, host: str) -> None:
    command = translate_command(_ssh(f"telecom@{host}"), environ=_env(_usb(tmp_path)))
    assert "100.108.0.45" in command


def test_scp_upload_and_download_become_pscp(tmp_path: Path) -> None:
    root = _usb(tmp_path)
    prefix = [
        str(root / "bin" / "pscp.exe"),
        "-batch", "-sftp", "-4", "-P", "22", "-noagent",
        "-i", str(root / "identity" / "controller.ppk"),
        "-hostkey", _FINGERPRINT,
    ]
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    remote = "telecom@coconutg2:D:/Academic/Proj/in put.csv"
    pinned = "telecom@100.108.0.45:D:/Academic/Proj/in put.csv"
    local = r"C:\Users\David\a@b\file.csv"
    upload = translate_command(["scp", *options, local, remote], environ=_env(root))
    download = translate_command(["scp", *options, remote, local], environ=_env(root))
    assert upload == [*prefix, local, pinned]
    assert download == [*prefix, pinned, local]


def test_other_programs_are_not_touched(tmp_path: Path) -> None:
    command = ["python.exe", "-c", "print(1)"]
    assert translate_command(command, environ=_env(_usb(tmp_path))) == command


@pytest.mark.parametrize(
    "command",
    [
        _ssh("telecom@unknownhost"),
        _ssh("someoneelse@coconutg2"),
        ["ssh", "-i", "C:/key", "telecom@coconutg2", "whoami"],
        ["ssh", "-p", "2222", "telecom@coconutg2", "whoami"],
        ["ssh", "-o", "ProxyCommand=evil", "telecom@coconutg2", "whoami"],
        ["ssh", "-A", "telecom@coconutg2", "whoami"],
        ["ssh", "telecom@coconutg2"],
        ["scp", "C:/a", "C:/b"],
        ["scp", "telecom@coconutg2:D:/a", "telecom@coconutg2:D:/b"],
    ],
)
def test_unsafe_or_unknown_requests_fail_closed(tmp_path: Path, command: list[str]) -> None:
    with pytest.raises(ShoreUsbError):
        translate_command(command, environ=_env(_usb(tmp_path)))


def test_missing_usb_identity_is_an_error_not_a_fallback(tmp_path: Path) -> None:
    root = _usb(tmp_path)
    (root / "identity" / "controller.ppk").unlink()
    with pytest.raises(ShoreUsbError, match="USB identity"):
        translate_command(_ssh(), environ=_env(root))


def test_node_with_non_tailnet_address_is_rejected(tmp_path: Path) -> None:
    root = _usb(
        tmp_path,
        nodes={
            "bad": {
                "HostName": "COCONUTG2", "UserName": "telecom", "Address": "192.168.1.5",
                "DNSName": "coconutg2.example.ts.net", "HostKey": _HOST_KEY,
            }
        },
    )
    with pytest.raises(ShoreUsbError):
        translate_command(_ssh(), environ=_env(root))


def test_default_runner_executes_the_translated_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _usb(tmp_path)
    monkeypatch.setenv(BACKEND_ENV, "shore-usb")
    monkeypatch.setenv(USB_ROOT_ENV, str(root))
    seen: list[list[str]] = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(transport.subprocess, "run", fake_run)
    device = device_from_mapping(
        {
            "id": "coconutg2",
            "enabled": True,
            "launch_mode": "ssh_process",
            "ssh_target": "telecom@coconutg2",
            "repo_root": r"D:\Academic\Proj_MSABP_2",
            "python_path": r"C:\Users\telecom\miniforge3\envs\maid\python.exe",
        }
    )
    result = transport.run_remote_powershell(device, "echo")
    assert result.stdout == "ok"
    assert Path(seen[0][0]).name == "plink.exe"
    assert seen[0][-1].startswith("powershell.exe -NoLogo")


def test_usb_failure_surfaces_as_transport_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(BACKEND_ENV, "shore-usb")
    monkeypatch.setenv(USB_ROOT_ENV, str(tmp_path / "absent"))
    device = device_from_mapping(
        {
            "id": "coconutg2",
            "enabled": True,
            "launch_mode": "ssh_process",
            "ssh_target": "telecom@coconutg2",
            "repo_root": r"D:\Academic\Proj_MSABP_2",
            "python_path": r"C:\Users\telecom\miniforge3\envs\maid\python.exe",
        }
    )
    with pytest.raises(transport.TransportError, match="could not start"):
        transport.run_remote_powershell(device, "echo")
