"""Deployment and launch transport for Princess-managed Maid devices.

Remote operations use Windows OpenSSH in batch mode.  PowerShell source is
sent with ``-EncodedCommand`` so paths and arguments are never interpolated by
an intermediate shell.  SSH is only used for short control operations: in the
normal ``ssh_process`` mode it attempts to start a hidden Maid process and then
returns.  Princess treats that launch as successful only after Maid contacts it
over HTTP, because some Windows remote-session policies reap child processes
when the SSH connection closes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from .bell import BellError, MaidBellClient
from .config import DeviceConfig, LaunchMode


REQUIRED_PYTHON_MODULES = (
    "cst.interface",
    "shapely",
    "scipy",
    "pandas",
    "numpy",
)
DEFAULT_MAID_RELATIVE_PATH = PureWindowsPath(
    "scripts", "simulation", "maid.py"
)
DEFAULT_TRANSFER_TIMEOUT_SECONDS = 600.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., Any]
BellClientFactory = Callable[..., MaidBellClient]


class TransportError(RuntimeError):
    """Raised when an SSH, SCP, local copy, or process launch fails."""


@dataclass(frozen=True)
class TransferReceipt:
    device_id: str
    source: str
    destination: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ModuleCheck:
    name: str
    available: bool
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DoctorReport:
    """Read-only readiness report for one Maid device."""

    device_id: str
    python_path: str
    python_exists: bool
    repo_exists: bool
    project_exists: bool
    runtime_config_exists: bool
    maid_entrypoint_exists: bool
    scheduled_task_exists: bool | None
    python_version: str | None
    modules: tuple[ModuleCheck, ...]
    bell_reachable: bool | None = None
    errors: tuple[str, ...] = ()

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.python_exists:
            missing.append("python")
        if not self.repo_exists:
            missing.append("repo_root")
        if not self.project_exists:
            missing.append("cst_project")
        if not self.runtime_config_exists:
            missing.append("runtime_config")
        if not self.maid_entrypoint_exists:
            missing.append("maid_entrypoint")
        if self.scheduled_task_exists is False:
            missing.append("scheduled_task")
        if self.bell_reachable is False:
            missing.append("maid_bell")
        missing.extend(check.name for check in self.modules if not check.available)
        return tuple(missing)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.missing_requirements


@dataclass(frozen=True)
class LaunchReceipt:
    device_id: str
    launch_mode: LaunchMode
    pid: int | None
    command: tuple[str, ...]
    stdout_path: str | None = None
    stderr_path: str | None = None


def _default_runner(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def encode_powershell_command(script: str) -> str:
    """Encode source exactly as required by Windows PowerShell."""

    if not isinstance(script, str) or not script.strip():
        raise ValueError("PowerShell script must be a non-empty string")
    return base64.b64encode(script.encode("utf-16le")).decode("ascii")


def decode_powershell_command(encoded: str) -> str:
    """Decode an EncodedCommand; primarily useful for audit tests."""

    try:
        return base64.b64decode(encoded, validate=True).decode("utf-16le")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid PowerShell EncodedCommand") from exc


def _remote_device(device: DeviceConfig) -> None:
    if not device.is_remote or device.ssh_target is None:
        raise ValueError(f"device {device.id!r} is not an SSH device")


def _ssh_options(device: DeviceConfig, *, for_scp: bool) -> list[str]:
    _remote_device(device)
    options = [
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={device.ssh_connect_timeout_seconds:g}",
    ]
    if device.identity_file:
        options.extend(["-i", os.path.expanduser(device.identity_file)])
    if device.ssh_port != 22:
        options.extend(["-P" if for_scp else "-p", str(device.ssh_port)])
    return options


def build_ssh_powershell_command(
    device: DeviceConfig,
    script: str,
    *,
    ssh_executable: str = "ssh",
) -> list[str]:
    """Build a non-interactive, injection-safe remote PowerShell command."""

    _remote_device(device)
    return [
        ssh_executable,
        *_ssh_options(device, for_scp=False),
        str(device.ssh_target),
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encode_powershell_command(script),
    ]


def _run_checked(
    command: Sequence[str],
    *,
    runner: CommandRunner,
    cwd: str | Path | None,
    timeout: float | None,
    action: str,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(list(command), cwd=cwd, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TransportError(f"{action} could not start: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        suffix = f": {detail}" if detail else ""
        raise TransportError(
            f"{action} failed with exit code {result.returncode}{suffix}"
        )
    return result


def run_remote_powershell(
    device: DeviceConfig,
    script: str,
    *,
    runner: CommandRunner = _default_runner,
    timeout: float | None = None,
    ssh_executable: str = "ssh",
    action: str = "remote PowerShell command",
) -> subprocess.CompletedProcess[str]:
    """Run one short PowerShell command through SSH BatchMode."""

    command = build_ssh_powershell_command(
        device,
        script,
        ssh_executable=ssh_executable,
    )
    effective_timeout = timeout or device.ssh_connect_timeout_seconds + 30.0
    return _run_checked(
        command,
        runner=runner,
        cwd=None,
        timeout=effective_timeout,
        action=f"{action} on {device.id}",
    )


def _ps_literal(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if any(character in text for character in ("\0", "\r", "\n")):
        raise ValueError("PowerShell literal contains a control character")
    return "'" + text.replace("'", "''") + "'"


def _windows_absolute_path(value: str, label: str) -> PureWindowsPath:
    if any(character in value for character in ("\0", "\r", "\n")):
        raise ValueError(f"{label} contains a control character")
    path = PureWindowsPath(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute Windows path")
    return path


def _scp_remote_spec(device: DeviceConfig, remote_path: PureWindowsPath) -> str:
    _remote_device(device)
    path_text = remote_path.as_posix()
    return f"{device.ssh_target}:{path_text}"


def _build_scp_command(
    device: DeviceConfig,
    source: str,
    destination: str,
    *,
    scp_executable: str,
) -> list[str]:
    return [
        scp_executable,
        *_ssh_options(device, for_scp=True),
        source,
        destination,
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_transfer_timeout(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("transfer_timeout must be a number or None")
    result = float(value)
    if result <= 0.0:
        raise ValueError("transfer_timeout must be positive")
    return result


def _local_copy_atomic(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination}")
    if source.resolve() == destination.resolve():
        if sha256_file(destination) != expected_sha256:
            raise TransportError("source and destination hash changed during copy")
        return

    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.msabp-part"
    )
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise TransportError("local copy failed SHA-256 verification")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def push_file_atomic(
    device: DeviceConfig,
    local_path: str | Path,
    destination_path: str,
    *,
    overwrite: bool = True,
    runner: CommandRunner = _default_runner,
    ssh_executable: str = "ssh",
    scp_executable: str = "scp",
    transfer_timeout: float | None = DEFAULT_TRANSFER_TIMEOUT_SECONDS,
) -> TransferReceipt:
    """Atomically distribute one file to a local or SSH Maid device."""

    source = Path(local_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"distribution source does not exist: {source}")
    destination = _windows_absolute_path(destination_path, "destination_path")
    transfer_timeout = _validated_transfer_timeout(transfer_timeout)
    digest = sha256_file(source)
    size = source.stat().st_size

    if device.launch_mode is LaunchMode.LOCAL:
        local_destination = Path(str(destination))
        _local_copy_atomic(
            source,
            local_destination,
            overwrite=overwrite,
            expected_sha256=digest,
        )
        return TransferReceipt(
            device_id=device.id,
            source=str(source),
            destination=str(local_destination.resolve()),
            size_bytes=size,
            sha256=digest,
        )

    temporary = destination.parent / (
        f".{destination.name}.{uuid4().hex}.msabp-part"
    )
    parent_literal = _ps_literal(str(destination.parent))
    temp_literal = _ps_literal(str(temporary))
    destination_literal = _ps_literal(str(destination))
    prepared = False
    committed = False
    try:
        run_remote_powershell(
            device,
            "\n".join(
                (
                    "$ErrorActionPreference = 'Stop'",
                    f"[System.IO.Directory]::CreateDirectory({parent_literal}) "
                    "| Out-Null",
                )
            ),
            runner=runner,
            ssh_executable=ssh_executable,
            action="prepare atomic upload",
        )
        prepared = True

        scp_command = _build_scp_command(
            device,
            str(source),
            _scp_remote_spec(device, temporary),
            scp_executable=scp_executable,
        )
        _run_checked(
            scp_command,
            runner=runner,
            cwd=None,
            timeout=transfer_timeout,
            action=f"SCP upload to {device.id}",
        )

        replace_lines = [
            "$ErrorActionPreference = 'Stop'",
            f"$temporary = {temp_literal}",
            f"$destination = {destination_literal}",
            "if (-not [System.IO.File]::Exists($temporary)) {",
            "    throw 'atomic upload temporary file is missing'",
            "}",
            "$actualHash = (Get-FileHash -LiteralPath $temporary "
            "-Algorithm SHA256).Hash.ToLowerInvariant()",
            f"if ($actualHash -ne '{digest}') {{",
            "    [System.IO.File]::Delete($temporary)",
            "    throw 'atomic upload SHA-256 mismatch'",
            "}",
            "if ([System.IO.File]::Exists($destination)) {",
        ]
        if overwrite:
            replace_lines.extend(
                (
                    # Windows PowerShell on both Maid hosts rejects a null
                    # backup path passed to File.Replace with
                    # "The path is not of a legal form".  The temporary file
                    # is already hash-verified and lives beside the target, so
                    # a forced same-directory move provides the required
                    # replace operation without that incompatible overload.
                    "    Move-Item -LiteralPath $temporary "
                    "-Destination $destination -Force",
                )
            )
        else:
            replace_lines.extend(
                (
                    "    [System.IO.File]::Delete($temporary)",
                    "    throw 'atomic upload destination already exists'",
                )
            )
        replace_lines.extend(
            (
                "} else {",
                "    [System.IO.File]::Move($temporary, $destination)",
                "}",
            )
        )
        run_remote_powershell(
            device,
            "\n".join(replace_lines),
            runner=runner,
            ssh_executable=ssh_executable,
            action="commit atomic upload",
        )
        committed = True
    finally:
        if prepared and not committed:
            try:
                run_remote_powershell(
                    device,
                    "\n".join(
                        (
                            "$ErrorActionPreference = 'SilentlyContinue'",
                            f"Remove-Item -LiteralPath {temp_literal} -Force",
                        )
                    ),
                    runner=runner,
                    ssh_executable=ssh_executable,
                    action="clean failed atomic upload",
                )
            except TransportError:
                pass

    return TransferReceipt(
        device_id=device.id,
        source=str(source),
        destination=str(destination),
        size_bytes=size,
        sha256=digest,
    )


def copy_device_file_atomic(
    device: DeviceConfig,
    source_path: str,
    destination_path: str,
    *,
    overwrite: bool = True,
    runner: CommandRunner = _default_runner,
    ssh_executable: str = "ssh",
) -> TransferReceipt:
    """Atomically copy a file already resident on one Maid device.

    Propagation templates contain manually configured infrastructure and are
    therefore authoritative on each Maid host.  This operation creates a
    private per-launch working copy without uploading a local CST file.
    """

    source = _windows_absolute_path(source_path, "source_path")
    destination = _windows_absolute_path(destination_path, "destination_path")
    if device.launch_mode is LaunchMode.LOCAL:
        local_source = Path(str(source)).resolve()
        local_destination = Path(str(destination)).resolve()
        if not local_source.is_file():
            raise FileNotFoundError(
                f"device-local copy source does not exist: {local_source}"
            )
        digest = sha256_file(local_source)
        _local_copy_atomic(
            local_source,
            local_destination,
            overwrite=overwrite,
            expected_sha256=digest,
        )
        return TransferReceipt(
            device_id=device.id,
            source=str(local_source),
            destination=str(local_destination),
            size_bytes=local_destination.stat().st_size,
            sha256=digest,
        )

    temporary = destination.parent / (
        f".{destination.name}.{uuid4().hex}.msabp-local-part"
    )
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"$source = {_ps_literal(str(source))}",
        f"$destination = {_ps_literal(str(destination))}",
        f"$temporary = {_ps_literal(str(temporary))}",
        "if (-not [System.IO.File]::Exists($source)) {",
        "    throw 'device-local CST template does not exist'",
        "}",
        "[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($destination)) | Out-Null",
        "$sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()",
        "if ([System.String]::Equals([System.IO.Path]::GetFullPath($source), [System.IO.Path]::GetFullPath($destination), [System.StringComparison]::OrdinalIgnoreCase)) {",
        "    $size = (Get-Item -LiteralPath $source).Length",
        "    Write-Output (\"{0}|{1}\" -f $size, $sourceHash)",
        "    exit 0",
        "}",
        "Copy-Item -LiteralPath $source -Destination $temporary -Force",
        "$temporaryHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant()",
        "if ($temporaryHash -ne $sourceHash) {",
        "    Remove-Item -LiteralPath $temporary -Force",
        "    throw 'device-local copy SHA-256 mismatch'",
        "}",
        "if ([System.IO.File]::Exists($destination)) {",
    ]
    if overwrite:
        lines.append(
            "    Move-Item -LiteralPath $temporary -Destination $destination -Force"
        )
    else:
        lines.extend(
            (
                "    Remove-Item -LiteralPath $temporary -Force",
                "    throw 'device-local copy destination already exists'",
            )
        )
    lines.extend(
        (
            "} else {",
            "    [System.IO.File]::Move($temporary, $destination)",
            "}",
            "$size = (Get-Item -LiteralPath $destination).Length",
            "$finalHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()",
            "Write-Output (\"{0}|{1}\" -f $size, $finalHash)",
        )
    )
    try:
        result = run_remote_powershell(
            device,
            "\n".join(lines),
            runner=runner,
            ssh_executable=ssh_executable,
            action="copy device-local project template",
        )
    except BaseException:
        try:
            run_remote_powershell(
                device,
                "\n".join(
                    (
                        "$ErrorActionPreference = 'SilentlyContinue'",
                        f"Remove-Item -LiteralPath {_ps_literal(str(temporary))} -Force",
                    )
                ),
                runner=runner,
                ssh_executable=ssh_executable,
                action="clean failed device-local copy",
            )
        except TransportError:
            pass
        raise

    receipt_pattern = re.compile(r"^(\d+)\|([0-9a-f]{64})$")
    match = next(
        (
            receipt_pattern.fullmatch(line.strip().lower())
            for line in reversed((result.stdout or "").splitlines())
            if receipt_pattern.fullmatch(line.strip().lower()) is not None
        ),
        None,
    )
    if match is None:
        raise TransportError("device-local copy returned no size/hash receipt")
    return TransferReceipt(
        device_id=device.id,
        source=str(source),
        destination=str(destination),
        size_bytes=int(match.group(1)),
        sha256=match.group(2),
    )


def _last_sha256_line(output: str) -> str:
    for line in reversed(output.splitlines()):
        candidate = line.strip().lower()
        if _SHA256_RE.fullmatch(candidate):
            return candidate
    raise TransportError("remote SHA-256 query returned no digest")


def pull_file_atomic(
    device: DeviceConfig,
    source_path: str,
    local_path: str | Path,
    *,
    overwrite: bool = True,
    expected_sha256: str | None = None,
    runner: CommandRunner = _default_runner,
    ssh_executable: str = "ssh",
    scp_executable: str = "scp",
    transfer_timeout: float | None = DEFAULT_TRANSFER_TIMEOUT_SECONDS,
) -> TransferReceipt:
    """Atomically retrieve one file from a local or SSH Maid device."""

    source = _windows_absolute_path(source_path, "source_path")
    transfer_timeout = _validated_transfer_timeout(transfer_timeout)
    destination = Path(local_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination}")

    if device.launch_mode is LaunchMode.LOCAL:
        local_source = Path(str(source)).resolve()
        if not local_source.is_file():
            raise FileNotFoundError(f"retrieval source does not exist: {local_source}")
        digest = sha256_file(local_source)
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise TransportError("local retrieval source SHA-256 mismatch")
        _local_copy_atomic(
            local_source,
            destination,
            overwrite=overwrite,
            expected_sha256=digest,
        )
        return TransferReceipt(
            device_id=device.id,
            source=str(local_source),
            destination=str(destination),
            size_bytes=destination.stat().st_size,
            sha256=digest,
        )

    if expected_sha256 is None:
        hash_result = run_remote_powershell(
            device,
            "\n".join(
                (
                    "$ErrorActionPreference = 'Stop'",
                    f"$source = {_ps_literal(str(source))}",
                    "if (-not [System.IO.File]::Exists($source)) {",
                    "    throw 'retrieval source does not exist'",
                    "}",
                    "(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash"
                    ".ToLowerInvariant()",
                )
            ),
            runner=runner,
            ssh_executable=ssh_executable,
            action="query retrieval hash",
        )
        expected_sha256 = _last_sha256_line(hash_result.stdout or "")
    else:
        expected_sha256 = expected_sha256.strip().lower()
        if not _SHA256_RE.fullmatch(expected_sha256):
            raise ValueError("expected_sha256 must contain 64 hexadecimal digits")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.msabp-part"
    )
    try:
        scp_command = _build_scp_command(
            device,
            _scp_remote_spec(device, source),
            str(temporary),
            scp_executable=scp_executable,
        )
        _run_checked(
            scp_command,
            runner=runner,
            cwd=None,
            timeout=transfer_timeout,
            action=f"SCP download from {device.id}",
        )
        if not temporary.is_file():
            raise TransportError("SCP reported success but no local file was created")
        actual_hash = sha256_file(temporary)
        if actual_hash != expected_sha256:
            raise TransportError("downloaded file failed SHA-256 verification")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return TransferReceipt(
        device_id=device.id,
        source=str(source),
        destination=str(destination),
        size_bytes=destination.stat().st_size,
        sha256=expected_sha256,
    )


def _python_probe_source() -> str:
    return """\
import json
import subprocess
import sys
names=("cst.interface","shapely","scipy","pandas","numpy")
result={"python_version":sys.version.split()[0],"modules":{}}
for name in names:
    code=("import importlib as i,json;n="+repr(name)+";m=i.import_module(n);"
          "r=i.import_module(n.split('.',1)[0]);v=getattr(m,'__version__',"
          "getattr(r,'__version__',None));print(json.dumps({'version':v}))")
    try:
        child=subprocess.run([sys.executable,"-c",code],capture_output=True,
                             text=True,timeout=15,check=False)
        if child.returncode:
            raise RuntimeError("child exit "+str(child.returncode)+": "+
                               (child.stderr or child.stdout).strip())
        result["modules"][name]={"available":True,
            "version":json.loads(child.stdout.splitlines()[-1])["version"],
            "error":None}
    except Exception as exc:
        result["modules"][name]={"available":False,"version":None,
                                 "error":type(exc).__name__+": "+str(exc)}
print(json.dumps(result,separators=(",",":")))
"""


def _json_object_from_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise TransportError("command output did not contain a JSON object")


def _module_checks(payload: Mapping[str, Any] | None) -> tuple[ModuleCheck, ...]:
    raw_modules = payload.get("modules", {}) if payload else {}
    if not isinstance(raw_modules, Mapping):
        raw_modules = {}
    checks: list[ModuleCheck] = []
    for name in REQUIRED_PYTHON_MODULES:
        raw = raw_modules.get(name, {})
        if not isinstance(raw, Mapping):
            raw = {}
        available = raw.get("available") is True
        version = raw.get("version")
        error = raw.get("error")
        checks.append(
            ModuleCheck(
                name=name,
                available=available,
                version=str(version) if version is not None else None,
                error=str(error) if error is not None else None,
            )
        )
    return tuple(checks)


def _placeholder_doctor_report(device: DeviceConfig) -> DoctorReport:
    return DoctorReport(
        device_id=device.id,
        python_path=device.python_path,
        python_exists=False,
        repo_exists=False,
        project_exists=False,
        runtime_config_exists=False,
        maid_entrypoint_exists=False,
        scheduled_task_exists=(
            False if device.launch_mode is LaunchMode.SCHEDULED_TASK else None
        ),
        python_version=None,
        modules=_module_checks(None),
        errors=("python_path is still a configuration placeholder",),
    )


def _doctor_from_payload(
    device: DeviceConfig,
    payload: Mapping[str, Any],
    *,
    extra_errors: Sequence[str] = (),
) -> DoctorReport:
    python_payload = payload.get("python")
    if not isinstance(python_payload, Mapping):
        python_payload = {}
    raw_errors = payload.get("errors", [])
    errors = [str(item) for item in raw_errors] if isinstance(raw_errors, list) else []
    errors.extend(extra_errors)
    return DoctorReport(
        device_id=device.id,
        python_path=device.python_path,
        python_exists=payload.get("python_exists") is True,
        repo_exists=payload.get("repo_exists") is True,
        project_exists=payload.get("project_exists") is True,
        runtime_config_exists=payload.get("runtime_config_exists") is True,
        maid_entrypoint_exists=payload.get("maid_entrypoint_exists") is True,
        scheduled_task_exists=(
            payload.get("scheduled_task_exists") is True
            if device.launch_mode is LaunchMode.SCHEDULED_TASK
            else None
        ),
        python_version=(
            str(python_payload.get("python_version"))
            if python_payload.get("python_version") is not None
            else None
        ),
        modules=_module_checks(python_payload),
        errors=tuple(errors),
    )


def doctor_device(
    device: DeviceConfig,
    *,
    runner: CommandRunner = _default_runner,
    ssh_executable: str = "ssh",
    bell_client_factory: BellClientFactory = MaidBellClient,
) -> DoctorReport:
    """Read-check Python, dependencies, project, runtime, and launch fallback."""

    if device.python_path_is_placeholder:
        return _placeholder_doctor_report(device)

    maid_path = str(PureWindowsPath(device.repo_root) / DEFAULT_MAID_RELATIVE_PATH)
    if device.launch_mode is LaunchMode.LOCAL:
        python_path = Path(device.python_path)
        repo_path = Path(device.repo_root)
        project_path = Path(device.project_path)
        runtime_path = Path(device.resolved_runtime_config_path)
        maid_entrypoint = Path(maid_path)
        python_exists = python_path.is_file()
        python_payload: Mapping[str, Any] = {}
        errors: list[str] = []
        if python_exists:
            try:
                result = _run_checked(
                    [str(python_path), "-c", _python_probe_source()],
                    runner=runner,
                    cwd=repo_path,
                    timeout=30.0,
                    action=f"Python dependency doctor on {device.id}",
                )
                python_payload = _json_object_from_output(result.stdout or "")
            except TransportError as exc:
                errors.append(str(exc))
        payload = {
            "python_exists": python_exists,
            "repo_exists": repo_path.is_dir(),
            "project_exists": project_path.is_file(),
            "runtime_config_exists": runtime_path.is_file(),
            "maid_entrypoint_exists": maid_entrypoint.is_file(),
            "scheduled_task_exists": None,
            "python": python_payload,
            "errors": errors,
        }
        return _with_bell_probe(
            device,
            _doctor_from_payload(device, payload),
            bell_client_factory=bell_client_factory,
        )

    probe_b64 = base64.b64encode(_python_probe_source().encode("utf-8")).decode(
        "ascii"
    )
    task_check: str
    if device.launch_mode is LaunchMode.SCHEDULED_TASK:
        task_check = (
            "$z.scheduled_task_exists=($null-ne(Get-ScheduledTask -TaskName "
            f"{_ps_literal(str(device.scheduled_task_name))} "
            "-ErrorAction SilentlyContinue))"
        )
    else:
        task_check = "$z.scheduled_task_exists=$null"

    script_lines = [
        f"$p={_ps_literal(device.python_path)}",
        "$e=[IO.Path]::GetDirectoryName($p)",
        "$env:CONDA_PREFIX=$e",
        "$env:PATH=$e+';'+(Join-Path $e 'Library\\bin')+';'"
        "+(Join-Path $e 'Scripts')+';'+$env:PATH",
        f"$r={_ps_literal(device.repo_root)}",
        f"$j={_ps_literal(device.project_path)}",
        f"$u={_ps_literal(device.resolved_runtime_config_path)}",
        f"$m={_ps_literal(maid_path)}",
        "$z=[ordered]@{python_exists=[IO.File]::Exists($p);"
        "repo_exists=[IO.Directory]::Exists($r);"
        "project_exists=[IO.File]::Exists($j);"
        "runtime_config_exists=[IO.File]::Exists($u);"
        "maid_entrypoint_exists=[IO.File]::Exists($m);"
        "scheduled_task_exists=$null;python=$null;errors=@()}",
        task_check,
        "if($z.python_exists){",
        "$c=\"import base64;exec(base64.b64decode("
        f"'{probe_b64}'))\"",
        "$o=@(& $p -c $c 2>&1);$x=$LASTEXITCODE",
        "if($x-eq 0){try{$z.python=$o[-1]|ConvertFrom-Json}"
        "catch{$z.errors+='invalid Python doctor JSON'}}"
        "else{$z.errors+=('Python doctor exit '+$x+': '+($o-join ' '))}",
        "}",
        "$z|ConvertTo-Json -Depth 8 -Compress",
    ]
    try:
        result = run_remote_powershell(
            device,
            "\n".join(script_lines),
            runner=runner,
            ssh_executable=ssh_executable,
            action="Maid doctor",
        )
        payload = _json_object_from_output(result.stdout or "")
        return _with_bell_probe(
            device,
            _doctor_from_payload(device, payload),
            bell_client_factory=bell_client_factory,
        )
    except TransportError as exc:
        report = DoctorReport(
            device_id=device.id,
            python_path=device.python_path,
            python_exists=False,
            repo_exists=False,
            project_exists=False,
            runtime_config_exists=False,
            maid_entrypoint_exists=False,
            scheduled_task_exists=(
                False if device.launch_mode is LaunchMode.SCHEDULED_TASK else None
            ),
            python_version=None,
            modules=_module_checks(None),
            errors=(str(exc),),
        )
        return _with_bell_probe(
            device,
            report,
            bell_client_factory=bell_client_factory,
        )


def _with_bell_probe(
    device: DeviceConfig,
    report: DoctorReport,
    *,
    bell_client_factory: BellClientFactory,
) -> DoctorReport:
    if device.launch_mode is not LaunchMode.BELL:
        return report
    try:
        client = bell_client_factory(
            str(device.bell_host),
            device.bell_port,
            timeout=device.bell_connect_timeout_seconds,
        )
        response = client.ping()
        if response.get("device_id") != device.id:
            raise BellError("Bell device_id does not match the registry")
    except Exception as exc:
        return replace(
            report,
            bell_reachable=False,
            errors=(*report.errors, f"Maid Bell probe failed: {exc}"),
        )
    return replace(report, bell_reachable=True)


def default_maid_arguments(device: DeviceConfig) -> tuple[str, ...]:
    """Return the stable checked-in Maid entrypoint/runtime argument pair."""

    maid_path = str(PureWindowsPath(device.repo_root) / DEFAULT_MAID_RELATIVE_PATH)
    return (maid_path, "--runtime-config", device.resolved_runtime_config_path)


def _windows_argument_line(arguments: Sequence[str]) -> str:
    if any(
        any(character in str(argument) for character in ("\0", "\r", "\n"))
        for argument in arguments
    ):
        raise ValueError("Maid arguments contain a control character")
    return subprocess.list2cmdline([str(argument) for argument in arguments])


def launch_maid(
    device: DeviceConfig,
    *,
    maid_arguments: Sequence[str] | None = None,
    runner: CommandRunner = _default_runner,
    popen_factory: PopenFactory = subprocess.Popen,
    ssh_executable: str = "ssh",
    bell_token: str | None = None,
    bell_client_factory: BellClientFactory = MaidBellClient,
) -> LaunchReceipt:
    """Wake one Maid without retaining a Princess-to-Maid SSH channel."""

    if not device.enabled:
        raise TransportError(f"device {device.id!r} is disabled")
    if device.python_path_is_placeholder:
        raise TransportError(f"device {device.id!r} has a placeholder python_path")

    arguments = tuple(
        str(item)
        for item in (
            default_maid_arguments(device)
            if maid_arguments is None
            else maid_arguments
        )
    )

    if device.launch_mode is LaunchMode.LOCAL:
        command = (device.python_path, *arguments)
        try:
            process = popen_factory(
                list(command),
                cwd=device.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise TransportError(f"local Maid launch failed: {exc}") from exc
        return LaunchReceipt(
            device_id=device.id,
            launch_mode=device.launch_mode,
            pid=int(process.pid),
            command=command,
            stdout_path=None,
            stderr_path=None,
        )

    if device.launch_mode is LaunchMode.BELL:
        if not isinstance(bell_token, str) or len(bell_token) < 32:
            raise TransportError("bell launch requires the current run API token")
        try:
            client = bell_client_factory(
                str(device.bell_host),
                device.bell_port,
                timeout=device.bell_connect_timeout_seconds,
            )
            response = client.wake(
                device_id=device.id,
                runtime_config_path=device.resolved_runtime_config_path,
                api_token=bell_token,
            )
        except BellError as exc:
            raise TransportError(f"Maid Bell launch failed on {device.id}: {exc}") from exc
        pid_value = response.get("pid")
        if isinstance(pid_value, bool) or not isinstance(pid_value, int) or pid_value < 1:
            raise TransportError("Maid Bell returned no valid PID")
        stdout_path = response.get("stdout_path")
        stderr_path = response.get("stderr_path")
        if not isinstance(stdout_path, str) or not stdout_path:
            raise TransportError("Maid Bell returned no stdout log path")
        if not isinstance(stderr_path, str) or not stderr_path:
            raise TransportError("Maid Bell returned no stderr log path")
        return LaunchReceipt(
            device_id=device.id,
            launch_mode=device.launch_mode,
            pid=pid_value,
            command=(
                "maid-bell",
                f"{device.bell_host}:{device.bell_port}",
                device.resolved_runtime_config_path,
            ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    if device.launch_mode is LaunchMode.SCHEDULED_TASK:
        script = "\n".join(
            (
                "$ErrorActionPreference = 'Stop'",
                f"$taskName = {_ps_literal(str(device.scheduled_task_name))}",
                "& schtasks.exe /Run /TN $taskName | Out-Null",
                "if ($LASTEXITCODE -ne 0) {",
                "    throw ('schtasks /Run failed with exit code ' + $LASTEXITCODE)",
                "}",
            )
        )
        command = tuple(
            build_ssh_powershell_command(
                device,
                script,
                ssh_executable=ssh_executable,
            )
        )
        _run_checked(
            command,
            runner=runner,
            cwd=None,
            timeout=device.ssh_connect_timeout_seconds + 30.0,
            action=f"scheduled Maid launch on {device.id}",
        )
        return LaunchReceipt(
            device_id=device.id,
            launch_mode=device.launch_mode,
            pid=None,
            command=command,
            stdout_path=None,
            stderr_path=None,
        )

    argument_line = _windows_argument_line(arguments)
    script = "\n".join(
        (
            "$ErrorActionPreference = 'Stop'",
            f"$pythonPath = {_ps_literal(device.python_path)}",
            "$environmentRoot = [System.IO.Path]::GetDirectoryName($pythonPath)",
            "$env:CONDA_PREFIX = $environmentRoot",
            "$env:PATH = $environmentRoot + ';' + "
            "(Join-Path $environmentRoot 'Library\\bin') + ';' + "
            "(Join-Path $environmentRoot 'Scripts') + ';' + $env:PATH",
            f"$repoRoot = {_ps_literal(device.repo_root)}",
            f"$runtimePath = {_ps_literal(device.resolved_runtime_config_path)}",
            f"$arguments = {_ps_literal(argument_line)}",
            "if (-not [System.IO.File]::Exists($pythonPath)) {",
            "    throw 'configured Maid Python does not exist'",
            "}",
            "if (-not [System.IO.File]::Exists($runtimePath)) {",
            "    throw 'Maid runtime JSON has not been distributed'",
            "}",
            "$logDirectory = Join-Path $repoRoot 'logs'",
            "[System.IO.Directory]::CreateDirectory($logDirectory) | Out-Null",
            "$launchId = [Guid]::NewGuid().ToString('N')",
            f"$logStem = 'maid.{device.id}.' + $launchId",
            "$stdoutPath = Join-Path $logDirectory ($logStem + '.stdout.log')",
            "$stderrPath = Join-Path $logDirectory ($logStem + '.stderr.log')",
            "$process = Start-Process -FilePath $pythonPath "
            "-ArgumentList $arguments -WorkingDirectory $repoRoot "
            "-WindowStyle Hidden -RedirectStandardOutput $stdoutPath "
            "-RedirectStandardError $stderrPath -PassThru",
            "[ordered]@{ pid = $process.Id; stdout_path = $stdoutPath; "
            "stderr_path = $stderrPath } | ConvertTo-Json -Compress",
        )
    )
    command = tuple(
        build_ssh_powershell_command(
            device,
            script,
            ssh_executable=ssh_executable,
        )
    )
    result = _run_checked(
        command,
        runner=runner,
        cwd=None,
        timeout=device.ssh_connect_timeout_seconds + 30.0,
        action=f"detached Maid launch on {device.id}",
    )
    payload = _json_object_from_output(result.stdout or "")
    pid_value = payload.get("pid")
    if isinstance(pid_value, bool) or not isinstance(pid_value, int) or pid_value < 1:
        raise TransportError("detached Maid launch returned no valid PID")
    stdout_path = payload.get("stdout_path")
    stderr_path = payload.get("stderr_path")
    if not isinstance(stdout_path, str) or not stdout_path:
        raise TransportError("detached Maid launch returned no stdout log path")
    if not isinstance(stderr_path, str) or not stderr_path:
        raise TransportError("detached Maid launch returned no stderr log path")
    return LaunchReceipt(
        device_id=device.id,
        launch_mode=device.launch_mode,
        pid=pid_value,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def start_maid(
    device: DeviceConfig,
    *,
    maid_arguments: Sequence[str] | None = None,
    runner: CommandRunner = _default_runner,
    popen_factory: PopenFactory = subprocess.Popen,
    ssh_executable: str = "ssh",
    bell_token: str | None = None,
    bell_client_factory: BellClientFactory = MaidBellClient,
) -> LaunchReceipt:
    """Compatibility spelling for :func:`launch_maid`."""

    return launch_maid(
        device,
        maid_arguments=maid_arguments,
        runner=runner,
        popen_factory=popen_factory,
        ssh_executable=ssh_executable,
        bell_token=bell_token,
        bell_client_factory=bell_client_factory,
    )
