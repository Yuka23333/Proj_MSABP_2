from __future__ import annotations

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

from msabp_opt.simulation.distributed.config import (  # noqa: E402
    DEFAULT_DEVICE_CONFIG_PATH,
    PYTHON_PATH_PLACEHOLDER,
    DeviceConfig,
    DeviceConfigError,
    LaunchMode,
    device_from_mapping,
    load_device_registry,
    registry_from_mapping,
)
from msabp_opt.simulation.distributed.transport import (  # noqa: E402
    DEFAULT_TRANSFER_TIMEOUT_SECONDS,
    TransportError,
    decode_powershell_command,
    doctor_device,
    encode_powershell_command,
    launch_maid,
    pull_file_atomic,
    push_file_atomic,
)


def _remote_device(
    *,
    launch_mode: str = "ssh_process",
    enabled: bool = True,
    python_path: str = r"C:\Users\telecom\miniforge3\envs\maid\python.exe",
) -> DeviceConfig:
    return device_from_mapping(
        {
            "id": "remote-one",
            "enabled": enabled,
            "launch_mode": launch_mode,
            "ssh_target": "telecom@remote-one",
            "repo_root": r"D:\Academic\Proj_MSABP_2",
            "python_path": python_path,
            "scheduled_task_name": (
                "Proj_MSABP_Maid" if launch_mode == "scheduled_task" else None
            ),
            "runtime_config_path": (
                r"D:\Academic\Proj_MSABP_2\simulations\runs"
                r"\active_maid_runtime.json"
            ),
            "ssh_port": 2222,
            "ssh_connect_timeout_seconds": 7,
        }
    )


def _completed(
    command: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


class RecordingRunner:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[float | None] = []
        self.outputs = list(outputs or [])

    def __call__(
        self,
        command: list[str],
        *,
        cwd: str | Path | None,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[str]:
        _ = cwd, timeout
        self.calls.append(list(command))
        self.timeouts.append(timeout)
        stdout = self.outputs.pop(0) if self.outputs else ""
        return _completed(command, stdout=stdout)


def _decode_ssh_script(command: list[str]) -> str:
    encoded_index = command.index("-EncodedCommand") + 1
    return decode_powershell_command(command[encoded_index])


def test_checked_in_registry_has_two_remote_maids_and_optional_local() -> None:
    registry = load_device_registry(DEFAULT_DEVICE_CONFIG_PATH)

    assert registry.bind_host == "100.99.182.30"
    assert registry.advertise_url == "http://100.99.182.30:8765"
    assert registry.port == 8765
    assert not registry.enabled_devices

    convallaria = registry.get_device("convallariag5")
    coconut = registry.get_device("coconutg2")
    local = registry.get_device("local")
    assert convallaria.launch_mode is LaunchMode.SSH_PROCESS
    assert coconut.launch_mode is LaunchMode.SSH_PROCESS
    assert convallaria.ssh_target == "telecom@convallariag5"
    assert coconut.ssh_target == "telecom@coconutg2"
    assert convallaria.python_path.endswith(r"miniforge3\envs\maid\python.exe")
    assert coconut.python_path.endswith(r"miniforge3\envs\maid\python.exe")
    assert local.launch_mode is LaunchMode.LOCAL


def test_enabled_device_cannot_keep_python_placeholder() -> None:
    with pytest.raises(DeviceConfigError, match="placeholder"):
        _remote_device(python_path=PYTHON_PATH_PLACEHOLDER)


def test_registry_rejects_duplicate_ids_and_mismatched_url_port() -> None:
    raw_device = {
        "id": "same",
        "enabled": False,
        "launch_mode": "local",
        "ssh_target": None,
        "repo_root": r"D:\Repo",
        "python_path": r"C:\Python\python.exe",
        "scheduled_task_name": None,
    }
    with pytest.raises(DeviceConfigError, match="duplicate"):
        registry_from_mapping(
            {
                "schema_version": 1,
                "bind_host": "127.0.0.1",
                "advertise_url": "http://127.0.0.1:8765",
                "port": 8765,
                "devices": [raw_device, raw_device],
            }
        )

    with pytest.raises(DeviceConfigError, match="must match"):
        registry_from_mapping(
            {
                "schema_version": 1,
                "bind_host": "127.0.0.1",
                "advertise_url": "http://127.0.0.1:9999",
                "port": 8765,
                "devices": [raw_device],
            }
        )


@pytest.mark.parametrize(
    ("advertise_url", "error_match"),
    [
        ("https://127.0.0.1:8765", "http://"),
        ("http://127.0.0.1:8765/api", "root path"),
        ("http://127.0.0.1:8765/?token=nope", "root path"),
    ],
)
def test_registry_rejects_unsupported_advertise_urls(
    advertise_url: str,
    error_match: str,
) -> None:
    with pytest.raises(DeviceConfigError, match=error_match):
        registry_from_mapping(
            {
                "schema_version": 1,
                "bind_host": "127.0.0.1",
                "advertise_url": advertise_url,
                "port": 8765,
                "devices": [
                    {
                        "id": "local",
                        "enabled": False,
                        "launch_mode": "local",
                        "ssh_target": None,
                        "repo_root": r"D:\Repo",
                        "python_path": r"C:\Python\python.exe",
                        "scheduled_task_name": None,
                    }
                ],
            }
        )


def test_powershell_encoded_command_round_trip_preserves_unicode() -> None:
    source = "Write-Output '公主与女仆'\n$ErrorActionPreference = 'Stop'"
    assert decode_powershell_command(encode_powershell_command(source)) == source


def test_remote_atomic_push_uses_batchmode_scp_hash_and_replace(tmp_path: Path) -> None:
    source = tmp_path / "samples.csv"
    source.write_bytes(b"sample_id,value\n0,1.25\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    runner = RecordingRunner()

    receipt = push_file_atomic(
        _remote_device(),
        source,
        r"D:\Academic\Proj_MSABP_2\simulations\runs\active.csv",
        runner=runner,
    )

    assert receipt.sha256 == digest
    assert len(runner.calls) == 3
    assert runner.timeouts[1] == DEFAULT_TRANSFER_TIMEOUT_SECONDS
    prepare, scp, commit = runner.calls
    assert prepare[0] == "ssh"
    assert "BatchMode=yes" in prepare
    assert "-p" in prepare and "2222" in prepare
    assert scp[0] == "scp"
    assert "BatchMode=yes" in scp
    assert "-P" in scp and "2222" in scp
    assert ".msabp-part" in scp[-1]
    commit_script = _decode_ssh_script(commit)
    assert digest in commit_script
    assert "Get-FileHash" in commit_script
    assert "[System.IO.File]::Replace" in commit_script


def test_remote_atomic_push_cleans_temporary_file_after_scp_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime.json"
    source.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(
        command: list[str],
        *,
        cwd: str | Path | None,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[str]:
        _ = cwd, timeout
        calls.append(command)
        if command[0] == "scp":
            return _completed(command, returncode=1, stderr="network down")
        return _completed(command)

    with pytest.raises(TransportError, match="SCP upload"):
        push_file_atomic(
            _remote_device(),
            source,
            r"D:\Academic\Proj_MSABP_2\simulations\runs\runtime.json",
            runner=runner,
        )

    assert len(calls) == 3
    cleanup_script = _decode_ssh_script(calls[-1])
    assert "Remove-Item -LiteralPath" in cleanup_script


def test_remote_atomic_pull_downloads_to_temporary_then_replaces(
    tmp_path: Path,
) -> None:
    payload = b"frequency,s11\n3.0,-12.5\n"
    digest = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "case_0000" / "S11.csv"
    calls: list[list[str]] = []
    timeouts: list[float | None] = []

    def runner(
        command: list[str],
        *,
        cwd: str | Path | None,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[str]:
        _ = cwd, timeout
        calls.append(command)
        timeouts.append(timeout)
        if command[0] == "scp":
            Path(command[-1]).write_bytes(payload)
            return _completed(command)
        return _completed(command, stdout=digest + "\n")

    receipt = pull_file_atomic(
        _remote_device(),
        r"D:\Academic\Proj_MSABP_2\results\raw\case_0000\S11.csv",
        destination,
        runner=runner,
        transfer_timeout=123.0,
    )

    assert destination.read_bytes() == payload
    assert receipt.sha256 == digest
    assert len(calls) == 2
    assert calls[0][0] == "ssh"
    assert calls[1][0] == "scp"
    assert timeouts[1] == 123.0
    assert ".msabp-part" in calls[1][-1]


def test_atomic_transfer_rejects_nonpositive_timeout(tmp_path: Path) -> None:
    source = tmp_path / "runtime.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        push_file_atomic(
            _remote_device(),
            source,
            r"D:\Academic\Proj_MSABP_2\simulations\runs\runtime.json",
            transfer_timeout=0,
        )


def test_doctor_reports_missing_modules_and_scheduled_task() -> None:
    device = _remote_device(launch_mode="scheduled_task")
    module_payload = {
        name: {"available": True, "version": "1", "error": None}
        for name in ("cst.interface", "shapely", "scipy", "pandas", "numpy")
    }
    module_payload["shapely"] = {
        "available": False,
        "version": None,
        "error": "ModuleNotFoundError: shapely",
    }
    payload = {
        "python_exists": True,
        "repo_exists": True,
        "project_exists": True,
        "runtime_config_exists": True,
        "maid_entrypoint_exists": True,
        "scheduled_task_exists": False,
        "python": {"python_version": "3.11.13", "modules": module_payload},
        "errors": [],
    }
    runner = RecordingRunner([json.dumps(payload)])

    report = doctor_device(device, runner=runner)

    assert not report.ok
    assert report.python_version == "3.11.13"
    assert report.scheduled_task_exists is False
    assert "scheduled_task" in report.missing_requirements
    assert "shapely" in report.missing_requirements
    doctor_command = runner.calls[0]
    encoded = doctor_command[doctor_command.index("-EncodedCommand") + 1]
    assert len(encoded) < 7500
    script = decode_powershell_command(encoded)
    assert "Get-ScheduledTask" in script
    assert "base64.b64decode" in script
    assert "CONDA_PREFIX" in script
    assert "Library\\bin" in script


def test_doctor_placeholder_is_explicit_and_never_runs_command() -> None:
    device = _remote_device(enabled=False, python_path=PYTHON_PATH_PLACEHOLDER)
    runner = RecordingRunner()

    report = doctor_device(device, runner=runner)

    assert not report.ok
    assert "python" in report.missing_requirements
    assert "placeholder" in report.errors[0]
    assert runner.calls == []


def test_ssh_process_launch_is_detached_hidden_and_returns_pid() -> None:
    runner = RecordingRunner(
        [
            json.dumps(
                {
                    "pid": 4321,
                    "stdout_path": (
                        r"D:\Academic\Proj_MSABP_2\logs"
                        r"\maid.remote-one.abc.stdout.log"
                    ),
                    "stderr_path": (
                        r"D:\Academic\Proj_MSABP_2\logs"
                        r"\maid.remote-one.abc.stderr.log"
                    ),
                }
            )
        ]
    )
    device = _remote_device()

    receipt = launch_maid(device, runner=runner)

    assert receipt.pid == 4321
    assert receipt.stdout_path is not None
    assert receipt.stdout_path.endswith(".abc.stdout.log")
    assert receipt.stderr_path is not None
    assert receipt.stderr_path.endswith(".abc.stderr.log")
    assert receipt.launch_mode is LaunchMode.SSH_PROCESS
    assert len(runner.calls) == 1
    command = runner.calls[0]
    assert command[0] == "ssh"
    assert "BatchMode=yes" in command
    script = _decode_ssh_script(command)
    assert "Start-Process" in script
    assert "-WindowStyle Hidden" in script
    assert "-PassThru" in script
    assert "[Guid]::NewGuid()" in script
    assert "CONDA_PREFIX" in script
    assert "Library\\bin" in script
    assert "active_maid_runtime.json" in script
    assert "--runtime-config" in script
    assert "scripts\\simulation\\maid.py" in script


def test_scheduled_task_launch_remains_a_fallback() -> None:
    runner = RecordingRunner()
    device = _remote_device(launch_mode="scheduled_task")

    receipt = launch_maid(device, runner=runner)

    assert receipt.pid is None
    script = _decode_ssh_script(runner.calls[0])
    assert "schtasks.exe /Run" in script
    assert "Start-Process" not in script


def test_disabled_device_cannot_be_launched() -> None:
    with pytest.raises(TransportError, match="disabled"):
        launch_maid(_remote_device(enabled=False), runner=RecordingRunner())
