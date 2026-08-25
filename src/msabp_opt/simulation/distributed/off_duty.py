"""Emergency, recoverable shutdown for currently running Maid Bells."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from .bell import MaidBellClient
from .config import DeviceConfig, DeviceRegistry, LaunchMode
from .state import PrincessState


DEFAULT_OFF_DUTY_REASON = "one-click all-Maid off-duty request"


@dataclass(frozen=True)
class RunDismissal:
    run_id: str
    released_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class DeviceDismissal:
    device_id: str
    status: str
    pid: int | None = None
    run_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class OffDutyReport:
    runs: tuple[RunDismissal, ...]
    devices: tuple[DeviceDismissal, ...]

    @property
    def ok(self) -> bool:
        return all(item.error is None for item in self.devices)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "runs": [asdict(item) for item in self.runs],
            "devices": [asdict(item) for item in self.devices],
        }


@dataclass(frozen=True)
class _RunningBell:
    device: DeviceConfig
    runtime_config_path: str
    pid: int | None
    run_id: str


def _run_id_from_runtime_path(runtime_config_path: str) -> str:
    parts = PureWindowsPath(runtime_config_path).parts
    lowered = tuple(part.casefold() for part in parts)
    for index in range(len(parts) - 2):
        if lowered[index : index + 2] == ("simulations", "runs"):
            run_id = parts[index + 2]
            if run_id:
                return run_id
    raise ValueError(
        "Bell runtime path does not contain simulations/runs/<run-id>: "
        f"{runtime_config_path}"
    )


def _load_run_token(
    runs_root: Path,
    *,
    run_id: str,
    device_id: str,
) -> str:
    runtime_path = runs_root / run_id / "princess_runtime.json"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    if payload.get("run_id") != run_id:
        raise ValueError(f"Princess runtime run_id mismatch: {runtime_path}")
    device_ids = payload.get("device_ids")
    if not isinstance(device_ids, list) or device_id not in device_ids:
        raise ValueError(
            f"Princess runtime does not authorize Maid {device_id!r}: {runtime_path}"
        )
    token = payload.get("api_token")
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError(f"Princess runtime has no strong API token: {runtime_path}")
    return token


def _bell_devices(
    registry: DeviceRegistry,
    device_ids: tuple[str, ...],
) -> tuple[DeviceConfig, ...]:
    if device_ids:
        selected = tuple(registry.get_device(device_id) for device_id in device_ids)
        unsupported = tuple(
            device.id
            for device in selected
            if device.launch_mode is not LaunchMode.BELL
        )
        if unsupported:
            raise ValueError(
                "one-click off-duty currently requires launch_mode=bell: "
                + ", ".join(unsupported)
            )
    else:
        selected = registry.devices
    return tuple(device for device in selected if device.launch_mode is LaunchMode.BELL)


def dismiss_all_maids(
    registry: DeviceRegistry,
    *,
    runs_root: str | Path,
    device_ids: tuple[str, ...] = (),
    reason: str = DEFAULT_OFF_DUTY_REASON,
    bell_client_factory: type[MaidBellClient] = MaidBellClient,
) -> OffDutyReport:
    """Stop current Bell Maids and return only their exact active runs to pending."""

    root = Path(runs_root).expanduser().resolve()
    devices = _bell_devices(registry, device_ids)
    device_results: list[DeviceDismissal] = []
    running_bells: list[_RunningBell] = []

    for device in devices:
        try:
            client = bell_client_factory(
                str(device.bell_host),
                device.bell_port,
                timeout=device.bell_connect_timeout_seconds,
            )
            response = client.ping()
            status = str(response.get("status", "unknown"))
            if status != "running":
                device_results.append(DeviceDismissal(device.id, status))
                continue
            runtime_path = response.get("runtime_config_path")
            if not isinstance(runtime_path, str) or not runtime_path:
                raise ValueError("running Maid Bell returned no runtime_config_path")
            pid_value = response.get("pid")
            pid = (
                int(pid_value)
                if isinstance(pid_value, int) and not isinstance(pid_value, bool)
                else None
            )
            running_bells.append(
                _RunningBell(
                    device,
                    runtime_path,
                    pid,
                    _run_id_from_runtime_path(runtime_path),
                )
            )
        except Exception as exc:
            device_results.append(
                DeviceDismissal(
                    device.id,
                    "probe_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    run_results: list[RunDismissal] = []
    active_run_ids = sorted({item.run_id for item in running_bells})
    for run_id in active_run_ids:
        database_path = root / run_id / "princess.sqlite3"
        try:
            if not database_path.is_file():
                raise FileNotFoundError(
                    f"Princess state database does not exist: {database_path}"
                )
            with PrincessState(database_path) as state:
                released = state.request_stop(run_id, reason=reason)
            run_results.append(RunDismissal(run_id, released))
        except Exception as exc:
            device_results.append(
                DeviceDismissal(
                    f"Princess:{run_id}",
                    "state_stop_failed",
                    run_id=run_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    for running in running_bells:
        try:
            token = _load_run_token(
                root,
                run_id=running.run_id,
                device_id=running.device.id,
            )
            client = bell_client_factory(
                str(running.device.bell_host),
                running.device.bell_port,
                timeout=running.device.bell_connect_timeout_seconds,
            )
            response = client.stop(
                device_id=running.device.id,
                runtime_config_path=running.runtime_config_path,
                api_token=token,
            )
            response_pid = response.get("pid")
            pid = (
                int(response_pid)
                if isinstance(response_pid, int)
                and not isinstance(response_pid, bool)
                else running.pid
            )
            device_results.append(
                DeviceDismissal(
                    running.device.id,
                    str(response.get("status", "stopped")),
                    pid=pid,
                    run_id=running.run_id,
                )
            )
        except Exception as exc:
            device_results.append(
                DeviceDismissal(
                    running.device.id,
                    "stop_failed",
                    pid=running.pid,
                    run_id=running.run_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return OffDutyReport(tuple(run_results), tuple(device_results))
