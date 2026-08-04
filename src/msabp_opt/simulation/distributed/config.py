"""Validated device configuration for Princess and Maid.

The checked-in registry contains no credentials.  SSH authentication remains
the responsibility of Windows OpenSSH, and the Princess API token is supplied
through the runtime JSON/environment rather than this file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEVICE_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "simulation" / "princess_devices.json"
)
DEFAULT_RUNTIME_CONFIG_RELATIVE_PATH = PureWindowsPath(
    "simulations", "runs", "active_maid_runtime.json"
)
DEFAULT_PROJECT_RELATIVE_PATH = PureWindowsPath(
    "simulations", "models", "msa-bp.cst"
)
PYTHON_PATH_PLACEHOLDER = "<SET_MAID_PYTHON_PATH>"

_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_REGISTRY_KEYS = {
    "schema_version",
    "bind_host",
    "advertise_url",
    "port",
    "devices",
}
_DEVICE_KEYS = {
    "id",
    "enabled",
    "launch_mode",
    "ssh_target",
    "repo_root",
    "python_path",
    "scheduled_task_name",
    "runtime_config_path",
    "ssh_port",
    "ssh_connect_timeout_seconds",
    "identity_file",
}


class DeviceConfigError(ValueError):
    """Raised when a Princess device registry is malformed or ambiguous."""


class LaunchMode(str, Enum):
    """How Princess wakes a Maid process on a device."""

    LOCAL = "local"
    SSH_PROCESS = "ssh_process"
    SCHEDULED_TASK = "scheduled_task"


def is_placeholder_path(value: str) -> bool:
    """Return whether a path is intentionally awaiting user configuration."""

    text = value.strip()
    return text == PYTHON_PATH_PLACEHOLDER or (
        text.startswith("<") and text.endswith(">")
    )


@dataclass(frozen=True)
class DeviceConfig:
    """One local or SSH-addressable Maid host."""

    id: str
    enabled: bool
    launch_mode: LaunchMode
    ssh_target: str | None
    repo_root: str
    python_path: str
    scheduled_task_name: str | None = None
    runtime_config_path: str | None = None
    ssh_port: int = 22
    ssh_connect_timeout_seconds: float = 10.0
    identity_file: str | None = None

    @property
    def is_remote(self) -> bool:
        """Whether this device is reached through OpenSSH."""

        return self.launch_mode is not LaunchMode.LOCAL

    @property
    def project_path(self) -> str:
        """Return the device-local CST project path."""

        return str(PureWindowsPath(self.repo_root) / DEFAULT_PROJECT_RELATIVE_PATH)

    @property
    def resolved_runtime_config_path(self) -> str:
        """Return the stable runtime JSON path read by Maid."""

        if self.runtime_config_path is not None:
            return self.runtime_config_path
        return str(
            PureWindowsPath(self.repo_root) / DEFAULT_RUNTIME_CONFIG_RELATIVE_PATH
        )

    @property
    def python_path_is_placeholder(self) -> bool:
        return is_placeholder_path(self.python_path)


@dataclass(frozen=True)
class DeviceRegistry:
    """Princess listener settings and its known Maid devices."""

    bind_host: str
    advertise_url: str
    port: int
    devices: tuple[DeviceConfig, ...]
    schema_version: int = SCHEMA_VERSION

    @property
    def enabled_devices(self) -> tuple[DeviceConfig, ...]:
        return tuple(device for device in self.devices if device.enabled)

    def get_device(self, device_id: str) -> DeviceConfig:
        for device in self.devices:
            if device.id == device_id:
                return device
        raise KeyError(f"unknown Maid device: {device_id!r}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeviceConfigError(f"{label} must be a JSON object")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise DeviceConfigError(f"{label} contains unknown keys: {sorted(unknown)}")


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeviceConfigError(f"{label} must be a non-empty string")
    if any(character in value for character in ("\0", "\r", "\n")):
        raise DeviceConfigError(f"{label} contains a control character")
    return value.strip()


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, label)


def _windows_absolute_path(value: Any, label: str) -> str:
    text = _required_text(value, label)
    if not PureWindowsPath(text).is_absolute():
        raise DeviceConfigError(f"{label} must be an absolute Windows path")
    return text


def _python_path(value: Any, label: str) -> str:
    text = _required_text(value, label)
    if is_placeholder_path(text):
        return text
    if not PureWindowsPath(text).is_absolute():
        raise DeviceConfigError(
            f"{label} must be an absolute Windows path or "
            f"{PYTHON_PATH_PLACEHOLDER!r}"
        )
    return text


def _positive_int(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeviceConfigError(f"{label} must be an integer")
    if not 1 <= value <= maximum:
        raise DeviceConfigError(f"{label} must be between 1 and {maximum}")
    return value


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeviceConfigError(f"{label} must be a number")
    result = float(value)
    if result <= 0.0:
        raise DeviceConfigError(f"{label} must be positive")
    return result


def _validate_ssh_target(value: str, label: str) -> str:
    if value.startswith("-") or any(character.isspace() for character in value):
        raise DeviceConfigError(f"{label} is not a safe OpenSSH target")
    return value


def device_from_mapping(
    raw_device: Mapping[str, Any],
    *,
    index: int = 0,
) -> DeviceConfig:
    """Validate and construct one :class:`DeviceConfig`."""

    raw = _mapping(raw_device, f"devices[{index}]")
    _reject_unknown_keys(raw, _DEVICE_KEYS, f"devices[{index}]")
    label = f"devices[{index}]"

    device_id = _required_text(raw.get("id"), f"{label}.id")
    if not _DEVICE_ID_RE.fullmatch(device_id):
        raise DeviceConfigError(
            f"{label}.id must match {_DEVICE_ID_RE.pattern!r}"
        )

    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise DeviceConfigError(f"{label}.enabled must be boolean")

    raw_mode = _required_text(raw.get("launch_mode"), f"{label}.launch_mode")
    try:
        launch_mode = LaunchMode(raw_mode)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in LaunchMode)
        raise DeviceConfigError(
            f"{label}.launch_mode must be one of: {choices}"
        ) from exc

    ssh_target = _optional_text(raw.get("ssh_target"), f"{label}.ssh_target")
    repo_root = _windows_absolute_path(raw.get("repo_root"), f"{label}.repo_root")
    python_path = _python_path(raw.get("python_path"), f"{label}.python_path")
    scheduled_task_name = _optional_text(
        raw.get("scheduled_task_name"),
        f"{label}.scheduled_task_name",
    )
    runtime_config_path_raw = raw.get("runtime_config_path")
    runtime_config_path = (
        None
        if runtime_config_path_raw is None
        else _windows_absolute_path(
            runtime_config_path_raw,
            f"{label}.runtime_config_path",
        )
    )
    identity_file = _optional_text(
        raw.get("identity_file"),
        f"{label}.identity_file",
    )
    ssh_port = _positive_int(raw.get("ssh_port", 22), f"{label}.ssh_port", 65535)
    connect_timeout = _positive_number(
        raw.get("ssh_connect_timeout_seconds", 10.0),
        f"{label}.ssh_connect_timeout_seconds",
    )

    if launch_mode is LaunchMode.LOCAL:
        if ssh_target is not None:
            raise DeviceConfigError(f"{label}.ssh_target must be null for local mode")
        if scheduled_task_name is not None:
            raise DeviceConfigError(
                f"{label}.scheduled_task_name must be null for local mode"
            )
    else:
        if ssh_target is None:
            raise DeviceConfigError(
                f"{label}.ssh_target is required for {launch_mode.value}"
            )
        ssh_target = _validate_ssh_target(ssh_target, f"{label}.ssh_target")

    if launch_mode is LaunchMode.SCHEDULED_TASK and scheduled_task_name is None:
        raise DeviceConfigError(
            f"{label}.scheduled_task_name is required for scheduled_task mode"
        )
    if launch_mode is LaunchMode.SSH_PROCESS and scheduled_task_name is not None:
        raise DeviceConfigError(
            f"{label}.scheduled_task_name must be null for ssh_process mode"
        )
    if enabled and is_placeholder_path(python_path):
        raise DeviceConfigError(
            f"{label} is enabled but python_path is still a placeholder"
        )

    return DeviceConfig(
        id=device_id,
        enabled=enabled,
        launch_mode=launch_mode,
        ssh_target=ssh_target,
        repo_root=repo_root,
        python_path=python_path,
        scheduled_task_name=scheduled_task_name,
        runtime_config_path=runtime_config_path,
        ssh_port=ssh_port,
        ssh_connect_timeout_seconds=connect_timeout,
        identity_file=identity_file,
    )


def registry_from_mapping(raw_registry: Mapping[str, Any]) -> DeviceRegistry:
    """Validate and construct a complete registry mapping."""

    raw = _mapping(raw_registry, "device registry")
    _reject_unknown_keys(raw, _REGISTRY_KEYS, "device registry")

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DeviceConfigError(
            f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}"
        )
    bind_host = _required_text(raw.get("bind_host"), "bind_host")
    advertise_url = _required_text(raw.get("advertise_url"), "advertise_url")
    port = _positive_int(raw.get("port"), "port", 65535)

    parsed_url = urlsplit(advertise_url)
    if parsed_url.scheme != "http" or not parsed_url.hostname:
        raise DeviceConfigError("advertise_url must be an absolute http:// URL")
    if parsed_url.path not in {"", "/"} or parsed_url.query or parsed_url.fragment:
        raise DeviceConfigError(
            "advertise_url must use the server root path without query or fragment"
        )
    if parsed_url.username is not None or parsed_url.password is not None:
        raise DeviceConfigError("advertise_url must not contain user information")
    try:
        advertised_port = parsed_url.port
    except ValueError as exc:
        raise DeviceConfigError(f"advertise_url has an invalid port: {exc}") from exc
    effective_port = advertised_port or 80
    if effective_port != port:
        raise DeviceConfigError(
            "advertise_url port must match the top-level port setting"
        )

    raw_devices = raw.get("devices")
    if not isinstance(raw_devices, list) or not raw_devices:
        raise DeviceConfigError("devices must be a non-empty JSON array")
    devices = tuple(
        device_from_mapping(_mapping(item, f"devices[{index}]"), index=index)
        for index, item in enumerate(raw_devices)
    )
    ids = [device.id for device in devices]
    duplicates = sorted({device_id for device_id in ids if ids.count(device_id) > 1})
    if duplicates:
        raise DeviceConfigError(f"duplicate device ids: {duplicates}")

    return DeviceRegistry(
        bind_host=bind_host,
        advertise_url=advertise_url.rstrip("/"),
        port=port,
        devices=devices,
        schema_version=version,
    )


def load_device_registry(
    path: str | Path = DEFAULT_DEVICE_CONFIG_PATH,
) -> DeviceRegistry:
    """Load a UTF-8 JSON device registry without changing external state."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise DeviceConfigError(
            f"invalid JSON in {source}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    return registry_from_mapping(_mapping(payload, str(source)))
