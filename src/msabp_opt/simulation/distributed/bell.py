"""Persistent device-local wake service for CST Maid processes.

The Bell is intentionally smaller and longer-lived than a Maid.  Princess
still distributes immutable run files over SSH/SCP, then sends one authenticated
JSON/TCP wake request.  The Bell validates the already-deployed runtime JSON and
starts ``maid.py`` locally, outside the Windows OpenSSH job object.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import socket
import socketserver
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


BELL_PROTOCOL_VERSION = 1
DEFAULT_BELL_PORT = 8766
DEFAULT_BELL_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_BELL_CLOCK_SKEW_SECONDS = 120.0
MAX_BELL_FRAME_BYTES = 64 * 1024
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BellError(RuntimeError):
    """Raised when a Bell request, configuration, or local launch fails."""


class BellAuthenticationError(BellError):
    """Raised when a wake request cannot be authenticated."""


class BellBusyError(BellError):
    """Raised when a different Maid launch already owns this device."""


def default_bell_config_path() -> Path:
    """Return the stable machine-local configuration used by the service."""

    program_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
    return Path(program_data) / "MSABP Maid Bell" / "bell.json"


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BellError(f"{label} must be a non-empty string")
    if any(character in value for character in ("\0", "\r", "\n")):
        raise BellError(f"{label} contains a control character")
    return value.strip()


def _safe_id(value: Any, label: str) -> str:
    text = _required_text(value, label)
    if not _SAFE_ID.fullmatch(text) or text in {".", ".."}:
        raise BellError(f"{label} is unsafe: {text!r}")
    return text


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BellError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BellError(f"{label} must be finite and positive")
    return result


def _port(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BellError(f"{label} must be an integer")
    if not 1 <= value <= 65535:
        raise BellError(f"{label} must be between 1 and 65535")
    return value


@dataclass(frozen=True)
class MaidBellConfig:
    """Machine-local Bell configuration; it deliberately contains no secret."""

    device_id: str
    listen_host: str
    port: int
    repo_root: Path
    python_path: Path
    maid_entrypoint: Path
    request_timeout_seconds: float = DEFAULT_BELL_CONNECT_TIMEOUT_SECONDS
    clock_skew_seconds: float = DEFAULT_BELL_CLOCK_SKEW_SECONDS
    schema_version: int = BELL_PROTOCOL_VERSION

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MaidBellConfig":
        allowed = {
            "schema_version",
            "device_id",
            "listen_host",
            "port",
            "repo_root",
            "python_path",
            "maid_entrypoint",
            "request_timeout_seconds",
            "clock_skew_seconds",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise BellError(f"Bell config contains unknown keys: {sorted(unknown)}")
        if raw.get("schema_version") != BELL_PROTOCOL_VERSION:
            raise BellError(
                "Bell config schema_version must be "
                f"{BELL_PROTOCOL_VERSION}"
            )
        repo_root = Path(_required_text(raw.get("repo_root"), "repo_root")).resolve()
        maid_raw = raw.get("maid_entrypoint")
        maid_entrypoint = (
            repo_root / "scripts" / "simulation" / "maid.py"
            if maid_raw is None
            else Path(_required_text(maid_raw, "maid_entrypoint")).resolve()
        )
        try:
            maid_entrypoint.relative_to(repo_root)
        except ValueError as exc:
            raise BellError("maid_entrypoint must be inside repo_root") from exc
        return cls(
            device_id=_safe_id(raw.get("device_id"), "device_id"),
            listen_host=_required_text(raw.get("listen_host"), "listen_host"),
            port=_port(raw.get("port", DEFAULT_BELL_PORT), "port"),
            repo_root=repo_root,
            python_path=Path(
                _required_text(raw.get("python_path"), "python_path")
            ).resolve(),
            maid_entrypoint=maid_entrypoint,
            request_timeout_seconds=_positive_float(
                raw.get(
                    "request_timeout_seconds",
                    DEFAULT_BELL_CONNECT_TIMEOUT_SECONDS,
                ),
                "request_timeout_seconds",
            ),
            clock_skew_seconds=_positive_float(
                raw.get("clock_skew_seconds", DEFAULT_BELL_CLOCK_SKEW_SECONDS),
                "clock_skew_seconds",
            ),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MaidBellConfig":
        source = Path(path or default_bell_config_path()).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise BellError("Bell config must contain one JSON object")
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("repo_root", "python_path", "maid_entrypoint"):
            payload[name] = str(payload[name])
        return payload

    @property
    def runs_root(self) -> Path:
        return (self.repo_root / "simulations" / "runs").resolve()

    @property
    def state_path(self) -> Path:
        return self.runs_root / f"maid-bell.{self.device_id}.state.json"


def write_bell_config(path: str | Path, config: MaidBellConfig) -> Path:
    """Atomically write a machine-local Bell configuration."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(config.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _signed_wake_request(
    *,
    device_id: str,
    runtime_config_path: str,
    api_token: str,
    timestamp: float | None = None,
    request_id: str | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    if not isinstance(api_token, str) or len(api_token) < 32:
        raise BellAuthenticationError("Bell wake requires a strong run API token")
    request = {
        "protocol_version": BELL_PROTOCOL_VERSION,
        "command": "wake",
        "request_id": request_id or uuid4().hex,
        "device_id": _safe_id(device_id, "device_id"),
        "runtime_config_path": _required_text(
            runtime_config_path,
            "runtime_config_path",
        ),
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "nonce": nonce or secrets.token_hex(16),
    }
    request["signature"] = hmac.new(
        api_token.encode("utf-8"),
        _canonical_bytes(request),
        hashlib.sha256,
    ).hexdigest()
    return request


def _read_json_line(stream: Any, *, maximum: int = MAX_BELL_FRAME_BYTES) -> dict[str, Any]:
    raw = stream.readline(maximum + 1)
    if not raw:
        raise BellError("Bell connection closed without a request")
    if len(raw) > maximum or not raw.endswith(b"\n"):
        raise BellError("Bell frame is too large or missing its newline terminator")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BellError("Bell frame is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise BellError("Bell frame must contain one JSON object")
    return dict(payload)


def _write_json_line(stream: Any, payload: Mapping[str, Any]) -> None:
    body = _canonical_bytes(payload) + b"\n"
    if len(body) > MAX_BELL_FRAME_BYTES:
        raise BellError("Bell response exceeds the frame limit")
    stream.write(body)
    stream.flush()


def _pid_is_alive(pid: int) -> bool:
    if pid < 1:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    process_query_limited_information = 0x1000
    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
            handle,
            ctypes.byref(exit_code),
        ):
            return False
        return exit_code.value == still_active
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class MaidLaunchState:
    pid: int
    runtime_config_path: str
    stdout_path: str
    stderr_path: str
    started_at: float


class MaidBellController:
    """Validate Bell requests and supervise one device-local Maid process."""

    def __init__(
        self,
        config: MaidBellConfig,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        pid_alive: Callable[[int], bool] = _pid_is_alive,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._popen_factory = popen_factory
        self._pid_alive = pid_alive
        self._time = time_fn
        self._lock = threading.RLock()
        self._process: Any | None = None
        self._state: MaidLaunchState | None = self._load_state()
        self._responses: dict[str, tuple[str, dict[str, Any], float]] = {}

    def _load_state(self) -> MaidLaunchState | None:
        try:
            raw = json.loads(self.config.state_path.read_text(encoding="utf-8"))
            state = MaidLaunchState(
                pid=int(raw["pid"]),
                runtime_config_path=str(raw["runtime_config_path"]),
                stdout_path=str(raw["stdout_path"]),
                stderr_path=str(raw["stderr_path"]),
                started_at=float(raw["started_at"]),
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return state if self._pid_alive(state.pid) else None

    def _write_state(self, state: MaidLaunchState | None) -> None:
        path = self.config.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if state is None:
            path.unlink(missing_ok=True)
            return
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _refresh_state(self) -> MaidLaunchState | None:
        state = self._state
        if state is None:
            return None
        alive = (
            self._process.poll() is None
            if self._process is not None and int(self._process.pid) == state.pid
            else self._pid_alive(state.pid)
        )
        if alive:
            return state
        self._process = None
        self._state = None
        self._write_state(None)
        return None

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._refresh_state()
            return {
                "ok": True,
                "protocol_version": BELL_PROTOCOL_VERSION,
                "device_id": self.config.device_id,
                "status": "running" if state is not None else "idle",
                "pid": None if state is None else state.pid,
                "runtime_config_path": (
                    None if state is None else state.runtime_config_path
                ),
            }

    def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "ping":
            return self.status()
        if command != "wake":
            raise BellError(f"unsupported Bell command: {command!r}")
        return self._wake(request)

    def _runtime_and_token(self, raw_path: Any) -> tuple[Path, dict[str, Any], str]:
        runtime_path = Path(_required_text(raw_path, "runtime_config_path")).resolve()
        try:
            runtime_path.relative_to(self.config.runs_root)
        except ValueError as exc:
            raise BellAuthenticationError(
                "runtime_config_path must be inside simulations/runs"
            ) from exc
        if runtime_path.name.casefold() != "maid_runtime.json":
            raise BellAuthenticationError(
                "runtime_config_path must name maid_runtime.json"
            )
        try:
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BellAuthenticationError("deployed runtime JSON is unavailable") from exc
        if not isinstance(payload, dict):
            raise BellAuthenticationError("deployed runtime JSON is invalid")
        if payload.get("worker_id") != self.config.device_id:
            raise BellAuthenticationError("runtime worker_id does not match this Bell")
        token = payload.get("api_token")
        if not isinstance(token, str) or len(token) < 32:
            raise BellAuthenticationError("runtime JSON has no strong API token")
        return runtime_path, payload, token

    def _verify_wake(
        self,
        request: Mapping[str, Any],
    ) -> tuple[Path, str, str]:
        allowed = {
            "protocol_version",
            "command",
            "request_id",
            "device_id",
            "runtime_config_path",
            "timestamp",
            "nonce",
            "signature",
        }
        if set(request) != allowed:
            raise BellAuthenticationError("wake request has missing or unknown fields")
        if request.get("protocol_version") != BELL_PROTOCOL_VERSION:
            raise BellAuthenticationError("unsupported Bell protocol version")
        if request.get("device_id") != self.config.device_id:
            raise BellAuthenticationError("wake request targets a different device")
        request_id = _safe_id(request.get("request_id"), "request_id")
        nonce = _required_text(request.get("nonce"), "nonce")
        if len(nonce) < 16 or len(nonce) > 128:
            raise BellAuthenticationError("wake nonce length is invalid")
        timestamp_raw = request.get("timestamp")
        if isinstance(timestamp_raw, bool) or not isinstance(timestamp_raw, (int, float)):
            raise BellAuthenticationError("wake timestamp must be numeric")
        timestamp = float(timestamp_raw)
        if not math.isfinite(timestamp) or abs(self._time() - timestamp) > self.config.clock_skew_seconds:
            raise BellAuthenticationError("wake timestamp is outside the allowed window")
        runtime_path, _runtime, token = self._runtime_and_token(
            request.get("runtime_config_path")
        )
        signature = str(request.get("signature", "")).lower()
        if not _SHA256.fullmatch(signature):
            raise BellAuthenticationError("wake signature is invalid")
        unsigned = dict(request)
        unsigned.pop("signature", None)
        expected = hmac.new(
            token.encode("utf-8"),
            _canonical_bytes(unsigned),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise BellAuthenticationError("wake signature does not match")
        return runtime_path, request_id, signature

    def _wake(self, request: Mapping[str, Any]) -> dict[str, Any]:
        runtime_path, request_id, signature = self._verify_wake(request)
        now = self._time()
        with self._lock:
            self._responses = {
                key: value
                for key, value in self._responses.items()
                if now - value[2] <= self.config.clock_skew_seconds * 2.0
            }
            cached = self._responses.get(request_id)
            if cached is not None:
                if not hmac.compare_digest(cached[0], signature):
                    raise BellAuthenticationError(
                        "request_id was reused with a different signature"
                    )
                return dict(cached[1])

            current = self._refresh_state()
            if current is not None:
                if Path(current.runtime_config_path) != runtime_path:
                    raise BellBusyError(
                        f"Maid pid {current.pid} already owns another runtime"
                    )
                response = self._response_for_state(current, "already_running")
                self._responses[request_id] = (signature, response, now)
                return response

            state = self._launch(runtime_path)
            response = self._response_for_state(state, "started")
            self._responses[request_id] = (signature, response, now)
            return response

    @staticmethod
    def _response_for_state(state: MaidLaunchState, status: str) -> dict[str, Any]:
        return {
            "ok": True,
            "protocol_version": BELL_PROTOCOL_VERSION,
            "status": status,
            "pid": state.pid,
            "runtime_config_path": state.runtime_config_path,
            "stdout_path": state.stdout_path,
            "stderr_path": state.stderr_path,
        }

    def _launch(self, runtime_path: Path) -> MaidLaunchState:
        if not self.config.python_path.is_file():
            raise BellError(f"Maid Python does not exist: {self.config.python_path}")
        if not self.config.maid_entrypoint.is_file():
            raise BellError(
                f"Maid entrypoint does not exist: {self.config.maid_entrypoint}"
            )
        log_root = self.config.repo_root / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        launch_id = uuid4().hex
        stem = f"maid.{self.config.device_id}.{launch_id}"
        stdout_path = log_root / f"{stem}.stdout.log"
        stderr_path = log_root / f"{stem}.stderr.log"

        environment = os.environ.copy()
        environment_root = self.config.python_path.parent
        environment["CONDA_PREFIX"] = str(environment_root)
        environment["PATH"] = os.pathsep.join(
            (
                str(environment_root),
                str(environment_root / "Library" / "bin"),
                str(environment_root / "Scripts"),
                environment.get("PATH", ""),
            )
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        command = [
            str(self.config.python_path),
            str(self.config.maid_entrypoint),
            "--runtime-config",
            str(runtime_path),
        ]
        try:
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                process = self._popen_factory(
                    command,
                    cwd=str(self.config.repo_root),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                    close_fds=True,
                    creationflags=creationflags,
                )
        except OSError as exc:
            raise BellError(f"local Maid launch failed: {exc}") from exc
        pid = int(process.pid)
        if pid < 1:
            raise BellError("local Maid launch returned an invalid PID")
        state = MaidLaunchState(
            pid=pid,
            runtime_config_path=str(runtime_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            started_at=self._time(),
        )
        self._process = process
        self._state = state
        self._write_state(state)
        return state


class _ThreadingBellServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        controller: MaidBellController,
    ) -> None:
        self.controller = controller
        super().__init__(server_address, _BellRequestHandler)


class _BellRequestHandler(socketserver.StreamRequestHandler):
    server: _ThreadingBellServer

    def handle(self) -> None:
        self.request.settimeout(self.server.controller.config.request_timeout_seconds)
        try:
            request = _read_json_line(self.rfile)
            response = self.server.controller.handle_request(request)
        except Exception as exc:
            response = {
                "ok": False,
                "protocol_version": BELL_PROTOCOL_VERSION,
                "error_kind": type(exc).__name__,
                "error": str(exc),
            }
        _write_json_line(self.wfile, response)


class MaidBellServer:
    """Small TCP server suitable for foreground debug or a Windows service."""

    def __init__(
        self,
        config: MaidBellConfig,
        *,
        controller: MaidBellController | None = None,
    ) -> None:
        self.config = config
        self.controller = controller or MaidBellController(config)
        self._lifecycle_lock = threading.Lock()
        self._serving = threading.Event()
        self._closed = False
        self._server = _ThreadingBellServer(
            (config.listen_host, config.port),
            self.controller,
        )

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise BellError("Maid Bell server is already closed")
            self._serving.set()
        try:
            self._server.serve_forever(poll_interval=0.25)
        finally:
            self._serving.clear()

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            serving = self._serving.is_set()
        if serving:
            self._server.shutdown()
        self._server.server_close()


class MaidBellClient:
    """Princess-side one-request-per-connection JSON/TCP Bell client."""

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_BELL_PORT,
        *,
        timeout: float = DEFAULT_BELL_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        self.host = _required_text(host, "Bell host")
        self.port = _port(port, "Bell port")
        self.timeout = _positive_float(timeout, "Bell timeout")

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = _canonical_bytes(payload) + b"\n"
        if len(body) > MAX_BELL_FRAME_BYTES:
            raise BellError("Bell request exceeds the frame limit")
        try:
            with socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout,
            ) as connection:
                connection.settimeout(self.timeout)
                connection.sendall(body)
                stream = connection.makefile("rb")
                response = _read_json_line(stream)
        except OSError as exc:
            raise BellError(
                f"Maid Bell {self.host}:{self.port} is unreachable: {exc}"
            ) from exc
        if response.get("protocol_version") != BELL_PROTOCOL_VERSION:
            raise BellError("Maid Bell returned an unsupported protocol version")
        if response.get("ok") is not True:
            kind = str(response.get("error_kind", "BellError"))
            raise BellError(f"{kind}: {response.get('error', 'wake failed')}")
        return response

    def ping(self) -> dict[str, Any]:
        return self._request(
            {
                "protocol_version": BELL_PROTOCOL_VERSION,
                "command": "ping",
                "request_id": uuid4().hex,
            }
        )

    def wake(
        self,
        *,
        device_id: str,
        runtime_config_path: str,
        api_token: str,
    ) -> dict[str, Any]:
        return self._request(
            _signed_wake_request(
                device_id=device_id,
                runtime_config_path=runtime_config_path,
                api_token=api_token,
            )
        )
