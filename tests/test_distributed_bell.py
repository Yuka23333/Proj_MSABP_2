from __future__ import annotations

import json
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from msabp_opt.simulation.distributed.bell import (
    BellAuthenticationError,
    BellBusyError,
    BellError,
    MaidBellClient,
    MaidBellConfig,
    MaidBellController,
    MaidBellServer,
    _signed_wake_request,
)


TOKEN = "bell-test-token-" + "x" * 32


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.alive = True

    def poll(self) -> int | None:
        return None if self.alive else 0


class RecordingPopen:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, command: list[str], **kwargs: Any) -> FakeProcess:
        process = FakeProcess(4100 + len(self.processes))
        self.calls.append((list(command), dict(kwargs)))
        self.processes.append(process)
        return process


def _bell_files(tmp_path: Path) -> tuple[MaidBellConfig, Path]:
    repo = tmp_path / "repo"
    maid = repo / "scripts" / "simulation" / "maid.py"
    maid.parent.mkdir(parents=True)
    maid.write_text("raise SystemExit(0)\n", encoding="utf-8")
    runtime = (
        repo
        / "simulations"
        / "runs"
        / "run-1"
        / "workers"
        / "maid-a"
        / "maid_runtime.json"
    )
    runtime.parent.mkdir(parents=True)
    runtime.write_text(
        json.dumps({"worker_id": "maid-a", "api_token": TOKEN}),
        encoding="utf-8",
    )
    config = MaidBellConfig.from_mapping(
        {
            "schema_version": 1,
            "device_id": "maid-a",
            "listen_host": "127.0.0.1",
            "port": 8766,
            "repo_root": str(repo),
            "python_path": sys.executable,
            "maid_entrypoint": str(maid),
        }
    )
    return config, runtime


def _request(runtime: Path, *, timestamp: float | None = None) -> dict[str, Any]:
    return _signed_wake_request(
        device_id="maid-a",
        runtime_config_path=str(runtime),
        api_token=TOKEN,
        timestamp=timestamp,
    )


def test_bell_starts_only_fixed_maid_and_is_idempotent(tmp_path: Path) -> None:
    config, runtime = _bell_files(tmp_path)
    popen = RecordingPopen()
    controller = MaidBellController(config, popen_factory=popen)
    request = _request(runtime)

    started = controller.handle_request(request)
    repeated = controller.handle_request(request)
    already_running = controller.handle_request(_request(runtime))

    assert started["status"] == "started"
    assert repeated == started
    assert already_running["status"] == "already_running"
    assert len(popen.calls) == 1
    command, kwargs = popen.calls[0]
    assert command == [
        str(config.python_path),
        str(config.maid_entrypoint),
        "--runtime-config",
        str(runtime.resolve()),
    ]
    assert kwargs["cwd"] == str(config.repo_root)
    assert kwargs["stdin"] is not None
    assert "shell" not in kwargs


def test_bell_rejects_tampering_stale_requests_and_path_escape(
    tmp_path: Path,
) -> None:
    config, runtime = _bell_files(tmp_path)
    controller = MaidBellController(
        config,
        popen_factory=RecordingPopen(),
        time_fn=lambda: 1_000.0,
    )

    tampered = _request(runtime, timestamp=1_000.0)
    tampered["timestamp"] = 999.0
    with pytest.raises(BellAuthenticationError, match="signature"):
        controller.handle_request(tampered)

    with pytest.raises(BellAuthenticationError, match="allowed window"):
        controller.handle_request(_request(runtime, timestamp=800.0))

    outside = tmp_path / "outside" / "maid_runtime.json"
    outside.parent.mkdir()
    outside.write_text(
        json.dumps({"worker_id": "maid-a", "api_token": TOKEN}),
        encoding="utf-8",
    )
    with pytest.raises(BellAuthenticationError, match="inside simulations/runs"):
        controller.handle_request(_request(outside, timestamp=1_000.0))


def test_bell_refuses_a_second_runtime_while_maid_is_alive(tmp_path: Path) -> None:
    config, runtime = _bell_files(tmp_path)
    second_runtime = runtime.parents[2] / "run-2" / "maid_runtime.json"
    second_runtime.parent.mkdir()
    second_runtime.write_text(
        json.dumps({"worker_id": "maid-a", "api_token": TOKEN}),
        encoding="utf-8",
    )
    controller = MaidBellController(config, popen_factory=RecordingPopen())
    controller.handle_request(_request(runtime))

    with pytest.raises(BellBusyError, match="already owns another runtime"):
        controller.handle_request(_request(second_runtime))


def test_bell_config_rejects_entrypoint_outside_repository(tmp_path: Path) -> None:
    with pytest.raises(BellError, match="inside repo_root"):
        MaidBellConfig.from_mapping(
            {
                "schema_version": 1,
                "device_id": "maid-a",
                "listen_host": "127.0.0.1",
                "port": 8766,
                "repo_root": str(tmp_path / "repo"),
                "python_path": sys.executable,
                "maid_entrypoint": str(tmp_path / "outside.py"),
            }
        )


def test_bell_tcp_ping_and_wake_round_trip(tmp_path: Path) -> None:
    config, runtime = _bell_files(tmp_path)
    config = replace(config, port=0)
    controller = MaidBellController(config, popen_factory=RecordingPopen())
    server = MaidBellServer(config, controller=controller)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = MaidBellClient(
        "127.0.0.1",
        server.server_address[1],
        timeout=2.0,
    )
    try:
        assert client.ping()["status"] == "idle"
        response = client.wake(
            device_id="maid-a",
            runtime_config_path=str(runtime),
            api_token=TOKEN,
        )
        assert response["status"] == "started"
        assert client.ping()["pid"] == response["pid"]
    finally:
        server.shutdown()
        server.shutdown()
        thread.join(timeout=2.0)
    assert not thread.is_alive()
