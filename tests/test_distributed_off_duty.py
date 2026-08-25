from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from msabp_opt.simulation.distributed.config import (
    DeviceConfig,
    DeviceRegistry,
    LaunchMode,
)
from msabp_opt.simulation.distributed.off_duty import dismiss_all_maids
from msabp_opt.simulation.distributed.state import PrincessState, freeze_csv


TOKEN = "off-duty-test-token-" + "x" * 32


class FakeBellClient:
    runtime_path = (
        r"D:\Academic\Proj_MSABP_2\simulations\runs\active-run"
        r"\workers\maid-a\maid_runtime.json"
    )
    stop_calls: list[dict[str, Any]] = []

    def __init__(self, _host: str, _port: int, *, timeout: float) -> None:
        self.timeout = timeout

    def ping(self) -> dict[str, Any]:
        return {
            "status": "running",
            "pid": 4321,
            "runtime_config_path": self.runtime_path,
        }

    def stop(self, **kwargs: Any) -> dict[str, Any]:
        self.stop_calls.append(dict(kwargs))
        return {"status": "stopped", "pid": 4321}


def _create_running_run(root: Path, run_id: str) -> None:
    run_root = root / run_id
    source = run_root / "source.csv"
    source.parent.mkdir(parents=True)
    source.write_text("sample_id,value\n0,1.0\n", encoding="utf-8")
    frozen = freeze_csv(source, run_root / "worklist.csv")
    with PrincessState(run_root / "princess.sqlite3") as state:
        state.initialize_run(run_id, frozen)
        state.register_worker(run_id, "maid-a")
        assert state.claim_next(run_id, "maid-a") is not None


def test_one_click_dismissal_only_touches_the_run_reported_by_bell(
    tmp_path: Path,
) -> None:
    FakeBellClient.stop_calls.clear()
    runs_root = tmp_path / "runs"
    _create_running_run(runs_root, "active-run")
    _create_running_run(runs_root, "unrelated-run")
    (runs_root / "active-run" / "princess_runtime.json").write_text(
        json.dumps(
            {
                "run_id": "active-run",
                "device_ids": ["maid-a"],
                "api_token": TOKEN,
            }
        ),
        encoding="utf-8",
    )
    device = DeviceConfig(
        id="maid-a",
        enabled=False,
        launch_mode=LaunchMode.BELL,
        ssh_target="telecom@maid-a",
        repo_root=r"D:\Academic\Proj_MSABP_2",
        python_path=r"C:\Users\telecom\miniforge3\envs\maid\python.exe",
        bell_host="maid-a",
    )
    registry = DeviceRegistry(
        bind_host="127.0.0.1",
        advertise_url="http://127.0.0.1:8765",
        port=8765,
        devices=(device,),
    )

    report = dismiss_all_maids(
        registry,
        runs_root=runs_root,
        bell_client_factory=FakeBellClient,
    )

    assert report.ok
    assert report.runs[0].run_id == "active-run"
    assert report.runs[0].released_case_ids == ("0",)
    assert FakeBellClient.stop_calls[0]["api_token"] == TOKEN
    with PrincessState(runs_root / "active-run" / "princess.sqlite3") as state:
        assert state.progress("active-run").pending == 1
        assert state.stop_request("active-run") is not None
        assert state.resume_run("active-run")
        state.wake_worker("active-run", "maid-a")
        retry = state.claim_next("active-run", "maid-a")
        assert retry is not None
        assert retry.attempt_number == 1
    with PrincessState(runs_root / "unrelated-run" / "princess.sqlite3") as state:
        assert state.progress("unrelated-run").running == 1
        assert state.stop_request("unrelated-run") is None
