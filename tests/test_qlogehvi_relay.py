from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from msabp_opt.optimization import proposal_relay, qlogehvi
from msabp_opt.simulation.distributed.config import DeviceConfig, LaunchMode


def _input_space() -> qlogehvi.InputSpace:
    return qlogehvi.InputSpace(
        names=("x", "y"),
        lower=np.asarray([0.0, 10.0]),
        upper=np.asarray([2.0, 20.0]),
    )


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "x": 0.0,
                "y": 10.0,
                "is_penalty": False,
                qlogehvi.WORST_S11_COLUMN: 0.5,
                qlogehvi.MEAN_TOT_EFF_COLUMN: 0.7,
                qlogehvi.AREA_COLUMN: 100.0,
            },
            {
                "x": 2.0,
                "y": 20.0,
                "is_penalty": True,
                qlogehvi.WORST_S11_COLUMN: 1.0,
                qlogehvi.MEAN_TOT_EFF_COLUMN: 0.0,
                qlogehvi.AREA_COLUMN: 200.0,
            },
        ]
    )


def _result() -> qlogehvi.ProposalResult:
    return qlogehvi.ProposalResult(
        unit_values=np.asarray([[0.25, 0.75]]),
        raw_values=np.asarray([[0.5, 17.5]]),
        acquisition_values=(1.25,),
        diagnostics={"device": "cuda", "dtype": "float64"},
    )


def test_request_contains_only_compact_training_arrays() -> None:
    payload = proposal_relay.build_request_payload(
        _observations(),
        _input_space(),
        settings=qlogehvi.ProposalSettings(q=1),
        iteration=3,
        compute_device="cuda",
    )

    assert payload["iteration"] == 3
    assert payload["compute"] == {"device": "cuda", "dtype": "float64"}
    assert len(payload["implementation"]["qlogehvi_source_sha256"]) == 64
    assert len(payload["implementation"]["proposal_relay_source_sha256"]) == 64
    assert payload["training"]["x_unit"] == [[0.0, 0.0], [1.0, 1.0]]
    assert payload["training"]["summary"]["penalty_observations"] == 1
    serialized = json.dumps(payload)
    assert "source_root" not in serialized
    assert "S11.csv" not in serialized


def test_run_request_passes_cuda_and_float64_arrays(monkeypatch) -> None:
    request = proposal_relay.build_request_payload(
        _observations(),
        _input_space(),
        settings=qlogehvi.ProposalSettings(q=1),
        iteration=4,
        compute_device="cuda:0",
    )
    captured: dict[str, object] = {}

    def fake_propose(train_x, train_y_rf, train_y_full, input_space, **kwargs):
        captured.update(
            train_x=train_x,
            train_y_rf=train_y_rf,
            train_y_full=train_y_full,
            input_space=input_space,
            kwargs=kwargs,
        )
        return _result()

    monkeypatch.setattr(
        qlogehvi,
        "propose_qlogehvi_from_training_arrays",
        fake_propose,
    )

    result = proposal_relay.run_request_payload(request)

    assert result.diagnostics["device"] == "cuda"
    assert captured["kwargs"]["device_name"] == "cuda:0"
    assert captured["kwargs"]["iteration"] == 4
    assert captured["train_x"].dtype == np.float64


def test_request_rejects_different_worker_implementation() -> None:
    request = proposal_relay.build_request_payload(
        _observations(),
        _input_space(),
        settings=qlogehvi.ProposalSettings(q=1),
        iteration=0,
    )
    request["implementation"]["qlogehvi_source_sha256"] = "0" * 64

    try:
        proposal_relay.run_request_payload(request)
    except RuntimeError as exc:
        assert "same Git commit" in str(exc)
    else:
        raise AssertionError("mismatched worker implementation was accepted")


def test_request_file_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    proposal_relay.write_request(request_path, {"schema_version": 1})
    calls = 0

    def fake_run(_payload):
        nonlocal calls
        calls += 1
        return _result()

    monkeypatch.setattr(proposal_relay, "run_request_payload", fake_run)

    assert not proposal_relay.execute_request_file(request_path, response_path)
    assert proposal_relay.execute_request_file(request_path, response_path)
    assert calls == 1


def test_remote_relay_uses_hash_addressed_files_and_bocuda(
    monkeypatch,
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    proposal_relay.write_request(request_path, {"schema_version": 1})
    request_sha = proposal_relay.sha256_file(request_path)
    device = DeviceConfig(
        id="coconutg2",
        enabled=True,
        launch_mode=LaunchMode.SSH_PROCESS,
        ssh_target="telecom@coconutg2",
        repo_root=r"D:\Academic\Proj_MSABP_2",
        python_path=r"C:\Users\telecom\miniforge3\envs\maid\python.exe",
    )
    remote = proposal_relay.RemoteProposalConfig(timeout_seconds=123.0)
    calls: dict[str, object] = {}

    def fake_push(_device, _local, destination, **_kwargs):
        calls["request_destination"] = destination

    def fake_run(_device, script, **kwargs):
        calls["script"] = script
        calls["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess([], 0, "worker ok\n", "")

    def fake_pull(_device, source, local, **_kwargs):
        calls["response_source"] = source
        payload = proposal_relay.response_payload(request_sha, _result())
        Path(local).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(proposal_relay, "push_file_atomic", fake_push)
    monkeypatch.setattr(proposal_relay, "run_remote_powershell", fake_run)
    monkeypatch.setattr(proposal_relay, "pull_file_atomic", fake_pull)

    result = proposal_relay.relay_remote_proposal(
        device=device,
        remote=remote,
        plan_id="gpu-plan",
        batch_index=2,
        local_request_path=request_path,
        local_response_path=response_path,
        expected_q=1,
        expected_dimension=2,
    )

    assert request_sha[:16] in str(calls["request_destination"])
    assert request_sha[:16] in str(calls["response_source"])
    assert remote.python_path in str(calls["script"])
    assert "qlogehvi_gpu_worker.py" in str(calls["script"])
    assert calls["timeout"] == 123.0
    assert result.diagnostics["proposal_executor"] == "remote_ssh"
    assert result.diagnostics["proposal_device_id"] == "coconutg2"
