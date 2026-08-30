from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from msabp_opt.optimization import krvea, krvea_relay
from msabp_opt.simulation.distributed.config import DeviceConfig, LaunchMode


NAMES = (
    "SLOT_MAIN_LENGTH",
    "PATCH_BRICK_1_SIDE_MARGIN",
    "PATCH_BRICK_1_TOP_MARGIN",
    "PATCH_BRICK_2_HEIGHT_MARGIN",
    "UPPER_CORNER_NOTCH_1_K1",
    "UPPER_CORNER_NOTCH_1_K2",
    "UPPER_CORNER_EAR_1_K1",
    "UPPER_CORNER_EAR_1_K2",
    "BRANCH_UP_1_K",
    "BRANCH_UP_1_K2",
    "BRANCH_UP_1_K3",
)


def _space() -> SimpleNamespace:
    return SimpleNamespace(
        names=NAMES,
        lower=np.asarray([47.7, 5.4, 2.34, 13.5, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05]),
        upper=np.asarray([58.3, 6.6, 2.86, 16.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )


def _training() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(
        [
            [0.0] * 11,
            [0.5] * 11,
            [1.0] * 11,
        ],
        dtype=np.float64,
    )
    full = np.asarray(
        [
            [0.4, 0.2, 0.9, -3.0],
            [0.6, 0.3, 1.0, -1.0],
            [99.0, 99.0, 99.0, 99.0],
        ],
        dtype=np.float64,
    )
    expensive = full[:, krvea_relay.EXPENSIVE_OBJECTIVE_INDICES]
    penalty = np.asarray([False, False, True])
    return x, expensive, full, penalty


def _request() -> dict[str, object]:
    x, expensive, full, penalty = _training()
    return krvea_relay.build_request_payload(
        x,
        expensive,
        full,
        penalty,
        _space(),
        config=krvea.KRVEAConfig(
            n_variables=11,
            n_objectives=4,
            q=2,
            inner_evaluations=0,
            seed=17,
        ),
        iteration=3,
        remaining_expensive_budget=2,
        previous_empty_reference_count=40,
        compute_device="cuda",
    )


def _result(q: int = 2) -> krvea_relay.ProposalResult:
    unit = np.asarray([[0.2] * 11, [0.8] * 11], dtype=np.float64)[:q]
    space = krvea_relay._wire_input_space(_space())
    mean = np.asarray([[0.5, 0.2, 1.0, -2.0], [0.6, 0.3, 1.1, -1.0]])[:q]
    std = np.asarray([[0.1, 0.1, 0.0, 0.2], [0.1, 0.1, 0.0, 0.2]])[:q]
    return krvea_relay.ProposalResult(
        unit_values=unit,
        raw_values=space.denormalize(unit),
        predicted_mean=mean,
        predicted_std=std,
        predicted_mean_standardized=mean.copy(),
        predicted_std_standardized=std.copy(),
        diagnostics={"device": "cuda", "dtype": "float64"},
    )


def test_request_is_compact_and_has_explicit_objective_contract() -> None:
    payload = _request()

    assert payload["algorithm"] == "K-RVEA"
    assert payload["compute"] == {"device": "cuda", "dtype": "float64"}
    assert payload["objective_contract"]["names"] == list(krvea_relay.OBJECTIVE_NAMES)
    assert payload["objective_contract"]["expensive_indices"] == [0, 1, 3]
    assert payload["objective_contract"]["exact_indices"] == [2]
    assert payload["objective_contract"]["penalty_rows_in_gp_fit"] is False
    assert payload["surrogate_settings"]["gp_kernel"] == "matern_5_2"
    assert payload["surrogate_settings"]["uncertainty_calibration_factors"] == (
        1.1,
        1.1,
        1.25,
    )
    assert payload["objective_contract"]["exact_objective"]["reference_area_mm2"] == pytest.approx(2720.2)
    assert payload["previous_empty_reference_count"] == 40
    assert payload["training"]["summary"]["penalty_observations"] == 1
    assert len(payload["implementation"]["krvea_source_sha256"]) == 64
    serialized = json.dumps(payload)
    assert "Farfield Source" not in serialized
    assert "source_root" not in serialized


def test_robust_scaler_ignores_penalty_and_makes_it_bad_in_every_dimension() -> None:
    _, _, full, penalty = _training()
    scaler = krvea_relay.fit_objective_scaler(full, penalty)
    standardized = scaler.transform(full, is_penalty=penalty)

    assert np.allclose(scaler.center, [0.5, 0.25, 0.95, -2.0])
    assert np.all(
        standardized[penalty][0] > np.max(standardized[~penalty], axis=0)
    )
    assert scaler.to_dict()["objective_names"] == list(krvea_relay.OBJECTIVE_NAMES)


def test_robust_scaler_is_safe_for_constant_successful_columns() -> None:
    values = np.asarray(
        [[0.5, 0.2, 1.0, -2.0], [0.5, 0.2, 1.0, -2.0], [9.0] * 4]
    )
    penalty = np.asarray([False, False, True])

    scaler = krvea_relay.fit_objective_scaler(values, penalty)
    standardized = scaler.transform(values, is_penalty=penalty)

    assert np.allclose(scaler.scale, 1.0)
    assert np.isfinite(standardized).all()
    assert np.all(standardized[-1] > standardized[:-1])


def test_surrogate_target_scaler_is_bounded_and_round_trips_zero_variance() -> None:
    physical = np.asarray(
        [
            [0.4, 0.2, -3.0],
            [0.6, 0.3, -1.0],
            [0.8, 0.7, 2.0],
        ],
        dtype=np.float64,
    )
    scaler = krvea_relay.SurrogateTargetScaler.fit(physical)
    model_values = scaler.transform(physical)

    recovered_mean, recovered_std = scaler.inverse_prediction(
        model_values,
        np.zeros_like(model_values),
    )
    wide_mean, wide_std = scaler.inverse_prediction(
        np.asarray([[-100.0, 100.0, 0.0]]),
        np.asarray([[10.0, 10.0, 1.0]]),
    )

    assert np.allclose(recovered_mean, physical, atol=1e-12)
    assert np.all(recovered_std == 0.0)
    assert np.all((wide_mean[:, :2] >= 0.0) & (wide_mean[:, :2] <= 1.0))
    assert np.all((wide_std[:, :2] >= 0.0) & (wide_std[:, :2] <= 0.5))


def test_worker_excludes_penalty_rows_from_gp_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    request["compute"]["device"] = "cpu"
    captured: dict[str, np.ndarray] = {}

    def fake_fit(train_x, train_y, **_kwargs):
        captured["x"] = np.asarray(train_x)
        captured["y"] = np.asarray(train_y)

        def predictor(unit_values):
            count = len(unit_values)
            return krvea.SurrogatePrediction(
                mean=np.zeros((count, 3)),
                std=np.full((count, 3), 0.1),
            )

        return predictor, {"device": "cpu", "dtype": "float64"}

    monkeypatch.setattr(
        krvea_relay,
        "_fit_surrogate_predictor",
        fake_fit,
    )

    result = krvea_relay.run_request_payload(request)

    assert captured["x"].shape == (2, 11)
    assert captured["y"].shape == (2, 3)
    assert result.diagnostics["training"]["gp_training_observations"] == 2
    assert (
        result.diagnostics["training"]["penalty_observations_excluded_from_gp"]
        == 1
    )
    assert np.all(
        (result.predicted_mean[:, :2] >= 0.0)
        & (result.predicted_mean[:, :2] <= 1.0)
    )


def test_exact_area_contract_is_one_at_reference_dimensions() -> None:
    space = krvea_relay._wire_input_space(_space())
    contract = krvea_relay._exact_area_contract(space)
    raw = np.asarray([53.0, 6.0, 2.6, 15.0] + [0.5] * 7)
    unit = (raw - space.lower) / (space.upper - space.lower)

    area = krvea_relay.exact_normalized_area(unit[None, :], space, contract)

    assert area.shape == (1, 1)
    assert area[0, 0] == pytest.approx(1.0)


def test_worker_rejects_different_implementation_before_importing_botorch() -> None:
    request = _request()
    request["implementation"]["krvea_source_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="same Git commit"):
        krvea_relay.run_request_payload(request)


def test_request_file_is_idempotent_without_botorch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    krvea_relay.write_request(request_path, {"schema_version": 1})
    calls = 0

    def fake_run(_payload):
        nonlocal calls
        calls += 1
        return _result(1)

    monkeypatch.setattr(krvea_relay, "run_request_payload", fake_run)

    assert not krvea_relay.execute_request_file(request_path, response_path)
    assert krvea_relay.execute_request_file(request_path, response_path)
    assert calls == 1


def test_response_validation_rejects_duplicates_and_observed_points() -> None:
    request_hash = "a" * 64
    payload = krvea_relay.response_payload(request_hash, _result())
    validated = krvea_relay.result_from_response(
        payload,
        expected_request_sha256=request_hash,
        expected_q=2,
        expected_dimension=11,
        input_space=_space(),
    )
    assert validated.unit_values.shape == (2, 11)
    assert np.allclose(validated.predicted_std[:, 2], 0.0)

    duplicate_payload = json.loads(json.dumps(payload))
    duplicate_payload["result"]["unit_values"][1] = duplicate_payload["result"]["unit_values"][0]
    duplicate_payload["result"]["raw_values"][1] = duplicate_payload["result"]["raw_values"][0]
    with pytest.raises(ValueError, match="duplicate"):
        krvea_relay.result_from_response(
            duplicate_payload,
            expected_request_sha256=request_hash,
            expected_q=2,
            expected_dimension=11,
        )

    with pytest.raises(ValueError, match="observed"):
        krvea_relay.result_from_response(
            payload,
            expected_request_sha256=request_hash,
            expected_q=2,
            expected_dimension=11,
            observed_x_unit=np.asarray([[0.2] * 11]),
        )


def test_remote_relay_uses_hash_addressed_files_bocuda_and_krvea_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    krvea_relay.write_request(request_path, {"schema_version": 1})
    request_sha = krvea_relay.sha256_file(request_path)
    device = DeviceConfig(
        id="coconutg2",
        enabled=True,
        launch_mode=LaunchMode.SSH_PROCESS,
        ssh_target="telecom@coconutg2",
        repo_root=r"D:\Academic\Proj_MSABP_2",
        python_path=r"C:\Users\telecom\miniforge3\envs\maid\python.exe",
    )
    remote = krvea_relay.RemoteProposalConfig(timeout_seconds=123.0)
    calls: dict[str, object] = {}

    def fake_push(_device, _local, destination, **_kwargs):
        calls["request_destination"] = destination

    def fake_run(_device, script, **kwargs):
        calls["script"] = script
        calls["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess([], 0, "worker ok\n", "")

    def fake_pull(_device, source, local, **_kwargs):
        calls["response_source"] = source
        Path(local).write_text(
            json.dumps(krvea_relay.response_payload(request_sha, _result())),
            encoding="utf-8",
        )

    monkeypatch.setattr(krvea_relay, "push_file_atomic", fake_push)
    monkeypatch.setattr(krvea_relay, "run_remote_powershell", fake_run)
    monkeypatch.setattr(krvea_relay, "pull_file_atomic", fake_pull)

    result = krvea_relay.relay_remote_proposal(
        device=device,
        remote=remote,
        plan_id="krvea-smoke",
        batch_index=2,
        local_request_path=request_path,
        local_response_path=response_path,
        expected_q=2,
        expected_dimension=11,
        input_space=_space(),
    )

    assert request_sha[:16] in str(calls["request_destination"])
    assert request_sha[:16] in str(calls["response_source"])
    assert remote.python_path in str(calls["script"])
    assert "krvea_gpu_worker.py" in str(calls["script"])
    assert calls["timeout"] == 123.0
    assert result.diagnostics["proposal_executor"] == "remote_ssh"
    assert result.diagnostics["proposal_device_id"] == "coconutg2"
