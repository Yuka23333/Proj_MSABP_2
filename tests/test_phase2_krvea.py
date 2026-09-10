from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from msabp_opt.optimization import krvea
from msabp_opt.optimization import phase2_krvea_data as phase2_data
from msabp_opt.optimization import phase2_krvea_relay as phase2_relay
from scripts.optimization import run_krvea as baseline
from scripts.optimization import run_phase2_krvea


def test_phase2_config_freezes_three_objectives_and_120_reference_vectors() -> None:
    config = run_phase2_krvea.build_config()

    assert config.proposal.n_variables == 11
    assert config.proposal.n_objectives == 3
    assert config.proposal.reference_partitions == 14
    assert math.comb(14 + 3 - 1, 3 - 1) == 120
    assert config.surrogate_settings.uncertainty_calibration_factors == (1.0, 1.0)
    assert phase2_data.OBJECTIVE_NAMES == (
        "worst_s11_linear_amplitude",
        "negative_roi_theta_radiation_gain_dbi",
        "normalized_substrate_area",
    )


def test_phase2_metric_contract_rejects_roi_drift(tmp_path: Path) -> None:
    payload = json.loads(run_phase2_krvea.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["phase2_metric"]["frequency_ghz"] = 3.7
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        run_phase2_krvea.load_config_document(path)
    except ValueError as exc:
        assert "metric contract is frozen" in str(exc)
    else:
        raise AssertionError("drifted ROI contract was accepted")


def test_phase2_controller_adapter_is_process_local() -> None:
    original_data = baseline.krvea_data
    original_request = baseline._request_remote_proposal

    with run_phase2_krvea._phase2_controller_adapter():
        assert baseline.krvea_data is phase2_data
        assert baseline._request_remote_proposal is run_phase2_krvea._request_remote_proposal

    assert baseline.krvea_data is original_data
    assert baseline._request_remote_proposal is original_request


def test_phase2_surrogate_transform_only_bounds_s11() -> None:
    physical = np.asarray(
        [[0.2, 8.0], [0.4, 12.0], [0.7, 16.0]],
        dtype=np.float64,
    )
    scaler = phase2_relay.SurrogateTargetScaler.fit(physical)
    model = scaler.transform(physical)
    restored_mean, restored_std = scaler.inverse_prediction(
        model,
        np.zeros_like(model),
    )

    np.testing.assert_allclose(restored_mean, physical, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(restored_std, 0.0, atol=1.0e-12, rtol=0.0)
    assert scaler.to_dict()["transforms"] == ["logit", "identity"]


def test_phase2_relay_request_has_two_gp_targets_and_exact_area() -> None:
    space = phase2_data.authoritative_input_space()
    x_unit = np.asarray([[0.2] * 11, [0.8] * 11], dtype=np.float64)
    full = np.asarray([[0.3, 8.0, 0.9], [0.4, 10.0, 1.1]], dtype=np.float64)
    config = krvea.KRVEAConfig(
        n_variables=11,
        n_objectives=3,
        reference_partitions=4,
        q=1,
        inner_evaluations=20,
        seed=7,
    )

    payload = phase2_relay.build_request_payload(
        x_unit,
        full[:, :2],
        full,
        np.asarray([False, False]),
        space,
        config=config,
        iteration=0,
        remaining_expensive_budget=1,
        previous_empty_reference_count=0,
        compute_device="cpu",
        surrogate_settings=phase2_relay.SurrogateFitSettings(),
    )

    assert payload["schema_version"] == 3
    assert payload["objective_contract"]["names"] == list(phase2_data.OBJECTIVE_NAMES)
    assert payload["objective_contract"]["expensive_indices"] == [0, 1]
    assert payload["objective_contract"]["exact_indices"] == [2]
    assert np.asarray(payload["training"]["y_expensive_minimize"]).shape == (2, 2)
    assert np.asarray(payload["training"]["y_full_minimize"]).shape == (2, 3)


def test_phase2_penalty_is_bad_in_all_three_objectives() -> None:
    config = run_phase2_krvea.build_config()
    space = phase2_data.authoritative_input_space()
    parameters = {
        **phase2_data.FIXED_PARAMETER_VALUES,
        **space.values(space.nominal),
    }

    penalty = run_phase2_krvea._penalty_objectives(config, parameters)

    assert penalty[phase2_data.WORST_S11_COLUMN] == 1.0
    assert (
        penalty[phase2_data.ROI_GAIN_LOSS_DBI_COLUMN]
        == phase2_data.PENALTY_ROI_GAIN_LOSS_DBI
    )
    assert (
        penalty[phase2_data.NORMALIZED_AREA_COLUMN]
        == phase2_data.PENALTY_NORMALIZED_AREA
    )
