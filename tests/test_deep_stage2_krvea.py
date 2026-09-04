from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.optimization import run_krvea
from scripts.optimization import 深度优化_阶段2_krvea as stage2
from scripts.optimization import 深度优化_阶段2_relay as stage2_relay


def test_stage2_campaign_is_a_separate_64_point_arm() -> None:
    config, policy = stage2.build_config()

    assert config.plan_id == "msabp-krvea-11var-stage2-learned-64-006"
    assert config.output_directory.name == config.plan_id
    assert config.total_budget == 64
    assert config.proposal.q == 4
    assert len(config.source_directories) == 6
    assert config.source_directories[-1].name == "msabp-krvea-11var-deep-64-005"
    assert config.strategy_name == stage2.STRATEGY_NAME
    assert config.strategy_source == Path(stage2.__file__).resolve()
    assert config.strategy_config_source == stage2.DEFAULT_CONFIG_PATH
    assert config.surrogate_settings.gp_noise_mode == "learned"
    assert config.surrogate_settings.uncertainty_calibration_factors == (
        1.0,
        1.0,
        1.0,
    )
    assert policy.physical_std_floor == pytest.approx(
        (0.05743483859200652, 0.0542825467769081, 0.2452539197516244)
    )


def test_stage2_reserves_one_exploration_candidate_in_every_batch() -> None:
    config, _ = stage2.build_config()
    settings = [
        run_krvea.proposal_settings_for_batch(config, batch_index=index, q=4)
        for index in range(4)
    ]

    assert [item.exploration_slots for item in settings] == [1, 1, 1, 1]
    assert [item.seed for item in settings] == [
        20260903,
        20260904,
        20260905,
        20260906,
    ]


def test_stage2_plan_fingerprints_new_strategy_and_full_json() -> None:
    config, _ = stage2.build_config()
    payload = run_krvea._plan_payload(
        config,
        run_krvea.krvea_data.authoritative_input_space(),
        {"sources": [], "trainable_count": 896},
    )

    assert payload["strategy"] == {
        "name": stage2.STRATEGY_NAME,
        "source": str(Path(stage2.__file__).resolve()),
        "config_source": str(stage2.DEFAULT_CONFIG_PATH),
    }
    assert payload["proposal"]["exploration_schedule"] == {
        "slots_per_exploration_batch": 1,
        "period_batches": 1,
        "first_exploration_batch_index": 0,
    }
    hashes = payload["software"]["implementation_sha256"]
    assert "strategy_source" in hashes
    assert "strategy_config" in hashes


def test_stage2_policy_is_embedded_with_worker_fingerprint() -> None:
    _, policy = stage2.build_config()
    payload = stage2_relay.attach_policy({"iteration": 0}, policy)
    wire = payload["stage2_policy"]

    assert wire["application_layer"] == (
        "physical_after_target_inverse_before_objective_scale"
    )
    assert len(wire["worker_source_sha256"]) == 64
    assert wire["physical_std_floor"] == {
        "worst_s11_linear_amplitude": pytest.approx(0.05743483859200652),
        "one_minus_mean_total_efficiency_linear": pytest.approx(
            0.0542825467769081
        ),
        "cap_realized_gain_dbi": pytest.approx(0.2452539197516244),
    }


def test_physical_floor_is_applied_per_objective() -> None:
    std = np.asarray([[0.01, 0.1, 0.2], [0.2, 0.01, 0.5]])
    guarded = stage2_relay.apply_physical_std_floor(std, (0.05, 0.06, 0.25))
    assert guarded == pytest.approx(
        np.asarray([[0.05, 0.1, 0.25], [0.2, 0.06, 0.5]])
    )


def test_stage2_json_rejects_unknown_policy_fields(tmp_path: Path) -> None:
    payload = json.loads(stage2.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["stage2_policy"]["mystery"] = 1
    path = tmp_path / "invalid-stage2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="policy fields mismatch"):
        stage2.build_config(config_path=path)
