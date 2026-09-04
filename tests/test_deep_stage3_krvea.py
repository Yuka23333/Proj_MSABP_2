from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from msabp_opt.optimization import krvea_relay
from scripts.optimization import run_krvea
from scripts.optimization import 深度优化_阶段3_krvea as stage3


def _pool_proposal(*, exploration_slots: int) -> krvea_relay.ProposalResult:
    count = 16
    unit = np.arange(count * 11, dtype=np.float64).reshape(count, 11)
    unit /= unit.max()
    raw = unit + 10.0
    mean = np.arange(count * 4, dtype=np.float64).reshape(count, 4)
    std = np.full((count, 4), 0.1)
    vectors = {
        "selected_reference_indices": list(range(100, 100 + count)),
        "selected_apd": list(np.linspace(0.1, 1.6, count)),
        "selected_mean_std": list(np.linspace(0.2, 1.7, count)),
        "selected_nearest_archive_distance": list(np.linspace(0.3, 1.8, count)),
        "selected_boundary_distance": list(np.linspace(0.4, 1.9, count)),
    }
    diagnostics = {
        **vectors,
        "requested_q": count,
        "proposed_count": count,
        "expensive_budget_remaining": count,
        "reserved_exploration_count": exploration_slots,
        "empty_reference_count": 3,
        "krvea": {
            **vectors,
            "requested_q": count,
            "proposed_count": count,
            "expensive_budget_remaining": count,
            "reserved_exploration_count": exploration_slots,
        },
    }
    return krvea_relay.ProposalResult(
        unit_values=unit,
        raw_values=raw,
        predicted_mean=mean,
        predicted_std=std,
        predicted_mean_standardized=mean / 2.0,
        predicted_std_standardized=std / 2.0,
        diagnostics=diagnostics,
    )


def test_stage3_configuration_releases_only_32_valid_proposals() -> None:
    config, uncertainty, pool = stage3.build_config()

    assert config.plan_id == "msabp-krvea-11var-stage3-feasible-32-007"
    assert config.total_budget == 32
    assert config.proposal.q == 4
    assert len(config.source_directories) == 7
    assert config.source_directories[-1].name == (
        "msabp-krvea-11var-stage2-learned-64-006"
    )
    assert config.strategy_name == stage3.STRATEGY_NAME
    assert config.strategy_source == Path(stage3.__file__).resolve()
    assert config.surrogate_settings.gp_noise_mode == "learned"
    assert uncertainty.physical_std_floor == pytest.approx(
        (0.05743483859200652, 0.0542825467769081, 0.2452539197516244)
    )
    assert pool.proposal_pool_size == 16
    assert pool.proposal_pool_exploration_slots == 4
    assert pool.rejected_candidates_consume_budget is False


def test_stage3_exploration_returns_every_fourth_batch() -> None:
    config, _, _ = stage3.build_config()
    settings = [
        run_krvea.proposal_settings_for_batch(config, batch_index=index, q=4)
        for index in range(5)
    ]
    assert [item.exploration_slots for item in settings] == [1, 0, 0, 0, 1]


def test_feasible_pool_preserves_three_plus_one_order(monkeypatch) -> None:
    proposal = _pool_proposal(exploration_slots=4)
    invalid = {0, 2, 5, 12}
    raw_to_index = {
        tuple(row): index for index, row in enumerate(proposal.raw_values)
    }

    def fake_preflight(raw, input_space, coordinate_quantum_mm):
        index = raw_to_index[tuple(raw)]
        return index not in invalid, (f"invalid-{index}" if index in invalid else None), {}

    monkeypatch.setattr(stage3, "_preflight_raw_candidate", fake_preflight)
    filtered = stage3.filter_feasible_pool(
        proposal,
        input_space=object(),
        desired_q=4,
        desired_exploration_slots=1,
        coordinate_quantum_mm=0.01,
        actual_remaining_budget=32,
        policy=stage3.FeasiblePoolPolicy(
            proposal_pool_size=16,
            proposal_pool_exploration_slots=4,
            selection_order="first_valid_preserve_exploitation_then_exploration",
            insufficient_valid_action="fail_before_campaign_state_mutation",
            rejected_candidates_consume_budget=False,
        ),
    )

    assert filtered.unit_values == pytest.approx(
        proposal.unit_values[[1, 3, 4, 13]]
    )
    assert filtered.diagnostics["reserved_exploration_count"] == 1
    assert filtered.diagnostics["selected_reference_indices"] == [101, 103, 104, 113]
    pool = filtered.diagnostics["feasibility_pool"]
    assert pool["selected_pool_indices"] == [1, 3, 4, 13]
    assert [item["pool_index"] for item in pool["rejected_infeasible"]] == [
        0,
        2,
        5,
        12,
    ]
    assert pool["rejected_candidates_consume_budget"] is False


def test_feasible_pool_fails_before_returning_short_batch(monkeypatch) -> None:
    proposal = _pool_proposal(exploration_slots=0)

    def fake_preflight(raw, input_space, coordinate_quantum_mm):
        index = int(np.where(np.all(proposal.raw_values == raw, axis=1))[0][0])
        return index < 3, (None if index < 3 else "invalid"), {}

    monkeypatch.setattr(stage3, "_preflight_raw_candidate", fake_preflight)
    _, _, policy = stage3.build_config()
    with pytest.raises(RuntimeError, match="no campaign state was changed"):
        stage3.filter_feasible_pool(
            proposal,
            input_space=object(),
            desired_q=4,
            desired_exploration_slots=0,
            coordinate_quantum_mm=0.01,
            actual_remaining_budget=32,
            policy=policy,
        )


def test_stage3_plan_fingerprints_new_strategy_and_json() -> None:
    config, _, _ = stage3.build_config()
    payload = run_krvea._plan_payload(
        config,
        run_krvea.krvea_data.authoritative_input_space(),
        {"sources": [], "trainable_count": 960},
    )
    assert payload["strategy"]["name"] == stage3.STRATEGY_NAME
    assert payload["proposal"]["exploration_schedule"]["period_batches"] == 4
    hashes = payload["software"]["implementation_sha256"]
    assert "strategy_source" in hashes
    assert "strategy_config" in hashes


def test_stage3_json_rejects_unknown_pool_fields(tmp_path: Path) -> None:
    payload = json.loads(stage3.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["stage3_policy"]["mystery"] = 1
    path = tmp_path / "invalid-stage3.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Stage-3 policy fields mismatch"):
        stage3.build_config(config_path=path)
