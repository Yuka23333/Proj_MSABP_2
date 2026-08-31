from __future__ import annotations

from scripts.optimization import run_krvea
from scripts.optimization import 深度优化_krvea as deep_krvea


def test_deep_config_is_isolated_from_baseline_policy() -> None:
    baseline_config = run_krvea.CampaignConfig(
        plan_id="baseline-test",
        source_directories=(deep_krvea.F5_SOURCE_DIRECTORIES[0],),
        output_directory=deep_krvea.REPOSITORY_ROOT / "baseline-test",
    )
    deep_config = deep_krvea.build_config()

    assert baseline_config.surrogate_settings is None
    assert baseline_config.exploration_period_batches == 1
    assert deep_config.strategy_name == "deep_late_stage_s11_guard_v1"
    assert deep_config.strategy_source == deep_krvea.Path(
        deep_krvea.__file__
    ).resolve()
    assert len(deep_config.source_directories) == 4
    assert deep_config.total_budget == 64
    assert deep_config.proposal.q == 4
    assert deep_config.surrogate_settings.uncertainty_calibration_factors == (
        2.5,
        1.1,
        1.25,
    )


def test_deep_exploration_is_reserved_every_other_batch() -> None:
    config = deep_krvea.build_config()

    even = run_krvea.proposal_settings_for_batch(config, batch_index=0, q=4)
    odd = run_krvea.proposal_settings_for_batch(config, batch_index=1, q=4)
    even_again = run_krvea.proposal_settings_for_batch(
        config,
        batch_index=2,
        q=4,
    )

    assert even.exploration_slots == 1
    assert odd.exploration_slots == 0
    assert even_again.exploration_slots == 1
    assert (even.seed, odd.seed, even_again.seed) == (
        deep_krvea.F5_SEED,
        deep_krvea.F5_SEED + 1,
        deep_krvea.F5_SEED + 2,
    )


def test_deep_plan_records_policy_and_strategy_hash() -> None:
    config = deep_krvea.build_config()
    snapshot = {"sources": [], "trainable_count": 768}
    payload = run_krvea._plan_payload(
        config,
        run_krvea.krvea_data.authoritative_input_space(),
        snapshot,
    )

    assert payload["strategy"]["name"] == deep_krvea.STRATEGY_NAME
    assert payload["historical_training_count"] == 768
    assert payload["proposal"]["exploration_schedule"] == {
        "slots_per_exploration_batch": 1,
        "period_batches": 2,
        "first_exploration_batch_index": 0,
    }
    assert payload["proposal"]["surrogate_settings"][
        "uncertainty_calibration_factors"
    ] == (2.5, 1.1, 1.25)
    assert "strategy_source" in payload["software"]["implementation_sha256"]
