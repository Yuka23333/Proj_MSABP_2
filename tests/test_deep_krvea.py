from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.optimization import run_krvea
from scripts.optimization import 深度优化_krvea as deep_krvea


def test_deep_json_config_is_isolated_from_baseline_policy() -> None:
    baseline_config = run_krvea.CampaignConfig(
        plan_id="baseline-test",
        source_directories=(deep_krvea.REPOSITORY_ROOT,),
        output_directory=deep_krvea.REPOSITORY_ROOT / "baseline-test",
    )
    deep_config = deep_krvea.build_config()

    assert baseline_config.surrogate_settings is None
    assert baseline_config.exploration_period_batches == 1
    assert baseline_config.strategy_config_source is None
    assert deep_config.strategy_name == deep_krvea.STRATEGY_NAME
    assert deep_config.strategy_source == Path(deep_krvea.__file__).resolve()
    assert deep_config.strategy_config_source == deep_krvea.DEFAULT_CONFIG_PATH
    assert len(deep_config.source_directories) == 5
    assert deep_config.plan_id == "msabp-krvea-11var-deep-64-005"
    assert deep_config.total_budget == 64
    assert deep_config.proposal.q == 4
    assert deep_config.proposal.seed == 20260902
    assert deep_config.surrogate_settings.uncertainty_calibration_factors == (
        4.75,
        3.05,
        1.85,
    )


def test_deep_exploration_is_reserved_every_fourth_batch() -> None:
    config = deep_krvea.build_config()
    slots = [
        run_krvea.proposal_settings_for_batch(config, batch_index=index, q=4)
        for index in range(5)
    ]

    assert [item.exploration_slots for item in slots] == [1, 0, 0, 0, 1]
    assert [item.seed for item in slots] == [
        20260902,
        20260903,
        20260904,
        20260905,
        20260906,
    ]


def test_deep_plan_records_strategy_and_json_hash() -> None:
    config = deep_krvea.build_config()
    snapshot = {"sources": [], "trainable_count": 832}
    payload = run_krvea._plan_payload(
        config,
        run_krvea.krvea_data.authoritative_input_space(),
        snapshot,
    )

    assert payload["strategy"] == {
        "name": deep_krvea.STRATEGY_NAME,
        "source": str(Path(deep_krvea.__file__).resolve()),
        "config_source": str(deep_krvea.DEFAULT_CONFIG_PATH),
    }
    assert payload["historical_training_count"] == 832
    assert payload["proposal"]["exploration_schedule"] == {
        "slots_per_exploration_batch": 1,
        "period_batches": 4,
        "first_exploration_batch_index": 0,
    }
    assert payload["proposal"]["surrogate_settings"][
        "uncertainty_calibration_factors"
    ] == (4.75, 3.05, 1.85)
    hashes = payload["software"]["implementation_sha256"]
    assert "strategy_source" in hashes
    assert "strategy_config" in hashes


def test_command_line_overrides_campaign_but_not_strategy() -> None:
    args = deep_krvea.parse_args(
        [
            "--config",
            str(deep_krvea.DEFAULT_CONFIG_PATH),
            "--plan-id",
            "override-plan",
            "--budget",
            "8",
            "--q",
            "2",
            "--device",
            "coconutg2",
        ]
    )
    config = deep_krvea.config_from_args(args)

    assert config.plan_id == "override-plan"
    assert config.total_budget == 8
    assert config.proposal.q == 2
    assert config.device_ids == ("coconutg2",)
    assert config.strategy_name == deep_krvea.STRATEGY_NAME
    assert config.surrogate_settings.uncertainty_calibration_factors == (
        4.75,
        3.05,
        1.85,
    )


def test_json_rejects_unknown_fields(tmp_path: Path) -> None:
    _, document = deep_krvea.load_config_document()
    modified = json.loads(json.dumps(document))
    modified["strategy"]["uncertainty_magic"] = 123
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(modified), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown.*uncertainty_magic"):
        deep_krvea.build_config(config_path=path)


def test_json_cannot_change_strategy_identity(tmp_path: Path) -> None:
    _, document = deep_krvea.load_config_document()
    modified = json.loads(json.dumps(document))
    modified["strategy"]["name"] = "different_algorithm"
    path = tmp_path / "wrong-strategy.json"
    path.write_text(json.dumps(modified), encoding="utf-8")

    with pytest.raises(ValueError, match="create another 深度优化"):
        deep_krvea.build_config(config_path=path)
