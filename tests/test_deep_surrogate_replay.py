from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "optimization" / "深度优化_阶段1代理回放.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("deep_surrogate_replay", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def replay():
    return _load_module()


def test_variant_settings_cover_fixed_and_learned_noise(replay) -> None:
    fixed = replay.variant_settings("fixed_1e-3")
    learned = replay.variant_settings("learned_floor_1e-4")
    assert fixed.gp_noise_mode == "fixed"
    assert fixed.gp_fixed_noise_variance == pytest.approx(1e-3)
    assert learned.gp_noise_mode == "learned"
    assert learned.gp_learned_noise_floor == pytest.approx(1e-4)
    assert fixed.uncertainty_calibration_factors == (1.0, 1.0, 1.0)


def test_rolling_archive_matches_saved_proposal_chronology(replay) -> None:
    _, observations, stages = replay.collect_rolling_archive()
    assert len(observations) == 896
    assert len(stages) == 32
    assert stages[0]["split"] == "round4_calibration"
    assert stages[0]["training_count"] == 768
    assert stages[0]["training_success_count"] == 760
    assert stages[16]["split"] == "round5_holdout"
    assert stages[16]["training_count"] == 832
    assert stages[-1]["training_count"] == 892
    assert sum(sum(stage["is_penalty"]) for stage in stages[:16]) == 5
    assert sum(sum(stage["is_penalty"]) for stage in stages[16:]) == 1


def test_ensemble_combines_between_model_disagreement(replay) -> None:
    base = {
        "split": "round4_calibration",
        "batch_index": 0,
        "case_id": "case",
        "is_penalty": False,
        "is_reserved_exploration": False,
        "objective_index": 0,
        "objective": replay.OBJECTIVE_NAMES[0],
        "actual": 0.8,
        "std": 0.1,
        "training_min": 0.2,
        "training_max": 0.6,
    }
    frame = pd.DataFrame(
        [{**base, "method": name, "mean": mean} for name, mean in zip(replay.VARIANTS, (0.1, 0.3, 0.5, 0.9), strict=True)]
    )
    combined = replay.add_ensemble_predictions(frame)
    median = combined.loc[combined["method"] == "ensemble_median"].iloc[0]
    clipped = combined.loc[combined["method"] == "ensemble_median_clipped"].iloc[0]
    assert median["mean"] == pytest.approx(0.4)
    assert median["std"] > 0.1
    assert clipped["mean"] == pytest.approx(0.4)


def test_physical_residual_floor_prevents_collapsed_guard(replay) -> None:
    rows = []
    for actual in (0.2, 0.4, 0.8, 0.9):
        rows.append(
            {
                    "method": "model",
                    "split": "round4_calibration",
                    "is_penalty": False,
                    "objective": replay.OBJECTIVE_NAMES[0],
                "actual": actual,
                "mean": 0.1,
                "std": 1e-12,
            }
        )
    calibration = replay.calibrate_residual_floors(pd.DataFrame(rows))
    floor = calibration["model"][replay.OBJECTIVE_NAMES[0]]["physical_std_floor"]
    assert np.isfinite(floor)
    assert floor > 0.1
