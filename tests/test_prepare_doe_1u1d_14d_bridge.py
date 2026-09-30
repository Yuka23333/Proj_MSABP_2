from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.geometry import shapely_antenna_model
from scripts.optimization import prepare_doe_1u1d_14d_bridge as bridge


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "optimization"
    / "doe_1u1d_14d_bridge_64.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _parents() -> pd.DataFrame:
    defaults = asdict(shapely_antenna_model.DEFAULT_PARAMETERS)
    rows = []
    for slot in range(1, 17):
        rows.append(
            {
                "parent_slot": slot,
                "parent_group": "test",
                "parent_role": "test_parent",
                "source": "test_source",
                "case_id": f"parent_{slot:02d}",
                "parent_case_relpath": f"results/raw/test/case_parent_{slot:02d}",
                "parent_manifest_sha256": "a" * 64,
                **defaults,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_down_settings_form_a_four_point_latin_hypercube() -> None:
    config = _config()
    settings = bridge._down_settings(config)
    lower, upper = config["down_branch"]["active_range"]

    for name in bridge.DOWN_PARAMETER_NAMES:
        unit = np.asarray(
            [(setting[name] - lower) / (upper - lower) for setting in settings]
        )
        strata = np.floor(unit * len(settings)).astype(int)
        assert sorted(strata.tolist()) == list(range(len(settings)))


def test_inactive_down_parameters_do_not_change_exported_vertices() -> None:
    config = _config()
    base = asdict(shapely_antenna_model.DEFAULT_PARAMETERS)
    expected = bridge._checked_vertices(base, 0.01)

    for setting in bridge._down_settings(config):
        candidate = dict(base)
        candidate.update(
            BRANCH_DOWN_1_K=setting["BRANCH_DOWN_1_K"],
            BRANCH_DOWN_1_K2=setting["BRANCH_DOWN_1_K2"],
            BRANCH_DOWN_1_K3=0.0,
        )
        assert bridge._checked_vertices(candidate, 0.01) == expected


def test_worklist_is_complete_crossed_design() -> None:
    worklist, diagnostics = bridge.build_worklist(_parents(), _config())

    assert len(worklist) == 64
    assert worklist["sample_id"].is_unique
    assert worklist["geometry_valid"].all()
    assert set(worklist["simulation_mode"]) == {"antenna_characterization"}
    assert set(worklist["topology_id"]) == {"1u1d"}
    assert set(worklist["design_space_dimension"]) == {14}
    assert worklist.groupby("parent_slot").size().eq(4).all()
    assert worklist.groupby("down_setting_id").size().eq(16).all()
    assert (worklist["BRANCH_DOWN_1_K3"] > 0.0).all()
    assert diagnostics == {
        "active_geometry_preflight_count": 64,
        "active_geometry_valid_count": 64,
        "inactive_equivalence_check_count": 64,
    }


def test_local_surprise_detects_an_isolated_response_outlier() -> None:
    x = np.linspace(0.0, 1.0, 20)[:, None]
    x = np.hstack([x, x**2])
    objectives = np.column_stack([x[:, 0], 2.0 * x[:, 0]])
    objectives[10] += np.asarray([10.0, -10.0])

    scores = bridge._local_surprise_scores(x, objectives, neighbors=4)

    assert int(np.argmax(scores)) == 10
