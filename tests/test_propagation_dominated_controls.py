from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from msabp_opt.optimization import krvea_data
from scripts.geometry import shapely_antenna_model
from scripts.simulation import prepare_propagation_dominated_controls as controls


def _synthetic_observations(count: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(20260905)
    defaults = asdict(shapely_antenna_model.DEFAULT_PARAMETERS)
    space = krvea_data.authoritative_input_space()
    rows = []
    for index in range(count):
        unit = np.full(len(space.names), (index + 0.5) / count)
        parameters = defaults | space.values(space.denormalize(unit))
        objectives = rng.uniform(0.05, 0.95, size=4)
        if index == 0:
            objectives[:] = 0.01
        row = {
            "source": "synthetic",
            "case_id": f"case_{index:04d}",
            "case_directory": f"synthetic/case_{index:04d}",
            "status": "completed",
            "is_penalty": False,
            "worst_s11_linear_amplitude": objectives[0],
            "one_minus_mean_total_efficiency_linear": objectives[1],
            "mean_total_efficiency_linear": 1.0 - objectives[1],
            "normalized_substrate_area": objectives[2] + 0.5,
            "cap_realized_gain_linear": objectives[3] + 0.05,
            "cap_realized_gain_dbi": 10.0 * np.log10(objectives[3] + 0.05),
            **parameters,
        }
        rows.append(row)
    rows[-1]["status"] = "penalized"
    rows[-1]["is_penalty"] = True
    return pd.DataFrame(rows)


def test_selects_interpretable_unique_completed_controls() -> None:
    archive = controls.prepare_archive(_synthetic_observations())
    selected = controls.select_representative_controls(archive, count=8)

    assert len(selected) == 8
    assert selected["case_id"].is_unique
    assert selected["is_dominated"].all()
    assert selected["selection_role"].tolist()[:5] == [
        "bad_s11",
        "bad_efficiency",
        "bad_area",
        "bad_cap_gain",
        "bad_overall",
    ]
    assert all(value > 0 for value in selected["dominated_by_count"])


def test_worklist_rows_validate_as_propagation_geometry() -> None:
    archive = controls.prepare_archive(_synthetic_observations())
    selected = controls.select_representative_controls(archive, count=8)

    rows = controls.build_worklist_rows(selected)

    assert len(rows) == 8
    assert rows[0]["sample_id"].startswith("bad_01_bad_s11")
    assert all(row["simulation_mode"] == "propagation_s21" for row in rows)
    assert all(row["geometry_valid"] == "True" for row in rows)
