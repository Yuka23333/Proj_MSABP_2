from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.simulation import prepare_propagation_active_learning_6_batch03 as batch03


def test_select_candidates_keeps_endpoints_and_is_deterministic() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "case_id": f"case_{index}",
                "phase2_nondominated": True,
                "return_loss_db": 8.0,
                "worst_s11_linear_amplitude": 0.2 + 0.02 * index,
                "negative_roi_theta_radiation_gain_dbi": 8.0 - 0.7 * index,
                "normalized_substrate_area": 0.9 + 0.03 * abs(index - 3),
            }
        )
    frame = pd.DataFrame(rows)

    first, diagnostics = batch03.select_candidates(frame)
    second, _ = batch03.select_candidates(frame)

    assert first["case_id"].tolist() == second["case_id"].tolist()
    assert len(first) == batch03.CANDIDATE_COUNT
    assert len(set(first["case_id"])) == batch03.CANDIDATE_COUNT
    for column in batch03.OBJECTIVE_COLUMNS:
        expected = frame.loc[frame[column].idxmin(), "case_id"]
        assert expected in set(first["case_id"])
    assert diagnostics["eligible_nondominated_count"] == len(frame)


def test_normalized_objectives_handles_constant_column() -> None:
    frame = pd.DataFrame(
        {
            "worst_s11_linear_amplitude": [0.2, 0.3],
            "negative_roi_theta_radiation_gain_dbi": [5.0, 5.0],
            "normalized_substrate_area": [0.9, 1.1],
        }
    )

    normalized = batch03._normalized_objectives(frame)

    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized[:, 1], 0.0)
