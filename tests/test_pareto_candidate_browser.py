from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.geometry.shapely_antenna_model import DEFAULT_PARAMETERS
from scripts.postprocessing.browse_pareto_candidates import (
    build_candidate_shortlist,
    nondominated_mask,
)


def test_nondominated_mask_uses_strict_all_minimization_dominance() -> None:
    objectives = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [0.5, 3.0, 3.0, 3.0],
        ]
    )

    assert nondominated_mask(objectives).tolist() == [True, True, False, True]


def test_shortlist_has_unique_output_columns() -> None:
    row = {
        "status": "completed",
        "is_penalty": False,
        "source": "test",
        "case_id": "case_1",
        "case_directory": "test/case_1",
        "worst_s11_linear_amplitude": 0.1,
        "one_minus_mean_total_efficiency_linear": 0.2,
        "normalized_substrate_area": 1.0,
        "cap_realized_gain_linear": 1.5,
        "mean_total_efficiency_linear": 0.8,
        "cap_realized_gain_dbi": 1.7609,
    }
    row.update(DEFAULT_PARAMETERS.__dict__)

    shortlist, front_size = build_candidate_shortlist(pd.DataFrame([row]))

    assert front_size == 1
    assert len(shortlist) == 1
    assert shortlist.columns.is_unique
    assert shortlist["normalized_substrate_area"].tolist() == [1.0]
