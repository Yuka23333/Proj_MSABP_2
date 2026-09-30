from __future__ import annotations

import pandas as pd

from scripts.postprocessing import fit_legacy_roi_validate_batch03 as legacy_roi


def test_combined_ber_thresholds_rejects_duplicate_cases(monkeypatch) -> None:
    monkeypatch.setattr(
        legacy_roi,
        "_ber_thresholds",
        lambda path: {"same_case": float(len(str(path)))},
    )

    try:
        legacy_roi._combined_ber_thresholds(
            [legacy_roi.Path("first.csv"), legacy_roi.Path("second.csv")]
        )
    except ValueError as exc:
        assert "duplicate legacy BER case" in str(exc)
    else:
        raise AssertionError("duplicate BER case was accepted")


def test_spearman_uses_higher_is_better_order() -> None:
    increasing = pd.Series([1.0, 2.0, 3.0, 4.0])
    decreasing = pd.Series([4.0, 3.0, 2.0, 1.0])

    assert legacy_roi._spearman(increasing, increasing) == 1.0
    assert legacy_roi._spearman(increasing, decreasing) == -1.0
