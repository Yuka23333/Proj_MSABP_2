from __future__ import annotations

import numpy as np

from scripts.simulation.prepare_propagation_active_learning_12_batch01 import (
    _greedy_select,
    _nondominated_mask,
    _ridge_fit,
    _ridge_predict,
)


def test_nondominated_mask_uses_minimization_semantics() -> None:
    objectives = np.asarray([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]])
    assert np.array_equal(_nondominated_mask(objectives), [True, False, True])


def test_ridge_fit_preserves_simple_linear_order() -> None:
    x = np.arange(6.0)[:, None]
    y = 2.0 + 3.0 * x[:, 0]
    prediction = _ridge_predict(x, _ridge_fit(x, y, alpha=0.01))
    assert np.corrcoef(prediction, y)[0, 1] > 0.999


def test_greedy_select_is_deterministic_and_unique() -> None:
    selected = _greedy_select(
        [1, 2, 3],
        2,
        base_values=np.asarray([0.0, 0.3, 0.8, 0.7]),
        selection_space=np.asarray([[0.0], [0.2], [0.7], [1.0]]),
        anchor_indices=[0],
        diversity_weight=0.2,
    )
    assert selected == [2, 3]
