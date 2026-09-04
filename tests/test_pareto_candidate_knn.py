from __future__ import annotations

import numpy as np
from shapely.geometry import box

from scripts.postprocessing.browse_pareto_candidate_knn import (
    geometric_jaccard_distance,
    nearest_neighbor_indices,
)


def test_geometric_jaccard_distance_is_symmetric() -> None:
    first = box(0.0, 0.0, 1.0, 1.0)
    second = box(0.5, 0.0, 1.5, 1.0)

    expected = 2.0 / 3.0
    assert geometric_jaccard_distance(first, first) == 0.0
    assert np.isclose(geometric_jaccard_distance(first, second), expected)
    assert np.isclose(geometric_jaccard_distance(second, first), expected)


def test_nearest_neighbor_indices_exclude_the_query() -> None:
    distances = np.asarray(
        [
            [0.0, 0.1, 0.5],
            [0.1, 0.0, 0.2],
            [0.5, 0.2, 0.0],
        ]
    )

    assert nearest_neighbor_indices(distances, 1).tolist() == [[1], [0], [1]]
