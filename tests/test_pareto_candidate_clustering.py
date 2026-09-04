from __future__ import annotations

import numpy as np

from scripts.postprocessing.cluster_pareto_candidates import (
    pam_kmedoids,
    silhouette_score_precomputed,
)


def test_pam_kmedoids_selects_one_medoid_per_separated_pair() -> None:
    points = np.asarray([0.0, 0.1, 10.0, 10.1])
    distances = np.abs(points[:, None] - points[None, :])

    medoids, labels, cost, _iterations = pam_kmedoids(distances, 2)

    assert len(medoids) == 2
    assert len(np.unique(labels)) == 2
    assert np.isclose(cost, 0.2)
    assert silhouette_score_precomputed(distances, labels) > 0.98
