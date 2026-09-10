from __future__ import annotations

import numpy as np

from scripts.postprocessing.search_link_spherical_roi import (
    region_feature,
    score_features,
    spherical_theta_cell_weights,
    symmetric_phi_mask,
)


def test_spherical_theta_cell_weights_cover_full_sphere() -> None:
    weights = spherical_theta_cell_weights(np.arange(0.0, 181.0, 5.0))
    assert np.all(weights > 0.0)
    assert np.isclose(weights.sum(), 2.0)


def test_symmetric_phi_mask_is_periodic_and_reaches_full_sphere() -> None:
    phi = np.arange(0.0, 360.0, 5.0)
    assert np.array_equal(np.flatnonzero(symmetric_phi_mask(phi, 90.0, 0.0)), [18])
    selected = phi[symmetric_phi_mask(phi, 90.0, 10.0)]
    assert np.array_equal(selected, [80.0, 85.0, 90.0, 95.0, 100.0])
    assert symmetric_phi_mask(phi, 90.0, 180.0).all()


def test_region_feature_preserves_constant_case_values() -> None:
    theta = np.arange(0.0, 181.0, 5.0)
    phi = np.arange(0.0, 360.0, 5.0)
    values = np.array([2.0, 7.0])
    field_map = np.broadcast_to(
        values[:, None, None, None],
        (2, 4, len(phi), len(theta)),
    )
    result = region_feature(
        field_map,
        theta_low=8,
        theta_high=24,
        phi_mask=symmetric_phi_mask(phi, 90.0, 35.0),
        frequency_interval=(1, 3),
        theta_weights=spherical_theta_cell_weights(theta),
    )
    assert np.allclose(result, values)


def test_score_features_orients_lower_ebn0_as_higher_is_better() -> None:
    feature = np.arange(6.0)
    scores = score_features(
        feature,
        s21_power=np.arange(6.0),
        ebn0_good=np.arange(6.0),
        reference_index=5,
    )
    assert np.isclose(scores["rho_s21_power"][0], 1.0)
    assert np.isclose(scores["rho_negative_ebn0"][0], 1.0)
    assert np.isclose(scores["robust_joint_rho"][0], 1.0)
