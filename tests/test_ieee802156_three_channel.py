from __future__ import annotations

import numpy as np
import pytest

from scripts.postprocessing import ieee802156_three_channel as channels


def test_ieee802156_low_band_channel_plan_is_exact() -> None:
    assert [item.center_frequency_hz / 1e6 for item in channels.LOW_BAND_CHANNELS] == [
        3494.4,
        3993.6,
        4492.8,
    ]
    assert all(
        item.bandwidth_hz / 1e6 == pytest.approx(499.2)
        for item in channels.LOW_BAND_CHANNELS
    )
    assert channels.LOW_BAND_CHANNELS[0].lower_frequency_hz / 1e6 == pytest.approx(
        3244.8
    )
    assert channels.LOW_BAND_CHANNELS[2].upper_frequency_hz / 1e6 == pytest.approx(
        4742.4
    )
    assert channels.LOW_BAND_CHANNELS[1].mandatory


def test_mandatory_worst_utility_implements_requested_formula() -> None:
    utilities = np.asarray([[0.8, 0.9, 0.7], [0.2, 0.1, 0.3]])

    score = channels.aggregate_mandatory_worst_utility(
        utilities,
        0.25,
        axis=1,
    )

    np.testing.assert_allclose(score, [0.75 * 0.9 + 0.25 * 0.7, 0.1])


def test_loss_form_is_exact_complement_for_ber() -> None:
    ber = np.asarray([0.02, 0.01, 0.03])
    success_score = channels.aggregate_mandatory_worst_utility(1.0 - ber, 0.4)
    robust_ber = channels.aggregate_mandatory_worst_loss(ber, 0.4)

    assert robust_ber == pytest.approx(1.0 - success_score)
    assert robust_ber == pytest.approx(0.6 * 0.01 + 0.4 * 0.03)


def test_quadratic_frequency_average_is_normalized_and_symmetric() -> None:
    channel = channels.LOW_BAND_CHANNELS[1]
    frequency = np.linspace(channel.lower_frequency_hz, channel.upper_frequency_hz, 51)

    assert channels.quadratic_weighted_mean(
        frequency,
        np.full(frequency.shape, 3.25),
        channel,
    ) == pytest.approx(3.25)
    assert channels.quadratic_weighted_mean(
        frequency,
        frequency - channel.center_frequency_hz,
        channel,
    ) == pytest.approx(0.0, abs=1e-7)


def test_quadratic_average_inserts_center_for_sparse_covering_curve() -> None:
    channel = channels.LOW_BAND_CHANNELS[0]

    result = channels.quadratic_weighted_mean(
        [3.0e9, 4.0e9],
        [2.0, 2.0],
        channel,
    )

    assert result == pytest.approx(2.0)


def test_invalid_lambda_or_incomplete_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        channels.aggregate_mandatory_worst_utility([1.0, 1.0, 1.0], 1.1)
    with pytest.raises(ValueError, match="does not cover"):
        channels.quadratic_weighted_mean(
            [3.8e9, 4.0e9, 4.2e9],
            [1.0, 1.0, 1.0],
            channels.LOW_BAND_CHANNELS[1],
        )
