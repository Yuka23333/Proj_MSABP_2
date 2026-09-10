"""Shared IEEE 802.15.6 low-band three-channel metric definitions.

The channel allocation is fixed here so BER, S21, efficiency, and pattern
post-processing cannot silently drift onto different frequency bands.  The
cross-channel score is a mandatory-channel/worst-channel blend::

    J = (1 - lambda) * X1 + lambda * min(X0, X1, X2)

``X`` must be a higher-is-better utility.  For lower-is-better losses, the
equivalent expression uses ``max`` instead of ``min``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


CHANNEL_BANDWIDTH_HZ = 499.2e6
MANDATORY_CHANNEL_ID = 1


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: int
    center_frequency_hz: float
    bandwidth_hz: float = CHANNEL_BANDWIDTH_HZ
    mandatory: bool = False

    @property
    def name(self) -> str:
        return f"ch{self.channel_id}"

    @property
    def lower_frequency_hz(self) -> float:
        return self.center_frequency_hz - self.bandwidth_hz / 2.0

    @property
    def upper_frequency_hz(self) -> float:
        return self.center_frequency_hz + self.bandwidth_hz / 2.0


LOW_BAND_CHANNELS = (
    ChannelSpec(0, 3494.4e6),
    ChannelSpec(1, 3993.6e6, mandatory=True),
    ChannelSpec(2, 4492.8e6),
)


def validate_channel_plan(
    channels: Sequence[ChannelSpec] = LOW_BAND_CHANNELS,
) -> None:
    if tuple(channel.channel_id for channel in channels) != (0, 1, 2):
        raise ValueError("three-channel plan must be ordered as ch0, ch1, ch2")
    if sum(channel.mandatory for channel in channels) != 1:
        raise ValueError("three-channel plan must contain one mandatory channel")
    if not channels[MANDATORY_CHANNEL_ID].mandatory:
        raise ValueError("ch1 must be the mandatory channel")
    for left, right in zip(channels, channels[1:], strict=False):
        if not np.isclose(
            left.upper_frequency_hz,
            right.lower_frequency_hz,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("adjacent IEEE 802.15.6 channels must share an edge")


def _validated_lambda(robustness_lambda: float) -> float:
    value = float(robustness_lambda)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("robustness_lambda must lie in [0, 1]")
    return value


def aggregate_mandatory_worst_utility(
    utilities: Sequence[float] | np.ndarray,
    robustness_lambda: float,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Aggregate three higher-is-better channel utilities."""

    values = np.asarray(utilities, dtype=np.float64)
    if values.shape[axis] != len(LOW_BAND_CHANNELS):
        raise ValueError("utility axis must contain exactly ch0, ch1, ch2")
    if not np.isfinite(values).all():
        raise ValueError("channel utilities must be finite")
    blend = _validated_lambda(robustness_lambda)
    mandatory = np.take(values, MANDATORY_CHANNEL_ID, axis=axis)
    worst = np.min(values, axis=axis)
    return (1.0 - blend) * mandatory + blend * worst


def aggregate_mandatory_worst_loss(
    losses: Sequence[float] | np.ndarray,
    robustness_lambda: float,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Equivalent aggregation for three lower-is-better channel losses."""

    values = np.asarray(losses, dtype=np.float64)
    if values.shape[axis] != len(LOW_BAND_CHANNELS):
        raise ValueError("loss axis must contain exactly ch0, ch1, ch2")
    if not np.isfinite(values).all():
        raise ValueError("channel losses must be finite")
    blend = _validated_lambda(robustness_lambda)
    mandatory = np.take(values, MANDATORY_CHANNEL_ID, axis=axis)
    worst = np.max(values, axis=axis)
    return (1.0 - blend) * mandatory + blend * worst


def quadratic_frequency_weights(
    frequency_hz: Sequence[float] | np.ndarray,
    channel: ChannelSpec,
) -> np.ndarray:
    """Return the centered, edge-zero quadratic window for one channel."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    normalized_offset = (
        2.0 * (frequency - channel.center_frequency_hz) / channel.bandwidth_hz
    )
    weights = 1.0 - normalized_offset**2
    inside = (frequency >= channel.lower_frequency_hz) & (
        frequency <= channel.upper_frequency_hz
    )
    return np.where(inside, np.maximum(weights, 0.0), 0.0)


def quadratic_weighted_mean(
    frequency_hz: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    channel: ChannelSpec,
) -> float:
    """Integrate a scalar curve with a normalized quadratic channel window.

    Exact channel edges are inserted by linear interpolation.  This makes the
    result insensitive to whether a sampled curve happens to include the two
    boundaries and avoids assigning a shared boundary to either neighbor.
    """

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    if frequency.ndim != 1 or samples.shape != frequency.shape:
        raise ValueError("frequency and values must be matching vectors")
    if frequency.size < 2 or np.any(np.diff(frequency) <= 0.0):
        raise ValueError("frequency must contain at least two increasing samples")
    if not np.isfinite(frequency).all() or not np.isfinite(samples).all():
        raise ValueError("frequency and values must be finite")
    if (
        frequency[0] > channel.lower_frequency_hz
        or frequency[-1] < channel.upper_frequency_hz
    ):
        raise ValueError(f"curve does not cover {channel.name}")

    internal = (frequency > channel.lower_frequency_hz) & (
        frequency < channel.upper_frequency_hz
    )
    integration_frequency = np.unique(
        np.concatenate(
            (
                np.asarray([channel.lower_frequency_hz]),
                frequency[internal],
                np.asarray([channel.center_frequency_hz]),
                np.asarray([channel.upper_frequency_hz]),
            )
        )
    )
    integration_values = np.interp(integration_frequency, frequency, samples)
    weights = quadratic_frequency_weights(integration_frequency, channel)
    normalization = float(np.trapezoid(weights, integration_frequency))
    if normalization <= 0.0:
        raise ValueError(f"quadratic window has zero support for {channel.name}")
    return float(
        np.trapezoid(weights * integration_values, integration_frequency)
        / normalization
    )


validate_channel_plan()
