from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.postprocessing.analyze_watermelon_progression import (
    build_sample_metrics,
)


def test_region_metrics_distinguish_plane_from_north_mid() -> None:
    theta_start = np.arange(0.0, 180.0, 10.0)
    theta_stop = theta_start + 10.0
    omega = 2.0 * np.pi * (
        np.cos(np.deg2rad(theta_start)) - np.cos(np.deg2rad(theta_stop))
    )
    directivity = np.ones((1, 2, 18), dtype=float)
    directivity[:, :, (theta_start >= 70.0) & (theta_stop <= 90.0)] = 4.0
    realized = 0.5 * directivity
    arrays = {
        "frequency_ghz": np.array([3.1, 4.8]),
        "theta_start_deg": theta_start,
        "theta_stop_deg": theta_stop,
        "solid_angle_sr": omega,
        "directivity_linear": directivity,
        "realized_gain_linear": realized,
    }
    sample_index = pd.DataFrame(
        [
            {
                "sample_index": 0,
                "sample_kind": "archive",
                "source": "synthetic",
                "case_id": "case_0",
            }
        ]
    )

    metrics, realized_band, directivity_band = build_sample_metrics(
        sample_index,
        arrays,
    )

    assert realized_band.shape == (1, 18)
    assert directivity_band.shape == (1, 18)
    np.testing.assert_allclose(
        metrics.loc[0, "plane_minus_north_mid_dB"],
        10.0 * np.log10(4.0),
    )
