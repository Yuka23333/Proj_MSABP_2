from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.postprocessing.watermelon_latitude_gain import (
    THETA_EDGES_DEG,
    broadband_table,
    latitude_belt_gain,
)


def _isotropic_ffs() -> dict[str, np.ndarray]:
    theta_deg = np.arange(0.0, 181.0, 5.0)
    phi_deg = np.arange(0.0, 361.0, 5.0)
    shape = (2, len(phi_deg), len(theta_deg))
    return {
        "freq": np.array([3.1e9, 4.8e9]),
        "p_rad": np.array([0.5, 0.5]),
        "p_acc": np.ones(2),
        "p_stim": np.full(2, 2.0),
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
        "E_theta": np.ones(shape, dtype=np.complex128),
        "E_phi": np.zeros(shape, dtype=np.complex128),
    }


def test_isotropic_pattern_has_uniform_belt_gain() -> None:
    result = latitude_belt_gain(_isotropic_ffs())

    assert result["directivity_linear"].shape == (2, 18)
    assert np.allclose(result["directivity_linear"], 1.0, atol=1e-12)
    assert np.allclose(result["gain_linear"], 0.5, atol=1e-12)
    assert np.allclose(result["realized_gain_linear"], 0.25, atol=1e-12)
    assert np.isclose(result["solid_angle_sr"].sum(), 4.0 * np.pi)
    assert np.array_equal(result["theta_start_deg"], THETA_EDGES_DEG[:-1])
    assert np.array_equal(result["theta_stop_deg"], THETA_EDGES_DEG[1:])


def test_broadband_table_keeps_linear_power_average() -> None:
    source = latitude_belt_gain(_isotropic_ffs())
    arrays = {
        **{
            name: values
            for name, values in source.items()
            if name
            in {
                "frequency_ghz",
                "theta_start_deg",
                "theta_stop_deg",
                "solid_angle_sr",
            }
        },
        "directivity_linear": source["directivity_linear"][None, :, :],
        "gain_linear": source["gain_linear"][None, :, :],
        "realized_gain_linear": source["realized_gain_linear"][None, :, :],
        "radiation_efficiency_linear": source["radiation_efficiency_linear"][
            None, :
        ],
        "total_efficiency_linear": source["total_efficiency_linear"][None, :],
    }
    index = pd.DataFrame(
        [
            {
                "sample_index": 0,
                "sample_kind": "test",
                "source": "isotropic",
                "case_id": "case_0",
            }
        ]
    )

    table = broadband_table(index, arrays)

    assert len(table) == 18
    assert np.allclose(table["realized_gain_linear"], 0.25)
    assert np.allclose(table["realized_gain_dbi"], 10.0 * np.log10(0.25))
    assert np.allclose(table["mean_total_efficiency_linear"], 0.25)
