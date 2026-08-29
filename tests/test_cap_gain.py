from __future__ import annotations

import numpy as np
import pytest

from scripts.postprocessing.cap_gain import cap_gain


def _isotropic_ffs() -> dict[str, np.ndarray]:
    theta_deg = np.arange(0.0, 181.0, 5.0)
    phi_deg = np.arange(0.0, 361.0, 5.0)
    shape = (1, len(phi_deg), len(theta_deg))
    return {
        "freq": np.array([4.0e9]),
        "p_rad": np.ones(1),
        "p_acc": np.ones(1),
        "p_stim": np.ones(1),
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
        "E_theta": np.ones(shape, dtype=np.complex128),
        "E_phi": np.zeros(shape, dtype=np.complex128),
    }


def test_isotropic_pattern_has_unit_cap_average_gain() -> None:
    result = cap_gain(_isotropic_ffs(), [15])

    assert result.loc[0, "D_cap_dBi"] == pytest.approx(0.0, abs=1e-12)
    assert result.loc[0, "G_cap_dBi"] == pytest.approx(0.0, abs=1e-12)
    assert result.loc[0, "G_realized_dBi"] == pytest.approx(0.0, abs=1e-12)


def test_cap_boundary_must_exist_on_exported_theta_grid() -> None:
    with pytest.raises(ValueError, match="not present on the exported theta grid"):
        cap_gain(_isotropic_ffs(), [13])
