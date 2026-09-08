from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.postprocessing.ffs_onbody_proxies import (
    collect_paths,
    compute_all,
    compute_onbody_proxies,
)


def _isotropic_ffs() -> dict[str, np.ndarray]:
    theta_deg = np.arange(0.0, 181.0, 5.0)
    phi_deg = np.arange(0.0, 361.0, 5.0)
    shape = (2, len(phi_deg), len(theta_deg))
    return {
        "freq": np.array([3.1e9, 4.8e9]),
        "p_rad": np.array([0.5, 0.6]),
        "p_acc": np.ones(2),
        "p_stim": np.full(2, 2.0),
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
        "E_theta": np.ones(shape, dtype=np.complex128),
        "E_phi": np.zeros(shape, dtype=np.complex128),
    }


def _theta_share(lower_deg: float, upper_deg: float) -> float:
    theta_deg = np.arange(0.0, 181.0, 5.0)
    theta_rad = np.deg2rad(theta_deg)
    mask = (theta_deg >= lower_deg) & (theta_deg <= upper_deg)
    numerator = np.trapezoid(np.sin(theta_rad[mask]), theta_rad[mask])
    denominator = np.trapezoid(np.sin(theta_rad), theta_rad)
    return float(numerator / denominator)


def _isotropic_grid_directivity() -> float:
    theta_rad = np.deg2rad(np.arange(0.0, 181.0, 5.0))
    return float(2.0 / np.trapezoid(np.sin(theta_rad), theta_rad))


def test_multifrequency_isotropic_pattern_preserves_original_proxies() -> None:
    result = compute_onbody_proxies(
        _isotropic_ffs(),
        source_path=Path("isotropic.ffs"),
    )

    assert len(result) == 2
    assert np.allclose(result["freq_ghz"], [3.1, 4.8])
    assert np.allclose(result["P_hor"], _theta_share(70.0, 90.0))
    assert np.allclose(result["P_hor80"], _theta_share(80.0, 90.0))
    assert np.allclose(result["P_cap"], _theta_share(0.0, 40.0))
    assert np.allclose(result["theta_centroid_deg"], 90.0)
    expected_directivity = _isotropic_grid_directivity()
    assert np.allclose(result["G_theta_hor"], expected_directivity)
    assert np.allclose(result["G_theta_hor_min"], expected_directivity)
    assert np.allclose(result["chi_TM"], 1.0)
    assert np.allclose(result["G_endfire_phi90_vertical"], expected_directivity)
    assert np.allclose(result["G_endfire_phi90_horizontal"], 0.0)
    assert np.allclose(result["G_endfire_phi90_total"], expected_directivity)
    assert np.allclose(result["chi_vertical_endfire_phi90"], 1.0)
    assert np.allclose(result["forward_endfire_phi_deg"], 90.0)
    assert np.allclose(result["G_cap_max"], expected_directivity)
    assert np.allclose(result["Rad_Eff_from_ffs"], [0.5, 0.6])
    assert np.allclose(result["Tot_Eff_from_ffs"], [0.25, 0.3])


def test_proxy_boundaries_must_exist_on_exported_grid() -> None:
    with pytest.raises(ValueError, match="must be present"):
        compute_onbody_proxies(_isotropic_ffs(), theta_h_deg=72.0)


def test_phi90_endfire_maps_theta_to_vertical_and_phi_to_horizontal() -> None:
    ffs = _isotropic_ffs()
    theta_index = int(np.flatnonzero(ffs["theta_deg"] == 90.0)[0])
    phi_index = int(np.flatnonzero(ffs["phi_deg"] == 90.0)[0])
    ffs["E_theta"][:, phi_index, theta_index] = 3.0
    ffs["E_phi"][:, phi_index, theta_index] = 4.0

    result = compute_onbody_proxies(ffs)

    np.testing.assert_allclose(
        result["G_endfire_phi90_vertical"] / result["G_endfire_phi90_total"],
        9.0 / 25.0,
    )
    np.testing.assert_allclose(
        result["G_endfire_phi90_horizontal"] / result["G_endfire_phi90_total"],
        16.0 / 25.0,
    )
    np.testing.assert_allclose(
        result["chi_vertical_endfire_phi90"],
        9.0 / 25.0,
    )


def test_collect_paths_is_recursive_case_insensitive_and_unique(
    tmp_path: Path,
) -> None:
    lower = tmp_path / "a.ffs"
    upper = tmp_path / "nested" / "b.FFS"
    upper.parent.mkdir()
    lower.touch()
    upper.touch()
    (tmp_path / "ignored.txt").touch()

    result = collect_paths([tmp_path, lower])

    assert result == [lower.resolve(), upper.resolve()]


def _write_minimal_ffs(path: Path, frequency_hz: float) -> None:
    theta_deg = (0.0, 40.0, 70.0, 80.0, 90.0, 180.0)
    phi_deg = (0.0, 90.0, 360.0)
    field_rows = [
        f"{phi:g} {theta:g} 1 0 0 0" for phi in phi_deg for theta in theta_deg
    ]
    lines = [
        "// #Frequencies",
        "1",
        "// Radiated/Accepted/Stimulated Power",
        "1",
        "1",
        "1",
        f"{frequency_hz:.17g}",
        "// >> Total #phi samples",
        f"{len(phi_deg)} {len(theta_deg)}",
        "// >> Phi, Theta, Re(Etheta), Im(Etheta), Re(Ephi), Im(Ephi)",
        *field_rows,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parallel_compute_matches_serial_and_preserves_input_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.ffs"
    second = tmp_path / "second.ffs"
    _write_minimal_ffs(first, 3.1e9)
    _write_minimal_ffs(second, 4.8e9)

    serial, serial_errors = compute_all([first, second], workers=1)
    parallel, parallel_errors = compute_all([first, second], workers=2)

    assert serial_errors == []
    assert parallel_errors == []
    pd.testing.assert_frame_equal(parallel, serial)
    assert parallel["file"].tolist() == ["first.ffs", "second.ffs"]


def test_compute_all_rejects_invalid_worker_count(tmp_path: Path) -> None:
    path = tmp_path / "sample.ffs"
    _write_minimal_ffs(path, 3.1e9)

    with pytest.raises(ValueError, match="workers must be positive"):
        compute_all([path], workers=0)
