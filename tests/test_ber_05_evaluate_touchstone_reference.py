from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.postprocessing import ber_05_evaluate_touchstone_reference as reference


def test_two_port_ri_order_is_s11_s21_s12_s22(tmp_path: Path) -> None:
    source = tmp_path / "reference.s2p"
    source.write_text(
        "! synthetic\n"
        "# GHz S RI R 50\n"
        "3.1  1 2  3 4  5 6  7 8\n"
        "4.8  2 3  4 5  6 7  8 9\n",
        encoding="utf-8",
    )

    loaded = reference.load_touchstone_s2p(source)

    np.testing.assert_array_equal(loaded.frequency_hz, [3.1e9, 4.8e9])
    np.testing.assert_array_equal(loaded.s11, [1 + 2j, 2 + 3j])
    np.testing.assert_array_equal(loaded.s21, [3 + 4j, 4 + 5j])
    np.testing.assert_array_equal(loaded.s12, [5 + 6j, 6 + 7j])
    np.testing.assert_array_equal(loaded.s22, [7 + 8j, 8 + 9j])
    assert loaded.reference_impedance_ohm == 50.0


@pytest.mark.parametrize(
    ("data_format", "first", "second", "expected"),
    [
        ("MA", 2.0, 90.0, 2.0j),
        ("DB", 20.0 * math.log10(2.0), 180.0, -2.0 + 0.0j),
    ],
)
def test_magnitude_phase_formats_are_supported(
    tmp_path: Path,
    data_format: str,
    first: float,
    second: float,
    expected: complex,
) -> None:
    source = tmp_path / "reference.s2p"
    pair = f"{first:.17g} {second:.17g}"
    source.write_text(
        f"# MHz S {data_format} R 75\n"
        f"3100 {pair} {pair} {pair} {pair}\n"
        f"4800 {pair} {pair} {pair} {pair}\n",
        encoding="utf-8",
    )

    loaded = reference.load_touchstone_s2p(source)

    assert loaded.s21[0] == pytest.approx(expected, abs=1e-12)
    assert loaded.reference_impedance_ohm == 75.0
    np.testing.assert_array_equal(loaded.frequency_hz, [3.1e9, 4.8e9])


def test_converted_csv_uses_common_complex_s21_contract(tmp_path: Path) -> None:
    source = tmp_path / "reference.s2p"
    source.write_text(
        "# GHz S RI R 50\n"
        "3.1 0 0 0.3 0.4 0.3 0.4 0 0\n"
        "4.8 0 0 0 0.5 0 0.5 0 0\n",
        encoding="utf-8",
    )
    loaded = reference.load_touchstone_s2p(source)
    output = tmp_path / "S21.csv"

    reference.write_s21_csv(loaded, output)

    with output.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == [
        "frequency_ghz",
        "s21_real",
        "s21_imag",
        "s21_magnitude",
        "s21_magnitude_db",
        "s21_phase_deg",
    ]
    assert float(rows[0]["s21_magnitude"]) == pytest.approx(0.5)
    assert float(rows[0]["s21_magnitude_db"]) == pytest.approx(
        20.0 * math.log10(0.5)
    )
    with pytest.raises(FileExistsError):
        reference.write_s21_csv(loaded, output)
