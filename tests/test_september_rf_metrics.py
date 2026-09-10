from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.postprocessing import ieee802156_three_channel as channels
from scripts.postprocessing import september_rf_metrics as september


def _write_s21(path: Path, rows: list[tuple[float, complex]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frequency_ghz", "s21_real", "s21_imag"])
        for frequency_ghz, value in rows:
            writer.writerow([frequency_ghz, value.real, value.imag])


def _write_cst_curve(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# CST ASCII curve\n")
        for frequency_ghz, value_db in rows:
            stream.write(f"{frequency_ghz:.9f}\t{value_db:.12g}\n")


def _channel_grid() -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.linspace(
                    channel.lower_frequency_hz,
                    channel.upper_frequency_hz,
                    11,
                )
                for channel in channels.LOW_BAND_CHANNELS
            ]
        )
    )


def test_s21_averages_linear_power_then_converts_to_db(tmp_path: Path) -> None:
    frequency_hz = _channel_grid()
    magnitudes = np.select(
        [
            frequency_hz < channels.LOW_BAND_CHANNELS[0].upper_frequency_hz,
            frequency_hz < channels.LOW_BAND_CHANNELS[1].upper_frequency_hz,
        ],
        [0.1, 0.2],
        default=0.4,
    )
    source = tmp_path / "baseline" / "S21.csv"
    _write_s21(
        source,
        [
            (frequency / 1.0e9, complex(magnitude))
            for frequency, magnitude in zip(
                frequency_hz,
                magnitudes,
                strict=True,
            )
        ],
    )

    record = september.summarize_s21_curve(
        source,
        input_root=tmp_path,
        robustness_lambda=0.25,
    )

    assert record.s21_power_ch0 == pytest.approx(0.01)
    assert record.s21_power_ch1_mandatory == pytest.approx(0.04)
    assert record.s21_power_ch2 == pytest.approx(0.16)
    assert record.s21_power_september == pytest.approx(0.75 * 0.04 + 0.25 * 0.01)
    assert record.s21_power_september_db == pytest.approx(
        10.0 * math.log10(record.s21_power_september)
    )
    assert record.s21_power_september != pytest.approx((0.75 * 0.2 + 0.25 * 0.1) ** 2)


def test_rad_eff_converts_each_db_sample_before_weighting(tmp_path: Path) -> None:
    frequency_hz = _channel_grid()
    efficiency = (
        0.2
        + 0.6
        * ((frequency_hz - frequency_hz[0]) / (frequency_hz[-1] - frequency_hz[0])) ** 2
    )
    source = tmp_path / "case_0001" / "Rad_Eff.csv"
    _write_cst_curve(
        source,
        [
            (frequency / 1.0e9, 10.0 * math.log10(value))
            for frequency, value in zip(
                frequency_hz,
                efficiency,
                strict=True,
            )
        ],
    )

    record = september.summarize_rad_eff_curve(
        source,
        input_root=tmp_path,
        robustness_lambda=0.4,
    )

    expected = np.asarray(
        [
            channels.quadratic_weighted_mean(
                frequency_hz,
                efficiency,
                channel,
            )
            for channel in channels.LOW_BAND_CHANNELS
        ]
    )
    assert record.rad_eff_ch0_linear == pytest.approx(expected[0])
    assert record.rad_eff_ch1_mandatory_linear == pytest.approx(expected[1])
    assert record.rad_eff_ch2_linear == pytest.approx(expected[2])
    assert record.rad_eff_september_linear == pytest.approx(
        0.6 * expected[1] + 0.4 * np.min(expected)
    )


def test_rad_eff_records_removed_solver_values_above_unity(tmp_path: Path) -> None:
    frequency_hz = _channel_grid()
    efficiency = np.full(frequency_hz.shape, 0.5)
    efficiency[len(efficiency) // 2] = 10.0
    source = tmp_path / "case_0002" / "Rad_Eff.csv"
    _write_cst_curve(
        source,
        [
            (frequency / 1.0e9, 10.0 * math.log10(value))
            for frequency, value in zip(
                frequency_hz,
                efficiency,
                strict=True,
            )
        ],
    )

    record = september.summarize_rad_eff_curve(
        source,
        input_root=tmp_path,
        robustness_lambda=0.5,
    )

    assert record.removed_samples_above_unity == 1
    assert record.rad_eff_september_linear == pytest.approx(0.5)


def test_writer_records_september_contract(tmp_path: Path) -> None:
    source = tmp_path / "baseline" / "S21.csv"
    _write_s21(
        source,
        [(3.0, 0.1 + 0.0j), (4.0, 0.1 + 0.0j), (5.0, 0.1 + 0.0j)],
    )
    records = [
        september.summarize_s21_curve(
            source,
            input_root=tmp_path,
            robustness_lambda=0.5,
        )
    ]

    files = september.write_september_results(
        records,
        tmp_path / "metrics" / "s21.csv",
        metric="s21_power",
        input_directory=tmp_path,
        robustness_lambda=0.5,
    )

    with files["manifest"].open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["method"] == "September"
    assert manifest["linear_domain"] is True
    assert "abs(S21)^2" in manifest["metric_definition"]
    assert all(path.is_file() for path in files.values())
