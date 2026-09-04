from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.postprocessing import ber_03_average_s21 as average_s21


def _write_s21(path: Path, frequencies: list[float], magnitudes: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frequency_ghz", "s21_magnitude"])
        writer.writerows(zip(frequencies, magnitudes, strict=True))


def test_linear_magnitude_is_averaged_before_db_conversion() -> None:
    frequencies = np.asarray([3.1, 3.2, 4.8])
    magnitudes = np.asarray([0.01, 0.1, 1.0])

    sample_count, mean_linear, mean_db = (
        average_s21.average_linear_magnitude_in_band(frequencies, magnitudes)
    )

    expected_linear = (0.01 + 0.1 + 1.0) / 3.0
    assert sample_count == 3
    assert mean_linear == pytest.approx(expected_linear)
    assert mean_db == pytest.approx(20.0 * math.log10(expected_linear))
    assert mean_db != pytest.approx(float(np.mean(20.0 * np.log10(magnitudes))))


def test_float32_like_band_endpoints_are_included() -> None:
    frequencies = np.asarray(
        [3.0999999046325684, 3.5, 4.0, 4.800000190734863],
        dtype=np.float64,
    )

    sample_count, mean_linear, mean_db = (
        average_s21.average_linear_magnitude_in_band(
            frequencies,
            np.asarray([0.1, 0.2, 0.3, 0.4]),
        )
    )

    assert sample_count == 4
    assert mean_linear == pytest.approx(0.25)
    assert mean_db == pytest.approx(20.0 * math.log10(0.25))


def test_collection_orders_baseline_first_and_extracts_candidate_rank(
    tmp_path: Path,
) -> None:
    for case_name, level in (
        ("case_prop_02_rank_08", 0.2),
        ("baseline", 0.1),
        ("case_prop_01_rank_03", 0.3),
    ):
        _write_s21(
            tmp_path / case_name / "S21.csv",
            [3.1, 4.0, 4.8],
            [level, level, level],
        )
    (tmp_path / "ignored_without_s21").mkdir()

    records = average_s21.collect_average_s21_records(tmp_path)

    assert [record.case_name for record in records] == [
        "baseline",
        "case_prop_01_rank_03",
        "case_prop_02_rank_08",
    ]
    assert [record.candidate_rank for record in records] == [None, 3, 8]


def test_csv_round_trip_and_overwrite_protection(tmp_path: Path) -> None:
    case_directory = tmp_path / "baseline"
    _write_s21(
        case_directory / "S21.csv",
        [3.1, 4.0, 4.8],
        [0.1, 0.2, 0.3],
    )
    record = average_s21.summarize_s21_directory(case_directory)
    output = tmp_path / "metrics" / "average.csv"

    written = average_s21.write_average_s21_csv([record], output)

    assert written == output.resolve()
    with output.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["case_name"] == "baseline"
    assert float(rows[0]["mean_s21_linear"]) == pytest.approx(0.2)
    with pytest.raises(FileExistsError, match="--overwrite"):
        average_s21.write_average_s21_csv([record], output)


def test_invalid_curve_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "bad.csv"
    _write_s21(invalid, [3.1, 4.8], [0.1, -0.2])

    with pytest.raises(ValueError, match="negative"):
        average_s21.load_s21_magnitude(invalid)
