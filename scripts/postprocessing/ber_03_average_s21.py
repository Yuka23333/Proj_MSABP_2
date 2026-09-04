"""Summarize band-averaged S21 magnitude for the BER candidate set.

The required order is deliberate: average the linear magnitudes first, then
convert that scalar to dB.  This is not the arithmetic mean of the dB column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "processed" / "propagation_s21_14"
)
DEFAULT_BAND_LOW_GHZ = 3.1
DEFAULT_BAND_HIGH_GHZ = 4.8
DEFAULT_OUTPUT_PATH = (
    DEFAULT_INPUT_DIRECTORY / "metrics" / "average_s21_3p1-4p8GHz.csv"
)
RANK_PATTERN = re.compile(r"rank_(\d+)$")


@dataclass(frozen=True)
class AverageS21Record:
    case_name: str
    candidate_rank: int | None
    band_low_ghz: float
    band_high_ghz: float
    band_sample_count: int
    mean_s21_linear: float
    mean_s21_db: float
    source_s21_path: str
    source_s21_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_s21_magnitude(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frequency_ghz", "s21_magnitude"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"S21 CSV is missing columns {sorted(missing)}: {source}")
        rows = [
            (float(row["frequency_ghz"]), float(row["s21_magnitude"]))
            for row in reader
        ]
    if len(rows) < 2:
        raise ValueError(f"S21 CSV contains fewer than two samples: {source}")
    frequencies = np.asarray([row[0] for row in rows], dtype=np.float64)
    magnitudes = np.asarray([row[1] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(magnitudes)):
        raise ValueError(f"S21 CSV contains non-finite data: {source}")
    if np.any(magnitudes < 0.0):
        raise ValueError(f"S21 magnitude contains a negative value: {source}")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError(f"S21 frequencies are not strictly increasing: {source}")
    return frequencies, magnitudes


def average_linear_magnitude_in_band(
    frequencies_ghz: np.ndarray,
    magnitudes: np.ndarray,
    band_low_ghz: float = DEFAULT_BAND_LOW_GHZ,
    band_high_ghz: float = DEFAULT_BAND_HIGH_GHZ,
) -> tuple[int, float, float]:
    frequencies = np.asarray(frequencies_ghz, dtype=np.float64)
    values = np.asarray(magnitudes, dtype=np.float64)
    if frequencies.ndim != 1 or values.shape != frequencies.shape:
        raise ValueError("frequency and S21 magnitude arrays must be matching vectors")
    if frequencies.size < 2 or np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("frequencies must contain at least two increasing samples")
    band_low_ghz = float(band_low_ghz)
    band_high_ghz = float(band_high_ghz)
    if not band_low_ghz < band_high_ghz:
        raise ValueError("S21 averaging band must satisfy low < high")

    median_spacing = float(np.median(np.diff(frequencies)))
    boundary_tolerance = max(1e-9, median_spacing * 1e-3)
    if (
        frequencies[0] > band_low_ghz + boundary_tolerance
        or frequencies[-1] < band_high_ghz - boundary_tolerance
    ):
        raise ValueError("S21 curve does not cover the requested averaging band")
    selected = (frequencies >= band_low_ghz - boundary_tolerance) & (
        frequencies <= band_high_ghz + boundary_tolerance
    )
    sample_count = int(np.count_nonzero(selected))
    if sample_count < 2:
        raise ValueError("fewer than two S21 samples lie inside the averaging band")
    mean_linear = float(np.mean(values[selected]))
    if not math.isfinite(mean_linear) or mean_linear <= 0.0:
        raise ValueError("band-averaged linear S21 magnitude must be positive and finite")
    mean_db = 20.0 * math.log10(mean_linear)
    return sample_count, mean_linear, mean_db


def summarize_s21_directory(
    directory: str | Path,
    *,
    band_low_ghz: float = DEFAULT_BAND_LOW_GHZ,
    band_high_ghz: float = DEFAULT_BAND_HIGH_GHZ,
) -> AverageS21Record:
    case_directory = Path(directory).expanduser().resolve()
    source = case_directory / "S21.csv"
    frequencies, magnitudes = load_s21_magnitude(source)
    sample_count, mean_linear, mean_db = average_linear_magnitude_in_band(
        frequencies,
        magnitudes,
        band_low_ghz,
        band_high_ghz,
    )
    rank_match = RANK_PATTERN.search(case_directory.name)
    candidate_rank = int(rank_match.group(1)) if rank_match else None
    return AverageS21Record(
        case_name=case_directory.name,
        candidate_rank=candidate_rank,
        band_low_ghz=float(band_low_ghz),
        band_high_ghz=float(band_high_ghz),
        band_sample_count=sample_count,
        mean_s21_linear=mean_linear,
        mean_s21_db=mean_db,
        source_s21_path=str(source),
        source_s21_sha256=_sha256(source),
    )


def collect_average_s21_records(
    input_directory: str | Path = DEFAULT_INPUT_DIRECTORY,
    *,
    band_low_ghz: float = DEFAULT_BAND_LOW_GHZ,
    band_high_ghz: float = DEFAULT_BAND_HIGH_GHZ,
) -> list[AverageS21Record]:
    root = Path(input_directory).expanduser().resolve()
    case_directories = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "S21.csv").is_file()
    ]
    case_directories.sort(key=lambda path: (path.name != "baseline", path.name))
    if not case_directories:
        raise FileNotFoundError(f"no case directories containing S21.csv: {root}")
    return [
        summarize_s21_directory(
            directory,
            band_low_ghz=band_low_ghz,
            band_high_ghz=band_high_ghz,
        )
        for directory in case_directories
    ]


def write_average_s21_csv(
    records: Sequence[AverageS21Record],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    if not records:
        raise ValueError("cannot write an empty average-S21 table")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite to replace it: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    fieldnames = tuple(asdict(records[0]))
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average linear S21 magnitude in-band, then convert to dB."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--band-low-ghz", type=float, default=DEFAULT_BAND_LOW_GHZ)
    parser.add_argument("--band-high-ghz", type=float, default=DEFAULT_BAND_HIGH_GHZ)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = collect_average_s21_records(
        args.input_dir,
        band_low_ghz=args.band_low_ghz,
        band_high_ghz=args.band_high_ghz,
    )
    destination = write_average_s21_csv(
        records,
        args.output,
        overwrite=args.overwrite,
    )
    print(
        f"[BER-03] linear |S21| mean -> 20*log10(mean), "
        f"band={args.band_low_ghz:g}--{args.band_high_ghz:g} GHz"
    )
    print(f"[BER-03] output: {destination}")
    print("[BER-03] strongest to weakest transmission:")
    for index, record in enumerate(
        sorted(records, key=lambda item: item.mean_s21_linear, reverse=True),
        start=1,
    ):
        print(
            f"  {index:2d}. {record.case_name:<24} "
            f"linear={record.mean_s21_linear:.9g}  "
            f"dB={record.mean_s21_db:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
