"""September broadband reducers for S21 power and radiation efficiency.

This is an isolated Phase-2 successor to the historical uniform-band
averagers. It does not alter their code or output contracts.

For each IEEE 802.15.6 channel, a normalized quadratic frequency window is
applied in the linear domain. The three channel values are then reduced with
the mandatory-channel/worst-channel rule:

    M = (1 - lambda) * M_ch1 + lambda * min(M_ch0, M_ch1, M_ch2)

Both implemented metrics are higher-is-better:

* S21 uses power transmission abs(S21)**2;
* Rad_Eff converts CST dB samples to linear efficiency before averaging.

dB values are produced only after the linear channel and aggregate values have
been calculated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing import ber_04_run_experiment as s21_io  # noqa: E402
from scripts.postprocessing import ieee802156_three_channel as channels  # noqa: E402


METHOD_NAME = "September"
SCHEMA_VERSION = 1
S21_FILENAME = "S21.csv"
RAD_EFF_FILENAME = "Rad_Eff.csv"
DEFAULT_S21_INPUT_DIRECTORY = s21_io.DEFAULT_INPUT_DIRECTORY
RANK_PATTERN = re.compile(r"rank_(\d+)$")
FLOAT_PAIR_PATTERN = re.compile(
    r"^\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)


@dataclass(frozen=True)
class SeptemberS21Record:
    method: str
    case_id: str
    case_name: str
    candidate_rank: int | None
    robustness_lambda: float
    s21_power_ch0: float
    s21_power_ch1_mandatory: float
    s21_power_ch2: float
    worst_channel: str
    s21_power_september: float
    s21_power_ch0_db: float
    s21_power_ch1_mandatory_db: float
    s21_power_ch2_db: float
    s21_power_september_db: float
    source_s21_path: str
    source_s21_sha256: str


@dataclass(frozen=True)
class SeptemberRadEffRecord:
    method: str
    case_id: str
    case_name: str
    candidate_rank: int | None
    robustness_lambda: float
    rad_eff_ch0_linear: float
    rad_eff_ch1_mandatory_linear: float
    rad_eff_ch2_linear: float
    worst_channel: str
    rad_eff_september_linear: float
    rad_eff_ch0_db: float
    rad_eff_ch1_mandatory_db: float
    rad_eff_ch2_db: float
    rad_eff_september_db: float
    removed_samples_above_unity: int
    source_rad_eff_path: str
    source_rad_eff_sha256: str


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_power_db(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("linear power-like value must be positive and finite")
    return 10.0 * math.log10(value)


def _candidate_rank(case_name: str) -> int | None:
    match = RANK_PATTERN.search(case_name)
    return int(match.group(1)) if match else None


def _case_id(root: Path, curve_path: Path) -> str:
    try:
        relative = curve_path.parent.relative_to(root)
    except ValueError:
        return curve_path.parent.name
    return relative.as_posix() if relative.parts else curve_path.parent.name


def discover_curves(input_directory: str | Path, filename: str) -> list[Path]:
    root = Path(input_directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    curves = sorted(
        (path.resolve() for path in root.rglob(filename) if path.is_file()),
        key=lambda path: (
            path.parent.name != "baseline",
            path.relative_to(root).as_posix().casefold(),
        ),
    )
    if not curves:
        raise FileNotFoundError(f"no {filename} files below {root}")
    return curves


def load_cst_db_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path).expanduser().resolve()
    rows: list[tuple[float, float]] = []
    with source.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            match = FLOAT_PAIR_PATTERN.match(line)
            if match is None:
                continue
            frequency_ghz = float(match.group(1))
            value_db = float(match.group(2))
            if math.isfinite(frequency_ghz) and math.isfinite(value_db):
                rows.append((frequency_ghz, value_db))
    if len(rows) < 2:
        raise ValueError(f"CST curve contains fewer than two numeric rows: {source}")
    rows.sort(key=lambda row: row[0])
    frequency_hz = np.asarray([row[0] * 1.0e9 for row in rows], dtype=np.float64)
    values_db = np.asarray([row[1] for row in rows], dtype=np.float64)
    if np.any(np.diff(frequency_hz) <= 0.0):
        raise ValueError(f"CST curve frequencies are not strictly increasing: {source}")
    return frequency_hz, values_db


def _channel_means(
    frequency_hz: np.ndarray,
    linear_values: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            channels.quadratic_weighted_mean(
                frequency_hz,
                linear_values,
                channel,
            )
            for channel in channels.LOW_BAND_CHANNELS
        ],
        dtype=np.float64,
    )


def summarize_s21_curve(
    source_path: str | Path,
    *,
    input_root: str | Path,
    robustness_lambda: float,
) -> SeptemberS21Record:
    root = Path(input_root).expanduser().resolve()
    transfer = s21_io.load_complex_s21(source_path)
    power = np.abs(transfer.response) ** 2
    channel_values = _channel_means(transfer.frequency_hz, power)
    aggregate = float(
        channels.aggregate_mandatory_worst_utility(
            channel_values,
            robustness_lambda,
        )
    )
    worst_index = int(np.argmin(channel_values))
    case_name = transfer.source_path.parent.name
    return SeptemberS21Record(
        method=METHOD_NAME,
        case_id=_case_id(root, transfer.source_path),
        case_name=case_name,
        candidate_rank=_candidate_rank(case_name),
        robustness_lambda=float(robustness_lambda),
        s21_power_ch0=float(channel_values[0]),
        s21_power_ch1_mandatory=float(channel_values[1]),
        s21_power_ch2=float(channel_values[2]),
        worst_channel=channels.LOW_BAND_CHANNELS[worst_index].name,
        s21_power_september=aggregate,
        s21_power_ch0_db=_to_power_db(channel_values[0]),
        s21_power_ch1_mandatory_db=_to_power_db(channel_values[1]),
        s21_power_ch2_db=_to_power_db(channel_values[2]),
        s21_power_september_db=_to_power_db(aggregate),
        source_s21_path=str(transfer.source_path),
        source_s21_sha256=transfer.sha256,
    )


def summarize_rad_eff_curve(
    source_path: str | Path,
    *,
    input_root: str | Path,
    robustness_lambda: float,
) -> SeptemberRadEffRecord:
    root = Path(input_root).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    frequency_hz, efficiency_db = load_cst_db_curve(source)
    efficiency = np.power(10.0, efficiency_db / 10.0)
    if np.any(efficiency < 0.0) or not np.isfinite(efficiency).all():
        raise ValueError(f"Rad_Eff contains invalid linear values: {source}")
    above_unity = efficiency > 1.0
    removed = int(np.count_nonzero(above_unity))
    if removed:
        frequency_hz = frequency_hz[~above_unity]
        efficiency = efficiency[~above_unity]
    if len(efficiency) < 2:
        raise ValueError(f"too few valid Rad_Eff samples after filtering: {source}")

    channel_values = _channel_means(frequency_hz, efficiency)
    aggregate = float(
        channels.aggregate_mandatory_worst_utility(
            channel_values,
            robustness_lambda,
        )
    )
    worst_index = int(np.argmin(channel_values))
    case_name = source.parent.name
    return SeptemberRadEffRecord(
        method=METHOD_NAME,
        case_id=_case_id(root, source),
        case_name=case_name,
        candidate_rank=_candidate_rank(case_name),
        robustness_lambda=float(robustness_lambda),
        rad_eff_ch0_linear=float(channel_values[0]),
        rad_eff_ch1_mandatory_linear=float(channel_values[1]),
        rad_eff_ch2_linear=float(channel_values[2]),
        worst_channel=channels.LOW_BAND_CHANNELS[worst_index].name,
        rad_eff_september_linear=aggregate,
        rad_eff_ch0_db=_to_power_db(channel_values[0]),
        rad_eff_ch1_mandatory_db=_to_power_db(channel_values[1]),
        rad_eff_ch2_db=_to_power_db(channel_values[2]),
        rad_eff_september_db=_to_power_db(aggregate),
        removed_samples_above_unity=removed,
        source_rad_eff_path=str(source),
        source_rad_eff_sha256=_sha256(source),
    )


def collect_s21_records(
    input_directory: str | Path,
    *,
    robustness_lambda: float,
) -> list[SeptemberS21Record]:
    root = Path(input_directory).expanduser().resolve()
    return [
        summarize_s21_curve(
            path,
            input_root=root,
            robustness_lambda=robustness_lambda,
        )
        for path in discover_curves(root, S21_FILENAME)
    ]


def collect_rad_eff_records(
    input_directory: str | Path,
    *,
    robustness_lambda: float,
) -> list[SeptemberRadEffRecord]:
    root = Path(input_directory).expanduser().resolve()
    return [
        summarize_rad_eff_curve(
            path,
            input_root=root,
            robustness_lambda=robustness_lambda,
        )
        for path in discover_curves(root, RAD_EFF_FILENAME)
    ]


def _atomic_csv(records: Sequence[object], destination: Path, overwrite: bool) -> Path:
    if not records:
        raise ValueError("cannot write an empty September table")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite to replace it: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _atomic_json(
    payload: Mapping[str, Any],
    destination: Path,
    overwrite: bool,
) -> Path:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"manifest already exists; pass --overwrite to replace it: {destination}"
        )
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_september_results(
    records: Sequence[object],
    output_path: str | Path,
    *,
    metric: str,
    input_directory: str | Path,
    robustness_lambda: float,
    overwrite: bool = False,
) -> dict[str, Path]:
    destination = Path(output_path).expanduser().resolve()
    manifest_path = destination.with_suffix(".manifest.json")
    if not overwrite:
        existing = [path for path in (destination, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "September output already exists; pass --overwrite to replace: "
                + ", ".join(str(path) for path in existing)
            )
    _atomic_csv(records, destination, overwrite=overwrite)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "metric": metric,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(Path(input_directory).expanduser().resolve()),
        "record_count": len(records),
        "robustness_lambda": float(robustness_lambda),
        "channel_plan": [
            {
                "channel_id": channel.channel_id,
                "name": channel.name,
                "mandatory": channel.mandatory,
                "center_frequency_mhz": channel.center_frequency_hz / 1.0e6,
                "bandwidth_mhz": channel.bandwidth_hz / 1.0e6,
            }
            for channel in channels.LOW_BAND_CHANNELS
        ],
        "within_channel_weight": ("normalized quadratic: max(0, 1-(2*(f-fc)/BW)^2)"),
        "cross_channel_aggregation": (
            "(1-lambda)*M_ch1 + lambda*min(M_ch0,M_ch1,M_ch2)"
        ),
        "linear_domain": True,
        "metric_definition": (
            "weighted mean of abs(S21)^2; dB is 10*log10 after aggregation"
            if metric == "s21_power"
            else (
                "weighted mean of linear Rad_Eff converted from CST dB; "
                "dB is 10*log10 after aggregation"
            )
        ),
        "output": destination.name,
        "output_sha256": _sha256(destination),
    }
    _atomic_json(manifest, manifest_path, overwrite=overwrite)
    return {"table": destination, "manifest": manifest_path}


def _default_output(input_directory: Path, metric: str) -> Path:
    name = "september_s21_power.csv" if metric == "s21" else "september_rad_eff.csv"
    return input_directory.expanduser().resolve() / "metrics" / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute isolated September S21 or Rad_Eff broadband metrics."
    )
    subparsers = parser.add_subparsers(dest="metric", required=True)
    for metric, default_input in (
        ("s21", DEFAULT_S21_INPUT_DIRECTORY),
        ("rad-eff", None),
    ):
        subparser = subparsers.add_parser(metric)
        subparser.add_argument(
            "--input-dir",
            type=Path,
            default=default_input,
            required=default_input is None,
        )
        subparser.add_argument("--output", type=Path)
        subparser.add_argument(
            "--robustness-lambda",
            type=float,
            required=True,
            help="lambda in [0,1]; no scientific default is assumed",
        )
        subparser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or _default_output(args.input_dir, args.metric)
    if args.metric == "s21":
        records: Sequence[object] = collect_s21_records(
            args.input_dir,
            robustness_lambda=args.robustness_lambda,
        )
        metric_name = "s21_power"
    else:
        records = collect_rad_eff_records(
            args.input_dir,
            robustness_lambda=args.robustness_lambda,
        )
        metric_name = "radiation_efficiency"
    files = write_september_results(
        records,
        output,
        metric=metric_name,
        input_directory=args.input_dir,
        robustness_lambda=args.robustness_lambda,
        overwrite=args.overwrite,
    )
    print(
        f"[September] metric={metric_name}, records={len(records)}, "
        f"lambda={args.robustness_lambda:g}"
    )
    for name, path in files.items():
        print(f"[September] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
