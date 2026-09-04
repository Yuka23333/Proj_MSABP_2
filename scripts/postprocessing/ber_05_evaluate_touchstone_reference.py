"""Import a CST two-port Touchstone reference and run the common BER metric.

Touchstone two-port network data uses the order S11, S21, S12, S22.  The
converted ``S21.csv`` follows the same contract as the propagation campaign,
so the average-S21 and BER implementations are reused without changing their
normalization or receiver assumptions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing import ber_02_modulate_uwb_pulses as uwb  # noqa: E402
from scripts.postprocessing import ber_03_average_s21 as average_s21  # noqa: E402
from scripts.postprocessing import ber_04_run_experiment as ber_experiment  # noqa: E402


DEFAULT_TOUCHSTONE_INPUT = (
    REPOSITORY_ROOT
    / "results"
    / "raw"
    / "msa-bp-propagation-baseline-001"
    / "MSA-BP_New-Notch_22-10-2013_CST2023.s2p"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "propagation_s21_14"
    / "roblin_wei_2012_reference"
)

FREQUENCY_SCALES_HZ = {
    "hz": 1.0,
    "khz": 1.0e3,
    "mhz": 1.0e6,
    "ghz": 1.0e9,
}
SUPPORTED_DATA_FORMATS = {"ri", "ma", "db"}


@dataclass(frozen=True)
class TouchstoneS2p:
    frequency_hz: np.ndarray
    s11: np.ndarray
    s21: np.ndarray
    s12: np.ndarray
    s22: np.ndarray
    frequency_unit: str
    data_format: str
    reference_impedance_ohm: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_to_complex(first: float, second: float, data_format: str) -> complex:
    if data_format == "ri":
        return complex(first, second)
    phase_rad = math.radians(second)
    magnitude = first if data_format == "ma" else 10.0 ** (first / 20.0)
    return complex(magnitude * math.cos(phase_rad), magnitude * math.sin(phase_rad))


def load_touchstone_s2p(path: str | Path) -> TouchstoneS2p:
    source = Path(path).expanduser().resolve()
    option_tokens: list[str] | None = None
    numeric_tokens: list[float] = []
    for raw_line in source.read_text(encoding="utf-8-sig").splitlines():
        content = raw_line.split("!", 1)[0].strip()
        if not content:
            continue
        if content.startswith("#"):
            if option_tokens is not None:
                raise ValueError(f"multiple Touchstone option lines: {source}")
            option_tokens = content[1:].casefold().split()
            continue
        if content.startswith("["):
            raise ValueError("Touchstone 2.0 keyword blocks are not supported here")
        try:
            numeric_tokens.extend(float(token) for token in content.split())
        except ValueError as exc:
            raise ValueError(f"invalid numeric Touchstone row: {raw_line!r}") from exc

    if option_tokens is None:
        raise ValueError(f"Touchstone file has no option line: {source}")
    if len(option_tokens) < 3:
        raise ValueError(f"incomplete Touchstone option line: {source}")
    frequency_unit, parameter_type, data_format = option_tokens[:3]
    if frequency_unit not in FREQUENCY_SCALES_HZ:
        raise ValueError(f"unsupported Touchstone frequency unit: {frequency_unit}")
    if parameter_type != "s":
        raise ValueError(f"expected S parameters, got {parameter_type!r}")
    if data_format not in SUPPORTED_DATA_FORMATS:
        raise ValueError(f"unsupported Touchstone data format: {data_format}")

    reference_impedance = 50.0
    if "r" in option_tokens:
        index = option_tokens.index("r")
        if index + 1 >= len(option_tokens):
            raise ValueError("Touchstone R option has no impedance value")
        reference_impedance = float(option_tokens[index + 1])
    if reference_impedance <= 0.0 or not math.isfinite(reference_impedance):
        raise ValueError("Touchstone reference impedance must be positive")

    values_per_record = 9
    if not numeric_tokens or len(numeric_tokens) % values_per_record:
        raise ValueError(
            "two-port Touchstone data must contain 9 numeric values per record"
        )
    records = np.asarray(numeric_tokens, dtype=np.float64).reshape(
        -1,
        values_per_record,
    )
    frequency_hz = records[:, 0] * FREQUENCY_SCALES_HZ[frequency_unit]
    if not np.all(np.isfinite(records)) or np.any(np.diff(frequency_hz) <= 0.0):
        raise ValueError("Touchstone samples must be finite and strictly increasing")

    networks = []
    for first_column in (1, 3, 5, 7):
        networks.append(
            np.asarray(
                [
                    _pair_to_complex(first, second, data_format)
                    for first, second in records[
                        :, first_column : first_column + 2
                    ]
                ],
                dtype=np.complex128,
            )
        )
    return TouchstoneS2p(
        frequency_hz=frequency_hz,
        s11=networks[0],
        s21=networks[1],
        s12=networks[2],
        s22=networks[3],
        frequency_unit=frequency_unit,
        data_format=data_format,
        reference_impedance_ohm=reference_impedance,
    )


def write_s21_csv(
    touchstone: TouchstoneS2p,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(destination).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"S21 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "frequency_ghz",
                    "s21_real",
                    "s21_imag",
                    "s21_magnitude",
                    "s21_magnitude_db",
                    "s21_phase_deg",
                ]
            )
            for frequency_hz, response in zip(
                touchstone.frequency_hz,
                touchstone.s21,
                strict=True,
            ):
                magnitude = abs(response)
                magnitude_db = (
                    20.0 * math.log10(magnitude) if magnitude > 0.0 else -math.inf
                )
                writer.writerow(
                    [
                        frequency_hz / 1.0e9,
                        response.real,
                        response.imag,
                        magnitude,
                        magnitude_db,
                        math.degrees(math.atan2(response.imag, response.real)),
                    ]
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _write_import_manifest(
    source: Path,
    converted_s21: Path,
    touchstone: TouchstoneS2p,
    destination: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "case_name": converted_s21.parent.name,
        "source_touchstone": str(source),
        "source_touchstone_sha256": _sha256(source),
        "converted_s21": converted_s21.name,
        "converted_s21_sha256": _sha256(converted_s21),
        "sample_count": int(touchstone.frequency_hz.size),
        "frequency_range_ghz": [
            float(touchstone.frequency_hz[0] / 1.0e9),
            float(touchstone.frequency_hz[-1] / 1.0e9),
        ],
        "touchstone_frequency_unit": touchstone.frequency_unit,
        "touchstone_data_format": touchstone.data_format,
        "touchstone_parameter_order": ["S11", "S21", "S12", "S22"],
        "reference_impedance_ohm": touchstone.reference_impedance_ohm,
    }
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the true-reference .s2p with the common S21/BER metrics."
    )
    parser.add_argument("--touchstone", type=Path, default=DEFAULT_TOUCHSTONE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--modulation-input",
        type=Path,
        default=uwb.DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--ebn0-db",
        type=float,
        nargs="+",
        default=list(ber_experiment.DEFAULT_EBN0_DB),
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=ber_experiment.DEFAULT_NOISE_MASTER_SEED,
    )
    parser.add_argument("--target-ber", type=float, default=1.0e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.touchstone.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    touchstone = load_touchstone_s2p(source)
    converted = write_s21_csv(
        touchstone,
        output_directory / "S21.csv",
        overwrite=args.overwrite,
    )
    _write_import_manifest(
        source,
        converted,
        touchstone,
        output_directory / "touchstone_import_manifest.json",
    )

    average_record = average_s21.summarize_s21_directory(output_directory)
    average_path = average_s21.write_average_s21_csv(
        [average_record],
        output_directory / "average_s21_3p1-4p8GHz.csv",
        overwrite=args.overwrite,
    )

    modulation = uwb.load_uwb_modulation(args.modulation_input)
    model = ber_experiment.build_channel_model(output_directory, modulation)
    error_counts, evaluated_bits = ber_experiment.run_ber_monte_carlo(
        modulation,
        [model],
        args.ebn0_db,
        noise_master_seed=args.noise_seed,
    )
    aggregates = ber_experiment.aggregate_ber_results(
        [model],
        args.ebn0_db,
        error_counts,
        evaluated_bits,
    )
    ber_directory = (
        output_directory
        / "ber_results"
        / f"matched_filter_{ber_experiment._rate_token(modulation.realized_bit_rate_bps)}Mbps"
    )
    files = ber_experiment.write_results(
        ber_directory,
        args.modulation_input,
        modulation,
        [model],
        args.ebn0_db,
        error_counts,
        evaluated_bits,
        aggregates,
        noise_master_seed=args.noise_seed,
        fft_length=ber_experiment.DEFAULT_FFT_LENGTH,
        max_isi_lags=ber_experiment.DEFAULT_MAX_ISI_LAGS,
        overwrite=args.overwrite,
    )
    figure = ber_experiment.plot_ber_curves(
        aggregates,
        ber_directory / "ber_curves.png",
        target_ber=args.target_ber,
    )
    threshold = ber_experiment._interpolated_threshold_ebn0(
        aggregates,
        args.target_ber,
    )

    print(
        "[BER-05] average S21: "
        f"linear={average_record.mean_s21_linear:.12g}, "
        f"dB={average_record.mean_s21_db:.6f}, "
        f"samples={average_record.band_sample_count}"
    )
    print(
        "[BER-05] received pulse energy: "
        f"linear={model.received_pulse_energy:.12g}, "
        f"dB={model.received_pulse_energy_db:.6f}"
    )
    print(
        f"[BER-05] BER={args.target_ber:g} crossing: "
        + ("not bracketed" if threshold is None else f"{threshold:.6f} dB")
    )
    print(f"[BER-05] converted S21: {converted}")
    print(f"[BER-05] average table: {average_path}")
    print(f"[BER-05] BER table: {files['ber_csv']}")
    print(f"[BER-05] figure: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
