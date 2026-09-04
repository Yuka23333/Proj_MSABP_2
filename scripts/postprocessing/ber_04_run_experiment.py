"""Run the reproducible BER comparison over all collected propagation cases.

The CST export contains complex S(2,1), so this stage preserves both magnitude
and phase.  For every case it interpolates the measured transfer function onto
the UWB pulse FFT grid and obtains the received pulse by an inverse real FFT.

The receiver is an ideal, case-specific matched filter with perfect timing and
channel knowledge.  Noise is added at the receiver with one common absolute
noise floor for every antenna.  ``tx_reference_ebn0_db`` therefore means that
the transmitted, unit-energy pulse defines Eb while the propagation loss in
S(2,1) is *not* normalized away.  Common random noise is reused across cases to
make pairwise antenna comparisons less noisy.

The sparse symbol representation is evaluated directly at matched-filter
decision instants; a multi-terabyte dense 24-GSa/s waveform is never created.
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
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing import ber_02_modulate_uwb_pulses as uwb  # noqa: E402


DEFAULT_INPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "processed" / "propagation_s21_14"
)
DEFAULT_MODULATION_INPUT = uwb.DEFAULT_OUTPUT_PATH
DEFAULT_EBN0_DB = (40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0)
DEFAULT_NOISE_MASTER_SEED = 20260905
DEFAULT_FFT_LENGTH = 65536
DEFAULT_MAX_ISI_LAGS = 4
DEFAULT_TARGET_BER = 1.0e-4
SCHEMA_VERSION = 1
RANK_PATTERN = re.compile(r"rank_(\d+)$")


def _rate_token(bit_rate_bps: float) -> str:
    return format(float(bit_rate_bps) / 1.0e6, ".12g").replace(".", "p")


def default_output_directory(bit_rate_bps: float = uwb.DEFAULT_BIT_RATE_BPS) -> Path:
    return (
        DEFAULT_INPUT_DIRECTORY
        / "ber_results"
        / f"matched_filter_{_rate_token(bit_rate_bps)}Mbps"
    )


@dataclass(frozen=True)
class ComplexS21:
    frequency_hz: np.ndarray
    response: np.ndarray
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class ChannelModel:
    case_name: str
    candidate_rank: int | None
    received_pulse_energy: float
    received_pulse_energy_db: float
    peak_delay_ns: float
    normalized_symbol_taps: np.ndarray
    max_abs_isi_ratio: float
    source_s21_path: str
    source_s21_sha256: str


@dataclass(frozen=True)
class BerAggregate:
    case_name: str
    candidate_rank: int | None
    tx_reference_ebn0_db: float
    error_count: int
    evaluated_bits: int
    ber: float
    ber_ci95_low: float
    ber_ci95_high: float
    theoretical_no_isi_ber: float
    received_pulse_energy: float
    received_pulse_energy_db: float
    max_abs_isi_ratio: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_complex_s21(path: str | Path) -> ComplexS21:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frequency_ghz", "s21_real", "s21_imag"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"S21 CSV is missing columns {sorted(missing)}: {source}")
        rows = [
            (
                float(row["frequency_ghz"]) * 1.0e9,
                complex(float(row["s21_real"]), float(row["s21_imag"])),
            )
            for row in reader
        ]
    if len(rows) < 2:
        raise ValueError(f"S21 CSV contains fewer than two samples: {source}")
    frequency_hz = np.asarray([row[0] for row in rows], dtype=np.float64)
    response = np.asarray([row[1] for row in rows], dtype=np.complex128)
    if not np.all(np.isfinite(frequency_hz)) or not np.all(np.isfinite(response)):
        raise ValueError(f"S21 CSV contains non-finite data: {source}")
    if np.any(np.diff(frequency_hz) <= 0.0):
        raise ValueError(f"S21 frequencies are not strictly increasing: {source}")
    return ComplexS21(
        frequency_hz=frequency_hz,
        response=response,
        source_path=source,
        sha256=_sha256(source),
    )


def discover_s21_cases(input_directory: str | Path) -> list[Path]:
    root = Path(input_directory).expanduser().resolve()
    cases = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "S21.csv").is_file()
    ]
    cases.sort(key=lambda path: (path.name != "baseline", path.name))
    if not cases:
        raise FileNotFoundError(f"no case directories containing S21.csv: {root}")
    return cases


def _linear_autocorrelation_at_lag(signal: np.ndarray, lag: int) -> float:
    if lag == 0:
        return float(np.dot(signal, signal))
    if lag < 0:
        lag = -lag
    if lag >= signal.size:
        return 0.0
    return float(np.dot(signal[:-lag], signal[lag:]))


def build_channel_model(
    case_directory: str | Path,
    modulation: uwb.UwbModulationBatch,
    *,
    fft_length: int = DEFAULT_FFT_LENGTH,
    max_isi_lags: int = DEFAULT_MAX_ISI_LAGS,
) -> ChannelModel:
    case_path = Path(case_directory).expanduser().resolve()
    transfer = load_complex_s21(case_path / "S21.csv")
    fft_length = int(fft_length)
    max_isi_lags = int(max_isi_lags)
    if fft_length < modulation.pulse.samples.size:
        raise ValueError("fft_length is shorter than the UWB pulse")
    if fft_length <= 2 * (max_isi_lags + 1) * modulation.samples_per_bit:
        raise ValueError(
            "fft_length is too short for the requested matched-filter ISI span"
        )
    if max_isi_lags < 0:
        raise ValueError("max_isi_lags cannot be negative")

    fft_frequency_hz = np.fft.rfftfreq(
        fft_length,
        d=1.0 / modulation.sample_rate_hz,
    )
    response = np.interp(
        fft_frequency_hz,
        transfer.frequency_hz,
        transfer.response.real,
        left=0.0,
        right=0.0,
    ) + 1j * np.interp(
        fft_frequency_hz,
        transfer.frequency_hz,
        transfer.response.imag,
        left=0.0,
        right=0.0,
    )
    pulse_spectrum = np.fft.rfft(modulation.pulse.samples, n=fft_length)
    received = np.fft.irfft(pulse_spectrum * response, n=fft_length)

    # Keep the main response away from the periodic FFT record boundary before
    # evaluating a linear autocorrelation.  This roll changes timing only.
    original_peak_index = int(np.argmax(np.abs(received)))
    received = np.roll(received, fft_length // 2 - original_peak_index)
    energy = _linear_autocorrelation_at_lag(received, 0)
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError(f"received pulse has invalid energy: {case_path.name}")

    positive_correlations = np.asarray(
        [
            _linear_autocorrelation_at_lag(
                received,
                offset * modulation.samples_per_bit,
            )
            for offset in range(1, max_isi_lags + 1)
        ],
        dtype=np.float64,
    )
    normalized_positive = positive_correlations / energy
    normalized_taps = np.concatenate(
        (normalized_positive[::-1], np.asarray([1.0]), normalized_positive)
    )
    max_abs_isi_ratio = (
        float(np.max(np.abs(normalized_positive)))
        if normalized_positive.size
        else 0.0
    )
    rank_match = RANK_PATTERN.search(case_path.name)
    candidate_rank = int(rank_match.group(1)) if rank_match else None
    return ChannelModel(
        case_name=case_path.name,
        candidate_rank=candidate_rank,
        received_pulse_energy=energy,
        received_pulse_energy_db=10.0 * math.log10(energy),
        peak_delay_ns=original_peak_index / modulation.sample_rate_hz * 1.0e9,
        normalized_symbol_taps=normalized_taps,
        max_abs_isi_ratio=max_abs_isi_ratio,
        source_s21_path=str(transfer.source_path),
        source_s21_sha256=transfer.sha256,
    )


def build_all_channel_models(
    input_directory: str | Path,
    modulation: uwb.UwbModulationBatch,
    *,
    fft_length: int = DEFAULT_FFT_LENGTH,
    max_isi_lags: int = DEFAULT_MAX_ISI_LAGS,
) -> list[ChannelModel]:
    return [
        build_channel_model(
            case,
            modulation,
            fft_length=fft_length,
            max_isi_lags=max_isi_lags,
        )
        for case in discover_s21_cases(input_directory)
    ]


def _wilson_interval(error_count: int, sample_count: int) -> tuple[float, float]:
    if sample_count <= 0 or not 0 <= error_count <= sample_count:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    proportion = error_count / sample_count
    denominator = 1.0 + z * z / sample_count
    center = (proportion + z * z / (2.0 * sample_count)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / sample_count
            + z * z / (4.0 * sample_count**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _theoretical_no_isi_ber(ebn0_db: float, channel_energy: float) -> float:
    linear_ebn0 = 10.0 ** (float(ebn0_db) / 10.0)
    return 0.5 * math.erfc(math.sqrt(linear_ebn0 * channel_energy))


def run_ber_monte_carlo(
    modulation: uwb.UwbModulationBatch,
    channel_models: Sequence[ChannelModel],
    ebn0_db: Sequence[float] = DEFAULT_EBN0_DB,
    *,
    noise_master_seed: int = DEFAULT_NOISE_MASTER_SEED,
    max_bits_per_repeat: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return error counts [case, SNR, repeat] and evaluated-bit counts."""

    uwb.validate_uwb_modulation(modulation)
    if not channel_models:
        raise ValueError("at least one channel model is required")
    ebn0_values = np.asarray(ebn0_db, dtype=np.float64)
    if (
        ebn0_values.ndim != 1
        or ebn0_values.size == 0
        or not np.all(np.isfinite(ebn0_values))
        or np.any(np.diff(ebn0_values) <= 0.0)
    ):
        raise ValueError("Eb/N0 values must be a finite, strictly increasing vector")
    tap_lengths = {model.normalized_symbol_taps.size for model in channel_models}
    if len(tap_lengths) != 1:
        raise ValueError("all channel models must use the same ISI tap span")
    tap_length = tap_lengths.pop()
    if tap_length % 2 != 1:
        raise ValueError("channel symbol-tap vectors must have odd length")
    guard_bits = tap_length // 2

    available_bits = modulation.bits_per_repeat
    requested_bits = (
        available_bits
        if max_bits_per_repeat is None
        else min(available_bits, int(max_bits_per_repeat))
    )
    if requested_bits <= 2 * guard_bits:
        raise ValueError("too few bits remain after discarding ISI edge guards")
    evaluated_per_repeat = requested_bits - 2 * guard_bits
    evaluated_bits = np.full(
        modulation.repeat_count,
        evaluated_per_repeat,
        dtype=np.int64,
    )
    error_counts = np.zeros(
        (len(channel_models), ebn0_values.size, modulation.repeat_count),
        dtype=np.int64,
    )

    noise_sequences = np.random.SeedSequence(int(noise_master_seed)).spawn(
        modulation.repeat_count * ebn0_values.size
    )
    linear_ebn0 = 10.0 ** (ebn0_values / 10.0)
    valid_slice = slice(guard_bits, requested_bits - guard_bits)
    for repeat_index in range(modulation.repeat_count):
        symbols = modulation.symbols[repeat_index, :requested_bits].astype(
            np.float64,
            copy=False,
        )
        truth = symbols[valid_slice]
        common_noise = np.empty(
            (ebn0_values.size, evaluated_per_repeat),
            dtype=np.float64,
        )
        for snr_index in range(ebn0_values.size):
            seed_index = repeat_index * ebn0_values.size + snr_index
            generator = np.random.Generator(
                np.random.PCG64(noise_sequences[seed_index])
            )
            common_noise[snr_index] = generator.standard_normal(
                evaluated_per_repeat
            )

        for case_index, model in enumerate(channel_models):
            noiseless = np.convolve(
                symbols,
                model.normalized_symbol_taps,
                mode="same",
            )[valid_slice]
            sigma = np.sqrt(1.0 / (2.0 * linear_ebn0 * model.received_pulse_energy))
            for snr_index in range(ebn0_values.size):
                decision = noiseless + sigma[snr_index] * common_noise[snr_index]
                error_counts[case_index, snr_index, repeat_index] = np.count_nonzero(
                    decision * truth <= 0.0
                )
    return error_counts, evaluated_bits


def aggregate_ber_results(
    channel_models: Sequence[ChannelModel],
    ebn0_db: Sequence[float],
    error_counts: np.ndarray,
    evaluated_bits_per_repeat: np.ndarray,
) -> list[BerAggregate]:
    ebn0_values = np.asarray(ebn0_db, dtype=np.float64)
    counts = np.asarray(error_counts, dtype=np.int64)
    evaluated = np.asarray(evaluated_bits_per_repeat, dtype=np.int64)
    expected_shape = (len(channel_models), ebn0_values.size, evaluated.size)
    if counts.shape != expected_shape:
        raise ValueError(f"error-count shape {counts.shape} != {expected_shape}")
    total_bits = int(evaluated.sum())
    records: list[BerAggregate] = []
    for case_index, model in enumerate(channel_models):
        for snr_index, snr_db in enumerate(ebn0_values):
            errors = int(counts[case_index, snr_index].sum())
            ci_low, ci_high = _wilson_interval(errors, total_bits)
            records.append(
                BerAggregate(
                    case_name=model.case_name,
                    candidate_rank=model.candidate_rank,
                    tx_reference_ebn0_db=float(snr_db),
                    error_count=errors,
                    evaluated_bits=total_bits,
                    ber=errors / total_bits,
                    ber_ci95_low=ci_low,
                    ber_ci95_high=ci_high,
                    theoretical_no_isi_ber=_theoretical_no_isi_ber(
                        float(snr_db),
                        model.received_pulse_energy,
                    ),
                    received_pulse_energy=model.received_pulse_energy,
                    received_pulse_energy_db=model.received_pulse_energy_db,
                    max_abs_isi_ratio=model.max_abs_isi_ratio,
                )
            )
    return records


def _atomic_csv(rows: Sequence[dict[str, object]], destination: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
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


def _atomic_json(payload: dict[str, object], destination: Path) -> None:
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


def _atomic_npz(payload: dict[str, np.ndarray], destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        np.savez(temporary, **payload)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_results(
    output_directory: str | Path,
    modulation_input: str | Path,
    modulation: uwb.UwbModulationBatch,
    channel_models: Sequence[ChannelModel],
    ebn0_db: Sequence[float],
    error_counts: np.ndarray,
    evaluated_bits: np.ndarray,
    aggregates: Sequence[BerAggregate],
    *,
    noise_master_seed: int,
    fft_length: int,
    max_isi_lags: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    destination = Path(output_directory).expanduser().resolve()
    files = {
        "ber_csv": destination / "ber_results.csv",
        "channel_csv": destination / "channel_metrics.csv",
        "raw_npz": destination / "ber_repeat_counts.npz",
        "manifest": destination / "manifest.json",
    }
    existing = [path for path in files.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "BER output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    destination.mkdir(parents=True, exist_ok=True)

    _atomic_csv([asdict(record) for record in aggregates], files["ber_csv"])
    channel_rows = []
    for model in channel_models:
        row = asdict(model)
        row["normalized_symbol_taps"] = json.dumps(
            model.normalized_symbol_taps.tolist(), separators=(",", ":")
        )
        channel_rows.append(row)
    _atomic_csv(channel_rows, files["channel_csv"])
    ebn0_values = np.asarray(ebn0_db, dtype=np.float64)
    _atomic_npz(
        {
            "case_names": np.asarray([model.case_name for model in channel_models]),
            "candidate_ranks": np.asarray(
                [
                    -1 if model.candidate_rank is None else model.candidate_rank
                    for model in channel_models
                ],
                dtype=np.int64,
            ),
            "tx_reference_ebn0_db": ebn0_values,
            "error_counts": np.asarray(error_counts, dtype=np.int64),
            "evaluated_bits_per_repeat": np.asarray(
                evaluated_bits,
                dtype=np.int64,
            ),
            "received_pulse_energy": np.asarray(
                [model.received_pulse_energy for model in channel_models]
            ),
            "max_abs_isi_ratio": np.asarray(
                [model.max_abs_isi_ratio for model in channel_models]
            ),
            "noise_master_seed": np.asarray(noise_master_seed, dtype=np.uint64),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        },
        files["raw_npz"],
    )
    modulation_path = Path(modulation_input).expanduser().resolve()
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method": "complex_S21_iFFT_then_case_specific_matched_filter_BPSK",
        "receiver": "perfect timing and channel knowledge",
        "noise_location": "receiver input after propagation channel",
        "normalization": (
            "transmitted pulse has unit discrete energy; channel attenuation "
            "is retained; no per-case AGC normalization"
        ),
        "common_random_numbers_across_cases": True,
        "sample_rate_hz": modulation.sample_rate_hz,
        "bit_rate_bps": modulation.realized_bit_rate_bps,
        "samples_per_bit": modulation.samples_per_bit,
        "pulse_band_hz": [
            modulation.pulse.band_low_hz,
            modulation.pulse.band_high_hz,
        ],
        "fft_length": int(fft_length),
        "max_isi_lags_each_side": int(max_isi_lags),
        "tx_reference_ebn0_db": ebn0_values.tolist(),
        "noise_master_seed": int(noise_master_seed),
        "repeat_count": modulation.repeat_count,
        "bits_per_repeat": modulation.bits_per_repeat,
        "evaluated_bits_per_snr": int(np.asarray(evaluated_bits).sum()),
        "source_bit_sha256": modulation.source_bit_sha256,
        "modulation_input": str(modulation_path),
        "modulation_input_sha256": _sha256(modulation_path),
        "case_count": len(channel_models),
        "cases": [
            {
                "case_name": model.case_name,
                "candidate_rank": model.candidate_rank,
                "s21_sha256": model.source_s21_sha256,
            }
            for model in channel_models
        ],
        "outputs": {name: path.name for name, path in files.items()},
    }
    _atomic_json(manifest, files["manifest"])
    return files


def plot_ber_curves(
    records: Sequence[BerAggregate],
    output_path: str | Path,
    *,
    target_ber: float = DEFAULT_TARGET_BER,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_path).expanduser().resolve()
    case_names = list(dict.fromkeys(record.case_name for record in records))
    figure, axis = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)
    for case_name in case_names:
        selected = [record for record in records if record.case_name == case_name]
        selected.sort(key=lambda record: record.tx_reference_ebn0_db)
        x = [record.tx_reference_ebn0_db for record in selected]
        # A zero-error observation is shown at half an error for log plotting.
        y = [
            record.ber
            if record.error_count > 0
            else 0.5 / record.evaluated_bits
            for record in selected
        ]
        is_baseline = case_name == "baseline"
        axis.semilogy(
            x,
            y,
            marker="o",
            markersize=4.5 if is_baseline else 3.2,
            linewidth=2.8 if is_baseline else 1.15,
            color="black" if is_baseline else None,
            label=case_name,
            zorder=5 if is_baseline else 2,
        )
    axis.axhline(target_ber, color="0.35", linestyle="--", linewidth=1.1)
    axis.set_xlabel("Transmitter-reference $E_b/N_0$ (dB)")
    axis.set_ylabel("BER")
    axis.set_title("UWB propagation BER (complex S21, matched-filter receiver)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=7, ncol=2, loc="lower left")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _interpolated_threshold_ebn0(
    records: Sequence[BerAggregate],
    target_ber: float,
) -> float | None:
    usable = sorted(records, key=lambda record: record.tx_reference_ebn0_db)
    for lower, upper in zip(usable, usable[1:], strict=False):
        lower_for_log = max(lower.ber, 0.5 / lower.evaluated_bits)
        upper_for_log = max(upper.ber, 0.5 / upper.evaluated_bits)
        if lower_for_log >= target_ber and upper_for_log <= target_ber:
            x0, x1 = lower.tx_reference_ebn0_db, upper.tx_reference_ebn0_db
            y0, y1 = math.log10(lower_for_log), math.log10(upper_for_log)
            if y0 == y1:
                return x1
            fraction = (math.log10(target_ber) - y0) / (y1 - y0)
            return x0 + fraction * (x1 - x0)
    return None


def print_summary(
    records: Sequence[BerAggregate],
    *,
    target_ber: float = DEFAULT_TARGET_BER,
) -> None:
    case_names = list(dict.fromkeys(record.case_name for record in records))
    summaries = []
    for case_name in case_names:
        selected = [record for record in records if record.case_name == case_name]
        threshold = _interpolated_threshold_ebn0(selected, target_ber)
        summaries.append((threshold, case_name))
    summaries.sort(key=lambda item: (item[0] is None, item[0], item[1]))
    print(f"[BER-04] empirical target BER={target_ber:g} crossing:")
    for index, (threshold, case_name) in enumerate(summaries, start=1):
        rendered = "not bracketed" if threshold is None else f"{threshold:.3f} dB"
        print(f"  {index:2d}. {case_name:<24} {rendered}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run complex-S21 UWB matched-filter BER comparisons."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument(
        "--modulation-input",
        type=Path,
        default=DEFAULT_MODULATION_INPUT,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--ebn0-db",
        type=float,
        nargs="+",
        default=list(DEFAULT_EBN0_DB),
    )
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_MASTER_SEED)
    parser.add_argument("--fft-length", type=int, default=DEFAULT_FFT_LENGTH)
    parser.add_argument("--max-isi-lags", type=int, default=DEFAULT_MAX_ISI_LAGS)
    parser.add_argument("--target-ber", type=float, default=DEFAULT_TARGET_BER)
    parser.add_argument(
        "--max-bits-per-repeat",
        type=int,
        help="Testing/smoke limit; omit to use every generated bit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modulation = uwb.load_uwb_modulation(args.modulation_input)
    output_directory = args.output_dir or default_output_directory(
        modulation.realized_bit_rate_bps
    )
    models = build_all_channel_models(
        args.input_dir,
        modulation,
        fft_length=args.fft_length,
        max_isi_lags=args.max_isi_lags,
    )
    print(
        f"[BER-04] channels={len(models)}, rate="
        f"{modulation.realized_bit_rate_bps / 1e6:g} Mbps, "
        f"repeats={modulation.repeat_count}, "
        f"bits/repeat={modulation.bits_per_repeat:,}"
    )
    print(
        "[BER-04] received energy range="
        f"{min(model.received_pulse_energy_db for model in models):.3f} to "
        f"{max(model.received_pulse_energy_db for model in models):.3f} dB; "
        "max |ISI/main|="
        f"{max(model.max_abs_isi_ratio for model in models):.3e}"
    )
    error_counts, evaluated_bits = run_ber_monte_carlo(
        modulation,
        models,
        args.ebn0_db,
        noise_master_seed=args.noise_seed,
        max_bits_per_repeat=args.max_bits_per_repeat,
    )
    aggregates = aggregate_ber_results(
        models,
        args.ebn0_db,
        error_counts,
        evaluated_bits,
    )
    files = write_results(
        output_directory,
        args.modulation_input,
        modulation,
        models,
        args.ebn0_db,
        error_counts,
        evaluated_bits,
        aggregates,
        noise_master_seed=args.noise_seed,
        fft_length=args.fft_length,
        max_isi_lags=args.max_isi_lags,
        overwrite=args.overwrite,
    )
    files["figure"] = plot_ber_curves(
        aggregates,
        Path(output_directory) / "ber_curves.png",
        target_ber=args.target_ber,
    )
    print_summary(aggregates, target_ber=args.target_ber)
    for name, path in files.items():
        print(f"[BER-04] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
