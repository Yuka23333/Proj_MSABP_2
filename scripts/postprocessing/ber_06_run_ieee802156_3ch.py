"""Run the BER 2.0 metric over three IEEE 802.15.6 low-band channels.

The three channel allocations exactly use 499.2 MHz bandwidth and centers at
3494.4, 3993.6, and 4492.8 MHz.  Channel 1 is mandatory.  At each transmitter-
reference Eb/N0, define ``Xi = 1 - BERi`` and compute::

    J = (1 - lambda) * X1 + lambda * min(X0, X1, X2)

The exported equivalent lower-is-better metric is therefore::

    BER_v2 = (1 - lambda) * BER1 + lambda * max(BER0, BER1, BER2)

Only the channel allocation and cross-channel score are IEEE-aligned here.
The underlying antipodal pulse and ideal matched-filter receiver intentionally
retain the project's BER 1.x assumptions for controlled comparison.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing import ber_01_generate_binary_sequences as bit_source  # noqa: E402
from scripts.postprocessing import ber_02_modulate_uwb_pulses as modulator  # noqa: E402
from scripts.postprocessing import ber_04_run_experiment as ber_v1  # noqa: E402
from scripts.postprocessing import ieee802156_three_channel as channel_plan  # noqa: E402


DEFAULT_INPUT_DIRECTORY = ber_v1.DEFAULT_INPUT_DIRECTORY
DEFAULT_BINARY_INPUT = bit_source.DEFAULT_OUTPUT_PATH
DEFAULT_OUTPUT_DIRECTORY = DEFAULT_INPUT_DIRECTORY / "ber_v2_ieee802156_3ch"
DEFAULT_BIT_RATE_BPS = modulator.DEFAULT_BIT_RATE_BPS
DEFAULT_SAMPLE_RATE_HZ = modulator.DEFAULT_SAMPLE_RATE_HZ
DEFAULT_EBN0_DB = ber_v1.DEFAULT_EBN0_DB
DEFAULT_NOISE_MASTER_SEED = ber_v1.DEFAULT_NOISE_MASTER_SEED
DEFAULT_FFT_LENGTH = ber_v1.DEFAULT_FFT_LENGTH
DEFAULT_MAX_ISI_LAGS = ber_v1.DEFAULT_MAX_ISI_LAGS
DEFAULT_TARGET_BER = ber_v1.DEFAULT_TARGET_BER
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ChannelBerRecord:
    case_name: str
    candidate_rank: int | None
    channel_id: int
    channel_name: str
    mandatory: bool
    center_frequency_mhz: float
    bandwidth_mhz: float
    tx_reference_ebn0_db: float
    error_count: int
    evaluated_bits: int
    ber: float
    success_probability_x: float
    theoretical_no_isi_ber: float
    received_pulse_energy: float
    received_pulse_energy_db: float
    max_abs_isi_ratio: float


@dataclass(frozen=True)
class RobustBerRecord:
    case_name: str
    candidate_rank: int | None
    tx_reference_ebn0_db: float
    robustness_lambda: float
    ber_ch0: float
    ber_ch1_mandatory: float
    ber_ch2: float
    x0_success: float
    x1_success_mandatory: float
    x2_success: float
    worst_channel: str
    mandatory_success: float
    worst_channel_success: float
    utility_j: float
    ber_v2: float
    theoretical_ber_v2_no_isi: float
    evaluated_bits_per_channel: int


@dataclass(frozen=True)
class CaseSummary:
    case_name: str
    candidate_rank: int | None
    robustness_lambda: float
    target_ber: float
    ebn0_ch0_at_target_db: float | None
    ebn0_ch1_mandatory_at_target_db: float | None
    ebn0_ch2_at_target_db: float | None
    ebn0_ber_v2_at_target_db: float | None


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_channel_modulations(
    binary_batch: bit_source.BinarySequenceBatch,
    *,
    source_binary_path: str | Path,
    bit_rate_bps: float = DEFAULT_BIT_RATE_BPS,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    edge_attenuation_db: float = modulator.DEFAULT_EDGE_ATTENUATION_DB,
    truncation_sigma: float = modulator.DEFAULT_TRUNCATION_SIGMA,
    preview_bits: int = modulator.DEFAULT_PREVIEW_BITS,
) -> tuple[modulator.UwbModulationBatch, ...]:
    """Build three pulses while reusing exactly the same bits and timing."""

    batches = tuple(
        modulator.build_uwb_modulation(
            binary_batch,
            source_binary_path=source_binary_path,
            bit_rate_bps=bit_rate_bps,
            sample_rate_hz=sample_rate_hz,
            band_low_hz=channel.lower_frequency_hz,
            band_high_hz=channel.upper_frequency_hz,
            edge_attenuation_db=edge_attenuation_db,
            truncation_sigma=truncation_sigma,
            preview_bits=preview_bits,
        )
        for channel in channel_plan.LOW_BAND_CHANNELS
    )
    symbol_hashes = {
        hashlib.sha256(batch.symbols.tobytes(order="C")).hexdigest()
        for batch in batches
    }
    if len(symbol_hashes) != 1:
        raise RuntimeError("three channels did not reuse identical symbol streams")
    return batches


def build_models_and_run(
    input_directory: str | Path,
    modulations: Sequence[modulator.UwbModulationBatch],
    ebn0_db: Sequence[float],
    *,
    noise_master_seed: int = DEFAULT_NOISE_MASTER_SEED,
    fft_length: int = DEFAULT_FFT_LENGTH,
    max_isi_lags: int = DEFAULT_MAX_ISI_LAGS,
    max_bits_per_repeat: int | None = None,
) -> tuple[tuple[ber_v1.ChannelModel, ...], np.ndarray, np.ndarray]:
    """Return models and counts shaped [channel, case, SNR, repeat]."""

    if len(modulations) != len(channel_plan.LOW_BAND_CHANNELS):
        raise ValueError("exactly three channel modulations are required")
    all_models: list[tuple[ber_v1.ChannelModel, ...]] = []
    all_counts: list[np.ndarray] = []
    reference_names: tuple[str, ...] | None = None
    reference_evaluated: np.ndarray | None = None
    for modulation in modulations:
        models = tuple(
            ber_v1.build_all_channel_models(
                input_directory,
                modulation,
                fft_length=fft_length,
                max_isi_lags=max_isi_lags,
            )
        )
        case_names = tuple(model.case_name for model in models)
        if reference_names is None:
            reference_names = case_names
        elif case_names != reference_names:
            raise RuntimeError("S21 case ordering changed between channels")
        counts, evaluated = ber_v1.run_ber_monte_carlo(
            modulation,
            models,
            ebn0_db,
            noise_master_seed=noise_master_seed,
            max_bits_per_repeat=max_bits_per_repeat,
        )
        if reference_evaluated is None:
            reference_evaluated = evaluated
        elif not np.array_equal(evaluated, reference_evaluated):
            raise RuntimeError("evaluated bit counts differ between channels")
        all_models.append(models)
        all_counts.append(counts)
    assert reference_evaluated is not None
    return tuple(all_models), np.stack(all_counts, axis=0), reference_evaluated


def build_result_records(
    models_by_channel: Sequence[Sequence[ber_v1.ChannelModel]],
    ebn0_db: Sequence[float],
    error_counts: np.ndarray,
    evaluated_bits_per_repeat: np.ndarray,
    robustness_lambda: float,
) -> tuple[list[ChannelBerRecord], list[RobustBerRecord]]:
    counts = np.asarray(error_counts, dtype=np.int64)
    evaluated = np.asarray(evaluated_bits_per_repeat, dtype=np.int64)
    ebn0 = np.asarray(ebn0_db, dtype=np.float64)
    case_count = len(models_by_channel[0])
    expected = (
        len(channel_plan.LOW_BAND_CHANNELS),
        case_count,
        len(ebn0),
        len(evaluated),
    )
    if counts.shape != expected:
        raise ValueError(f"error-count shape {counts.shape} != {expected}")
    total_bits = int(evaluated.sum())
    if total_bits <= 0:
        raise ValueError("evaluated bit count must be positive")

    per_channel_aggregates = [
        ber_v1.aggregate_ber_results(
            models,
            ebn0,
            counts[channel_index],
            evaluated,
        )
        for channel_index, models in enumerate(models_by_channel)
    ]
    channel_records: list[ChannelBerRecord] = []
    for channel, aggregates in zip(
        channel_plan.LOW_BAND_CHANNELS,
        per_channel_aggregates,
        strict=True,
    ):
        for record in aggregates:
            channel_records.append(
                ChannelBerRecord(
                    case_name=record.case_name,
                    candidate_rank=record.candidate_rank,
                    channel_id=channel.channel_id,
                    channel_name=channel.name,
                    mandatory=channel.mandatory,
                    center_frequency_mhz=channel.center_frequency_hz / 1.0e6,
                    bandwidth_mhz=channel.bandwidth_hz / 1.0e6,
                    tx_reference_ebn0_db=record.tx_reference_ebn0_db,
                    error_count=record.error_count,
                    evaluated_bits=record.evaluated_bits,
                    ber=record.ber,
                    success_probability_x=1.0 - record.ber,
                    theoretical_no_isi_ber=record.theoretical_no_isi_ber,
                    received_pulse_energy=record.received_pulse_energy,
                    received_pulse_energy_db=record.received_pulse_energy_db,
                    max_abs_isi_ratio=record.max_abs_isi_ratio,
                )
            )

    robust_records: list[RobustBerRecord] = []
    for case_index in range(case_count):
        reference_model = models_by_channel[0][case_index]
        for snr_index, snr_db in enumerate(ebn0):
            ber_values = counts[:, case_index, snr_index, :].sum(axis=1) / total_bits
            success_values = 1.0 - ber_values
            theoretical_values = np.asarray(
                [
                    ber_v1._theoretical_no_isi_ber(
                        float(snr_db),
                        models_by_channel[channel_index][
                            case_index
                        ].received_pulse_energy,
                    )
                    for channel_index in range(len(channel_plan.LOW_BAND_CHANNELS))
                ]
            )
            utility_j = float(
                channel_plan.aggregate_mandatory_worst_utility(
                    success_values,
                    robustness_lambda,
                )
            )
            ber_v2 = float(
                channel_plan.aggregate_mandatory_worst_loss(
                    ber_values,
                    robustness_lambda,
                )
            )
            robust_records.append(
                RobustBerRecord(
                    case_name=reference_model.case_name,
                    candidate_rank=reference_model.candidate_rank,
                    tx_reference_ebn0_db=float(snr_db),
                    robustness_lambda=float(robustness_lambda),
                    ber_ch0=float(ber_values[0]),
                    ber_ch1_mandatory=float(ber_values[1]),
                    ber_ch2=float(ber_values[2]),
                    x0_success=float(success_values[0]),
                    x1_success_mandatory=float(success_values[1]),
                    x2_success=float(success_values[2]),
                    worst_channel=channel_plan.LOW_BAND_CHANNELS[
                        int(np.argmax(ber_values))
                    ].name,
                    mandatory_success=float(success_values[1]),
                    worst_channel_success=float(np.min(success_values)),
                    utility_j=utility_j,
                    ber_v2=ber_v2,
                    theoretical_ber_v2_no_isi=float(
                        channel_plan.aggregate_mandatory_worst_loss(
                            theoretical_values,
                            robustness_lambda,
                        )
                    ),
                    evaluated_bits_per_channel=total_bits,
                )
            )
            if not math.isclose(utility_j + ber_v2, 1.0, abs_tol=1.0e-12):
                raise RuntimeError("BER utility/loss complement invariant failed")
    return channel_records, robust_records


def _interpolated_crossing(
    ebn0_db: Sequence[float],
    ber: Sequence[float],
    target_ber: float,
    evaluated_bits: int,
) -> float | None:
    x = np.asarray(ebn0_db, dtype=np.float64)
    y = np.asarray(ber, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("Eb/N0 and BER must be matching vectors")
    floor = 0.5 / int(evaluated_bits)
    plot_ber = np.maximum(y, floor)
    for index in range(len(x) - 1):
        if plot_ber[index] >= target_ber and plot_ber[index + 1] <= target_ber:
            log0 = math.log10(float(plot_ber[index]))
            log1 = math.log10(float(plot_ber[index + 1]))
            if log0 == log1:
                return float(x[index + 1])
            fraction = (math.log10(target_ber) - log0) / (log1 - log0)
            return float(x[index] + fraction * (x[index + 1] - x[index]))
    return None


def summarize_cases(
    channel_records: Sequence[ChannelBerRecord],
    robust_records: Sequence[RobustBerRecord],
    *,
    target_ber: float,
) -> list[CaseSummary]:
    target_ber = float(target_ber)
    if not 0.0 < target_ber < 1.0:
        raise ValueError("target_ber must lie strictly between 0 and 1")
    case_names = list(dict.fromkeys(record.case_name for record in robust_records))
    summaries: list[CaseSummary] = []
    for case_name in case_names:
        robust = sorted(
            (record for record in robust_records if record.case_name == case_name),
            key=lambda record: record.tx_reference_ebn0_db,
        )
        channel_thresholds: list[float | None] = []
        for channel in channel_plan.LOW_BAND_CHANNELS:
            selected = sorted(
                (
                    record
                    for record in channel_records
                    if record.case_name == case_name
                    and record.channel_id == channel.channel_id
                ),
                key=lambda record: record.tx_reference_ebn0_db,
            )
            channel_thresholds.append(
                _interpolated_crossing(
                    [record.tx_reference_ebn0_db for record in selected],
                    [record.ber for record in selected],
                    target_ber,
                    selected[0].evaluated_bits,
                )
            )
        summaries.append(
            CaseSummary(
                case_name=case_name,
                candidate_rank=robust[0].candidate_rank,
                robustness_lambda=robust[0].robustness_lambda,
                target_ber=target_ber,
                ebn0_ch0_at_target_db=channel_thresholds[0],
                ebn0_ch1_mandatory_at_target_db=channel_thresholds[1],
                ebn0_ch2_at_target_db=channel_thresholds[2],
                ebn0_ber_v2_at_target_db=_interpolated_crossing(
                    [record.tx_reference_ebn0_db for record in robust],
                    [record.ber_v2 for record in robust],
                    target_ber,
                    robust[0].evaluated_bits_per_channel,
                ),
            )
        )
    return summaries


def _atomic_csv(rows: Sequence[object], destination: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    dictionaries = [asdict(row) for row in rows]
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(dictionaries[0]))
            writer.writeheader()
            writer.writerows(dictionaries)
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
    *,
    binary_input: str | Path,
    modulations: Sequence[modulator.UwbModulationBatch],
    models_by_channel: Sequence[Sequence[ber_v1.ChannelModel]],
    ebn0_db: Sequence[float],
    error_counts: np.ndarray,
    evaluated_bits_per_repeat: np.ndarray,
    channel_records: Sequence[ChannelBerRecord],
    robust_records: Sequence[RobustBerRecord],
    summaries: Sequence[CaseSummary],
    robustness_lambda: float,
    noise_master_seed: int,
    fft_length: int,
    max_isi_lags: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    destination = Path(output_directory).expanduser().resolve()
    files = {
        "channel_ber_csv": destination / "ber_by_channel.csv",
        "robust_ber_csv": destination / "ber_v2_aggregate.csv",
        "summary_csv": destination / "case_summary.csv",
        "raw_npz": destination / "ber_v2_repeat_counts.npz",
        "manifest": destination / "manifest.json",
    }
    existing = [path for path in files.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "BER 2.0 output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_csv(channel_records, files["channel_ber_csv"])
    _atomic_csv(robust_records, files["robust_ber_csv"])
    _atomic_csv(summaries, files["summary_csv"])
    _atomic_npz(
        {
            "channel_ids": np.asarray([0, 1, 2], dtype=np.int64),
            "center_frequency_hz": np.asarray(
                [item.center_frequency_hz for item in channel_plan.LOW_BAND_CHANNELS]
            ),
            "bandwidth_hz": np.asarray(
                [item.bandwidth_hz for item in channel_plan.LOW_BAND_CHANNELS]
            ),
            "case_names": np.asarray(
                [model.case_name for model in models_by_channel[0]]
            ),
            "tx_reference_ebn0_db": np.asarray(ebn0_db, dtype=np.float64),
            "error_counts": np.asarray(error_counts, dtype=np.int64),
            "evaluated_bits_per_repeat": np.asarray(
                evaluated_bits_per_repeat,
                dtype=np.int64,
            ),
            "robustness_lambda": np.asarray(robustness_lambda, dtype=np.float64),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        },
        files["raw_npz"],
    )
    source_binary = Path(binary_input).expanduser().resolve()
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "metric_name": "BER_2.0_IEEE_802.15.6_three_channel",
        "channel_plan": [
            {
                "channel_id": channel.channel_id,
                "name": channel.name,
                "mandatory": channel.mandatory,
                "center_frequency_mhz": channel.center_frequency_hz / 1.0e6,
                "bandwidth_mhz": channel.bandwidth_hz / 1.0e6,
                "lower_frequency_mhz": channel.lower_frequency_hz / 1.0e6,
                "upper_frequency_mhz": channel.upper_frequency_hz / 1.0e6,
            }
            for channel in channel_plan.LOW_BAND_CHANNELS
        ],
        "utility_definition": "Xi = 1 - BERi",
        "score_definition": "J = (1-lambda)*X1 + lambda*min(X0,X1,X2)",
        "loss_definition": ("BER_v2 = (1-lambda)*BER1 + lambda*max(BER0,BER1,BER2)"),
        "robustness_lambda": float(robustness_lambda),
        "target_ber": float(summaries[0].target_ber),
        "phy_scope_note": (
            "IEEE alignment applies to the three center frequencies and 499.2 MHz "
            "channel bandwidths; BER 1.x antipodal-pulse and matched-filter "
            "assumptions are retained"
        ),
        "bit_rate_bps": modulations[0].realized_bit_rate_bps,
        "sample_rate_hz": modulations[0].sample_rate_hz,
        "edge_attenuation_db": modulations[0].pulse.edge_attenuation_db,
        "fft_length": int(fft_length),
        "max_isi_lags_each_side": int(max_isi_lags),
        "tx_reference_ebn0_db": np.asarray(ebn0_db, dtype=np.float64).tolist(),
        "noise_master_seed": int(noise_master_seed),
        "common_bits_and_noise_across_all_channels_and_cases": True,
        "binary_input": str(source_binary),
        "binary_input_sha256": _sha256(source_binary),
        "case_count": len(models_by_channel[0]),
        "outputs": {name: path.name for name, path in files.items()},
    }
    _atomic_json(manifest, files["manifest"])
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the IEEE 802.15.6 three-channel BER 2.0 metric."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--binary-input", type=Path, default=DEFAULT_BINARY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--bit-rate-mbps", type=float, default=5.0)
    parser.add_argument("--sample-rate-ghz", type=float, default=24.0)
    parser.add_argument(
        "--robustness-lambda",
        type=float,
        required=True,
        help="lambda in [0,1]; required so a scientific trade-off is never implicit",
    )
    parser.add_argument(
        "--ebn0-db",
        type=float,
        nargs="+",
        default=list(DEFAULT_EBN0_DB),
    )
    parser.add_argument("--target-ber", type=float, default=DEFAULT_TARGET_BER)
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_MASTER_SEED)
    parser.add_argument("--fft-length", type=int, default=DEFAULT_FFT_LENGTH)
    parser.add_argument("--max-isi-lags", type=int, default=DEFAULT_MAX_ISI_LAGS)
    parser.add_argument("--max-bits-per-repeat", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binary_batch = bit_source.load_binary_sequence_batch(args.binary_input)
    modulations = build_channel_modulations(
        binary_batch,
        source_binary_path=args.binary_input,
        bit_rate_bps=args.bit_rate_mbps * 1.0e6,
        sample_rate_hz=args.sample_rate_ghz * 1.0e9,
    )
    models, counts, evaluated = build_models_and_run(
        args.input_dir,
        modulations,
        args.ebn0_db,
        noise_master_seed=args.noise_seed,
        fft_length=args.fft_length,
        max_isi_lags=args.max_isi_lags,
        max_bits_per_repeat=args.max_bits_per_repeat,
    )
    channel_records, robust_records = build_result_records(
        models,
        args.ebn0_db,
        counts,
        evaluated,
        args.robustness_lambda,
    )
    summaries = summarize_cases(
        channel_records,
        robust_records,
        target_ber=args.target_ber,
    )
    files = write_results(
        args.output_dir,
        binary_input=args.binary_input,
        modulations=modulations,
        models_by_channel=models,
        ebn0_db=args.ebn0_db,
        error_counts=counts,
        evaluated_bits_per_repeat=evaluated,
        channel_records=channel_records,
        robust_records=robust_records,
        summaries=summaries,
        robustness_lambda=args.robustness_lambda,
        noise_master_seed=args.noise_seed,
        fft_length=args.fft_length,
        max_isi_lags=args.max_isi_lags,
        overwrite=args.overwrite,
    )
    print(
        "[BER-06] BER 2.0: J=(1-lambda)X1+lambda*min(X0,X1,X2), "
        f"lambda={args.robustness_lambda:g}, Xi=1-BERi"
    )
    print(
        "[BER-06] channels: ch0=3494.4, ch1=3993.6 mandatory, "
        "ch2=4492.8 MHz; BW=499.2 MHz"
    )
    for summary in sorted(
        summaries,
        key=lambda item: (
            item.ebn0_ber_v2_at_target_db is None,
            item.ebn0_ber_v2_at_target_db,
            item.case_name,
        ),
    ):
        threshold = (
            "not bracketed"
            if summary.ebn0_ber_v2_at_target_db is None
            else f"{summary.ebn0_ber_v2_at_target_db:.6f} dB"
        )
        print(f"  {summary.case_name:<24} BER_v2 target: {threshold}")
    for name, path in files.items():
        print(f"[BER-06] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
