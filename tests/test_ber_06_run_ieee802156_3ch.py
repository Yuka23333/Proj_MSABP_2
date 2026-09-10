from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.postprocessing import ber_01_generate_binary_sequences as bit_source
from scripts.postprocessing import ber_04_run_experiment as ber_v1
from scripts.postprocessing import ber_06_run_ieee802156_3ch as ber_v2


def _models() -> tuple[tuple[ber_v1.ChannelModel, ...], ...]:
    return tuple(
        (
            ber_v1.ChannelModel(
                case_name="baseline",
                candidate_rank=None,
                received_pulse_energy=energy,
                received_pulse_energy_db=10.0 * np.log10(energy),
                peak_delay_ns=0.0,
                normalized_symbol_taps=np.asarray([1.0]),
                max_abs_isi_ratio=0.0,
                source_s21_path=f"ch{channel_id}.csv",
                source_s21_sha256=str(channel_id) * 64,
            ),
        )
        for channel_id, energy in enumerate((0.1, 0.2, 0.3))
    )


def test_channel_modulations_reuse_bits_and_exact_channel_edges() -> None:
    bits = bit_source.generate_binary_sequences(2, 64, 123)

    batches = ber_v2.build_channel_modulations(
        bits,
        source_binary_path="bits.npz",
        preview_bits=2,
    )

    assert len(batches) == 3
    for batch in batches[1:]:
        np.testing.assert_array_equal(batch.symbols, batches[0].symbols)
    assert batches[0].pulse.band_low_hz / 1e6 == pytest.approx(3244.8)
    assert batches[1].pulse.center_frequency_hz / 1e6 == pytest.approx(3993.6)
    assert batches[2].pulse.band_high_hz / 1e6 == pytest.approx(4742.4)


def test_ber_v2_records_follow_success_utility_formula() -> None:
    models = _models()
    # [channel, case, SNR, repeat]: BERs are 0.2, 0.1, and 0.3.
    counts = np.asarray([[[[20]]], [[[10]]], [[[30]]]], dtype=np.int64)
    evaluated = np.asarray([100], dtype=np.int64)

    channel_records, robust = ber_v2.build_result_records(
        models,
        [40.0],
        counts,
        evaluated,
        robustness_lambda=0.25,
    )

    assert len(channel_records) == 3
    assert len(robust) == 1
    record = robust[0]
    assert record.worst_channel == "ch2"
    assert record.utility_j == pytest.approx(0.75 * 0.9 + 0.25 * 0.7)
    assert record.ber_v2 == pytest.approx(0.75 * 0.1 + 0.25 * 0.3)
    assert record.utility_j + record.ber_v2 == pytest.approx(1.0)


def test_target_summary_uses_aggregate_ber_curve() -> None:
    models = _models()
    # Same BER in all channels: 1e-3 at 40 dB, 1e-5 at 42 dB.
    counts = np.asarray(
        [
            [[[100], [1]]],
            [[[100], [1]]],
            [[[100], [1]]],
        ],
        dtype=np.int64,
    )
    evaluated = np.asarray([100_000], dtype=np.int64)
    channel_records, robust = ber_v2.build_result_records(
        models,
        [40.0, 42.0],
        counts,
        evaluated,
        robustness_lambda=0.6,
    )

    summary = ber_v2.summarize_cases(
        channel_records,
        robust,
        target_ber=1.0e-4,
    )[0]

    assert summary.ebn0_ber_v2_at_target_db == pytest.approx(41.0)
    assert summary.ebn0_ch1_mandatory_at_target_db == pytest.approx(41.0)


def test_small_end_to_end_run_writes_auditable_outputs(tmp_path: Path) -> None:
    binary = bit_source.generate_binary_sequences(2, 128, 456)
    binary_path = tmp_path / "bits.npz"
    bit_source.save_binary_sequence_batch(binary, binary_path)
    case_directory = tmp_path / "cases" / "baseline"
    case_directory.mkdir(parents=True)
    with (case_directory / "S21.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frequency_ghz", "s21_real", "s21_imag"])
        writer.writerows([(0.0, 0.1, 0.0), (6.0, 0.1, 0.0), (12.0, 0.1, 0.0)])
    modulations = ber_v2.build_channel_modulations(
        binary,
        source_binary_path=binary_path,
        bit_rate_bps=100.0e6,
        preview_bits=2,
    )
    models, counts, evaluated = ber_v2.build_models_and_run(
        tmp_path / "cases",
        modulations,
        [40.0, 42.0],
        fft_length=16384,
        max_isi_lags=1,
        max_bits_per_repeat=64,
    )
    channel_records, robust = ber_v2.build_result_records(
        models,
        [40.0, 42.0],
        counts,
        evaluated,
        robustness_lambda=0.5,
    )
    summaries = ber_v2.summarize_cases(
        channel_records,
        robust,
        target_ber=1.0e-4,
    )
    files = ber_v2.write_results(
        tmp_path / "output",
        binary_input=binary_path,
        modulations=modulations,
        models_by_channel=models,
        ebn0_db=[40.0, 42.0],
        error_counts=counts,
        evaluated_bits_per_repeat=evaluated,
        channel_records=channel_records,
        robust_records=robust,
        summaries=summaries,
        robustness_lambda=0.5,
        noise_master_seed=ber_v2.DEFAULT_NOISE_MASTER_SEED,
        fft_length=16384,
        max_isi_lags=1,
    )

    assert all(path.is_file() for path in files.values())
    with files["manifest"].open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["schema_version"] == 2
    assert manifest["channel_plan"][1]["mandatory"] is True
    with np.load(files["raw_npz"], allow_pickle=False) as archive:
        assert archive["error_counts"].shape == (3, 1, 2, 2)
