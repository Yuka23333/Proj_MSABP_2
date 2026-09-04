from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.postprocessing import ber_01_generate_binary_sequences as bit_source
from scripts.postprocessing import ber_02_modulate_uwb_pulses as uwb
from scripts.postprocessing import ber_04_run_experiment as experiment


def _write_constant_s21(path: Path, response: complex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
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
        for frequency_ghz in (0.0, 6.0, 12.0):
            writer.writerow(
                [
                    frequency_ghz,
                    response.real,
                    response.imag,
                    abs(response),
                    20.0 * math.log10(abs(response)),
                    math.degrees(math.atan2(response.imag, response.real)),
                ]
            )


def _small_modulation(bits_per_repeat: int = 4096) -> uwb.UwbModulationBatch:
    bits = bit_source.generate_binary_sequences(
        repeat_count=2,
        bits_per_repeat=bits_per_repeat,
        master_seed=1234,
    )
    return uwb.build_uwb_modulation(
        bits,
        source_binary_path="synthetic_bits.npz",
        bit_rate_bps=100.0e6,
        preview_bits=2,
    )


def test_complex_s21_loader_preserves_real_and_imaginary_parts(tmp_path: Path) -> None:
    source = tmp_path / "S21.csv"
    _write_constant_s21(source, 0.1 + 0.2j)

    loaded = experiment.load_complex_s21(source)

    np.testing.assert_allclose(loaded.response, 0.1 + 0.2j)
    assert loaded.frequency_hz.tolist() == [0.0, 6.0e9, 12.0e9]
    assert len(loaded.sha256) == 64


def test_constant_transfer_scales_received_pulse_energy(tmp_path: Path) -> None:
    modulation = _small_modulation()
    identity = tmp_path / "baseline"
    attenuated = tmp_path / "case_prop_01_rank_07"
    _write_constant_s21(identity / "S21.csv", 1.0 + 0.0j)
    _write_constant_s21(attenuated / "S21.csv", 0.1 + 0.0j)

    identity_model = experiment.build_channel_model(
        identity,
        modulation,
        fft_length=4096,
        max_isi_lags=2,
    )
    attenuated_model = experiment.build_channel_model(
        attenuated,
        modulation,
        fft_length=4096,
        max_isi_lags=2,
    )

    assert identity_model.received_pulse_energy == pytest.approx(1.0, rel=1e-12)
    assert attenuated_model.received_pulse_energy == pytest.approx(0.01, rel=1e-12)
    assert attenuated_model.received_pulse_energy_db == pytest.approx(-20.0)
    assert attenuated_model.candidate_rank == 7
    assert identity_model.normalized_symbol_taps[2] == pytest.approx(1.0)


def test_monte_carlo_retains_channel_loss_and_matches_bpsk_theory() -> None:
    modulation = _small_modulation(bits_per_repeat=100_000)
    strong = experiment.ChannelModel(
        case_name="strong",
        candidate_rank=1,
        received_pulse_energy=1.0,
        received_pulse_energy_db=0.0,
        peak_delay_ns=0.0,
        normalized_symbol_taps=np.asarray([1.0]),
        max_abs_isi_ratio=0.0,
        source_s21_path="strong.csv",
        source_s21_sha256="a" * 64,
    )
    weak = experiment.ChannelModel(
        case_name="weak",
        candidate_rank=2,
        received_pulse_energy=0.25,
        received_pulse_energy_db=10.0 * math.log10(0.25),
        peak_delay_ns=0.0,
        normalized_symbol_taps=np.asarray([1.0]),
        max_abs_isi_ratio=0.0,
        source_s21_path="weak.csv",
        source_s21_sha256="b" * 64,
    )

    errors, evaluated = experiment.run_ber_monte_carlo(
        modulation,
        [strong, weak],
        [5.0],
        noise_master_seed=5678,
    )
    records = experiment.aggregate_ber_results(
        [strong, weak],
        [5.0],
        errors,
        evaluated,
    )

    assert records[0].ber < records[1].ber
    for record in records:
        assert record.ber == pytest.approx(
            record.theoretical_no_isi_ber,
            abs=1.5e-3,
        )


def test_results_round_trip_uses_non_pickle_npz(tmp_path: Path) -> None:
    modulation = _small_modulation(bits_per_repeat=64)
    model = experiment.ChannelModel(
        case_name="baseline",
        candidate_rank=None,
        received_pulse_energy=0.1,
        received_pulse_energy_db=-10.0,
        peak_delay_ns=1.0,
        normalized_symbol_taps=np.asarray([0.0, 1.0, 0.0]),
        max_abs_isi_ratio=0.0,
        source_s21_path="baseline/S21.csv",
        source_s21_sha256="c" * 64,
    )
    errors = np.asarray([[[1, 2]]], dtype=np.int64)
    evaluated = np.asarray([62, 62], dtype=np.int64)
    records = experiment.aggregate_ber_results([model], [10.0], errors, evaluated)
    modulation_path = tmp_path / "modulation.npz"
    uwb.save_uwb_modulation(modulation, modulation_path)

    files = experiment.write_results(
        tmp_path / "output",
        modulation_path,
        modulation,
        [model],
        [10.0],
        errors,
        evaluated,
        records,
        noise_master_seed=7,
        fft_length=4096,
        max_isi_lags=1,
    )

    assert files["ber_csv"].is_file()
    assert files["channel_csv"].is_file()
    assert files["manifest"].is_file()
    with np.load(files["raw_npz"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["error_counts"], errors)
        assert str(archive["case_names"][0]) == "baseline"
    with pytest.raises(FileExistsError, match="--overwrite"):
        experiment.write_results(
            tmp_path / "output",
            modulation_path,
            modulation,
            [model],
            [10.0],
            errors,
            evaluated,
            records,
            noise_master_seed=7,
            fft_length=4096,
            max_isi_lags=1,
        )


def test_target_interpolation_can_use_a_zero_error_endpoint() -> None:
    base = {
        "case_name": "case",
        "candidate_rank": 1,
        "evaluated_bits": 1_000_000,
        "ber_ci95_low": 0.0,
        "ber_ci95_high": 1.0,
        "theoretical_no_isi_ber": 0.0,
        "received_pulse_energy": 0.1,
        "received_pulse_energy_db": -10.0,
        "max_abs_isi_ratio": 0.0,
    }
    records = [
        experiment.BerAggregate(
            **base,
            tx_reference_ebn0_db=40.0,
            error_count=1_000,
            ber=1.0e-3,
        ),
        experiment.BerAggregate(
            **base,
            tx_reference_ebn0_db=42.0,
            error_count=0,
            ber=0.0,
        ),
    ]

    threshold = experiment._interpolated_threshold_ebn0(records, 1.0e-4)

    assert threshold is not None
    assert 40.0 < threshold < 42.0
