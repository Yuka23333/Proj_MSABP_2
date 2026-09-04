from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.postprocessing import ber_01_generate_binary_sequences as binary_source
from scripts.postprocessing import ber_02_modulate_uwb_pulses as modulator


def test_default_pulse_matches_requested_uwb_band() -> None:
    pulse = modulator.design_gaussian_uwb_pulse()

    assert pulse.center_frequency_hz == pytest.approx(3.95e9)
    assert pulse.samples.size % 2 == 1
    assert pulse.time_s[pulse.center_index] == pytest.approx(0.0)
    assert pulse.discrete_energy == pytest.approx(1.0, rel=1e-12)
    assert pulse.lower_edge_db == pytest.approx(-10.0, abs=1e-5)
    assert pulse.upper_edge_db == pytest.approx(-10.0, abs=1e-5)
    assert pulse.in_band_energy_fraction == pytest.approx(0.9681, abs=1e-3)


def test_bits_map_to_antipodal_symbols() -> None:
    bits = np.asarray([[0, 1, 1, 0]], dtype=np.uint8)

    symbols = modulator.bits_to_antipodal_symbols(bits)

    assert symbols.dtype == np.int8
    np.testing.assert_array_equal(symbols, [[-1, 1, 1, -1]])


def test_sparse_renderer_places_and_adds_pulses() -> None:
    symbols = np.asarray([1, -1], dtype=np.int8)
    pulse = np.asarray([1.0, 2.0, 1.0])

    waveform = modulator.render_pulse_train(symbols, pulse, samples_per_bit=2)

    np.testing.assert_array_equal(waveform, [1.0, 2.0, 0.0, -2.0, -1.0])


def test_sparse_renderer_refuses_accidental_full_expansion() -> None:
    with pytest.raises(ValueError, match="use block processing"):
        modulator.render_pulse_train(
            np.ones(100, dtype=np.int8),
            np.ones(11),
            samples_per_bit=1000,
            max_output_samples=100,
        )


def test_build_uses_variable_rate_and_preserves_source_identity(tmp_path: Path) -> None:
    binary_batch = binary_source.generate_binary_sequences(2, 64, 123)
    source_path = tmp_path / "bits.npz"
    binary_source.save_binary_sequence_batch(binary_batch, source_path)

    modulation = modulator.build_uwb_modulation(
        binary_batch,
        source_binary_path=source_path,
        bit_rate_bps=10.0e6,
        preview_bits=4,
    )

    assert modulation.symbols.shape == (2, 64)
    assert modulation.samples_per_bit == 2400
    assert modulation.realized_bit_rate_bps == pytest.approx(10.0e6)
    assert modulation.source_bit_sha256 == binary_batch.bit_sha256
    assert modulation.preview_bit_count == 4
    assert modulation.preview_waveform.size == (
        3 * modulation.samples_per_bit + modulation.pulse.samples.size
    )
    np.testing.assert_array_equal(
        modulation.symbols,
        binary_batch.bits.astype(np.int8) * 2 - 1,
    )


def test_modulation_archive_round_trips_without_pickle(tmp_path: Path) -> None:
    binary_batch = binary_source.generate_binary_sequences(2, 32, 456)
    modulation = modulator.build_uwb_modulation(
        binary_batch,
        source_binary_path=tmp_path / "bits.npz",
        preview_bits=3,
    )
    output = tmp_path / "uwb.npz"

    modulator.save_uwb_modulation(modulation, output)
    loaded = modulator.load_uwb_modulation(output)

    np.testing.assert_array_equal(loaded.symbols, modulation.symbols)
    np.testing.assert_array_equal(loaded.repeat_seeds, modulation.repeat_seeds)
    np.testing.assert_array_equal(loaded.pulse.samples, modulation.pulse.samples)
    np.testing.assert_array_equal(loaded.preview_waveform, modulation.preview_waveform)
    assert loaded.source_bit_sha256 == modulation.source_bit_sha256
    assert loaded.samples_per_bit == 4800
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["modulation"]) == modulator.MODULATION_NAME
        assert str(archive["representation"]) == modulator.REPRESENTATION_NAME
        assert archive["symbols"].dtype == np.int8


def test_default_output_name_tracks_variable_rate() -> None:
    assert modulator.default_output_path(5.0e6).name.startswith("uwb_bpsk_5Mbps")
    assert modulator.default_output_path(12.5e6).name.startswith(
        "uwb_bpsk_12p5Mbps"
    )


def test_invalid_sampling_or_band_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="twice the upper band edge"):
        modulator.design_gaussian_uwb_pulse(sample_rate_hz=9.6e9)
    with pytest.raises(ValueError, match="0 < low < high"):
        modulator.design_gaussian_uwb_pulse(
            band_low_hz=4.8e9,
            band_high_hz=3.1e9,
        )


def test_cli_generates_a_small_modulation_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    binary_batch = binary_source.generate_binary_sequences(2, 16, 789)
    binary_path = tmp_path / "bits.npz"
    output_path = tmp_path / "uwb.npz"
    binary_source.save_binary_sequence_batch(binary_batch, binary_path)

    exit_code = modulator.main(
        [
            "--binary-input",
            str(binary_path),
            "--output",
            str(output_path),
            "--preview-bits",
            "2",
        ]
    )

    assert exit_code == 0
    loaded = modulator.load_uwb_modulation(output_path)
    assert loaded.symbols.shape == (2, 16)
    stdout = capsys.readouterr().out
    assert "rate=5 Mbps" in stdout
    assert "dense float32 expansion avoided" in stdout
    assert math.isclose(loaded.pulse.discrete_energy, 1.0)

