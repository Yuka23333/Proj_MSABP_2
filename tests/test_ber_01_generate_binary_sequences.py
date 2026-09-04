from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.postprocessing import ber_01_generate_binary_sequences as binary_generator


def test_default_campaign_size_matches_ber_precision_plan() -> None:
    assert binary_generator.DEFAULT_REPEAT_COUNT == 16
    assert binary_generator.DEFAULT_BITS_PER_REPEAT == 2**18
    assert (
        binary_generator.DEFAULT_REPEAT_COUNT
        * binary_generator.DEFAULT_BITS_PER_REPEAT
        == 2**22
    )


def test_generation_is_deterministic_and_uses_independent_repeat_seeds() -> None:
    first = binary_generator.generate_binary_sequences(4, 1024, 12345)
    second = binary_generator.generate_binary_sequences(4, 1024, 12345)
    different = binary_generator.generate_binary_sequences(4, 1024, 12346)

    np.testing.assert_array_equal(first.bits, second.bits)
    np.testing.assert_array_equal(first.repeat_seeds, second.repeat_seeds)
    assert first.bit_sha256 == second.bit_sha256
    assert np.unique(first.repeat_seeds).size == 4
    assert not np.array_equal(first.bits, different.bits)
    assert first.bits.dtype == np.uint8
    assert set(np.unique(first.bits)) == {0, 1}


def test_saved_archive_round_trips_without_pickle(tmp_path: Path) -> None:
    batch = binary_generator.generate_binary_sequences(3, 257, 9876)
    output = tmp_path / "binary_sequences.npz"

    saved = binary_generator.save_binary_sequence_batch(batch, output)
    loaded = binary_generator.load_binary_sequence_batch(saved)

    assert loaded.master_seed == batch.master_seed
    assert loaded.bit_generator == batch.bit_generator
    assert loaded.bit_sha256 == batch.bit_sha256
    np.testing.assert_array_equal(loaded.bits, batch.bits)
    np.testing.assert_array_equal(loaded.repeat_seeds, batch.repeat_seeds)

    with np.load(saved, allow_pickle=False) as archive:
        assert archive["bits"].dtype == np.uint8
        assert archive["repeat_seeds"].dtype == np.uint64
        assert int(archive["total_bits"]) == 3 * 257


def test_save_refuses_to_replace_an_existing_archive(tmp_path: Path) -> None:
    batch = binary_generator.generate_binary_sequences(2, 32, 42)
    output = tmp_path / "binary_sequences.npz"
    output.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError, match="--overwrite"):
        binary_generator.save_binary_sequence_batch(batch, output)
    assert output.read_bytes() == b"keep-me"

    binary_generator.save_binary_sequence_batch(batch, output, overwrite=True)
    assert binary_generator.load_binary_sequence_batch(output).total_bits == 64


def test_loader_rejects_non_binary_or_wrong_dtype_archive(tmp_path: Path) -> None:
    output = tmp_path / "invalid.npz"
    np.savez(
        output,
        bits=np.asarray([[0.0, 0.5], [1.0, 0.0]], dtype=np.float64),
        repeat_seeds=np.asarray([1, 2], dtype=np.uint64),
        master_seed=np.asarray(1, dtype=np.uint64),
        repeat_count=np.asarray(2, dtype=np.int64),
        bits_per_repeat=np.asarray(2, dtype=np.int64),
        total_bits=np.asarray(4, dtype=np.int64),
        bit_generator=np.asarray(binary_generator.BIT_GENERATOR_NAME),
        bit_sha256=np.asarray("invalid"),
        schema_version=np.asarray(binary_generator.SCHEMA_VERSION, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="bits must use uint8"):
        binary_generator.load_binary_sequence_batch(output)


def test_cli_main_generates_and_verifies_a_small_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cli_sequences.npz"

    exit_code = binary_generator.main(
        [
            "--output",
            str(output),
            "--repeats",
            "2",
            "--bits-per-repeat",
            "128",
            "--seed",
            "2026",
        ]
    )

    assert exit_code == 0
    assert binary_generator.load_binary_sequence_batch(output).bits.shape == (2, 128)
    stdout = capsys.readouterr().out
    assert "[BER-01] shape: 2 x 128 = 256 bits" in stdout
    assert "bit SHA-256" in stdout

