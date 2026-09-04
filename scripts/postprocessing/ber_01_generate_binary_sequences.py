"""Generate the shared reproducible binary sequences for BER experiments.

The baseline and every antenna candidate must reuse the same repeat sequences.
Noise seeds belong to a later BER stage and are intentionally not generated
here.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# F5 defaults: about 419 expected bit errors when the true BER is 1e-4.
DEFAULT_REPEAT_COUNT = 16
DEFAULT_BITS_PER_REPEAT = 2**18
DEFAULT_MASTER_SEED = 20260904
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "propagation_s21_14"
    / "ber_inputs"
    / (
        f"binary_sequences_{DEFAULT_REPEAT_COUNT}x{DEFAULT_BITS_PER_REPEAT}"
        f"_seed{DEFAULT_MASTER_SEED}.npz"
    )
)

SCHEMA_VERSION = 1
BIT_GENERATOR_NAME = "PCG64"


@dataclass(frozen=True)
class BinarySequenceBatch:
    bits: np.ndarray
    repeat_seeds: np.ndarray
    master_seed: int
    bit_generator: str = BIT_GENERATOR_NAME

    @property
    def repeat_count(self) -> int:
        return int(self.bits.shape[0])

    @property
    def bits_per_repeat(self) -> int:
        return int(self.bits.shape[1])

    @property
    def total_bits(self) -> int:
        return int(self.bits.size)

    @property
    def bit_sha256(self) -> str:
        return hashlib.sha256(self.bits.tobytes(order="C")).hexdigest()


def _validate_generation_request(
    repeat_count: int,
    bits_per_repeat: int,
    master_seed: int,
) -> None:
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")
    if bits_per_repeat <= 0:
        raise ValueError("bits_per_repeat must be positive")
    if not 0 <= master_seed <= np.iinfo(np.uint64).max:
        raise ValueError("master_seed must fit in an unsigned 64-bit integer")


def validate_binary_sequence_batch(batch: BinarySequenceBatch) -> None:
    bits = np.asarray(batch.bits)
    repeat_seeds = np.asarray(batch.repeat_seeds)
    if bits.ndim != 2:
        raise ValueError(f"bits must be a two-dimensional array, got {bits.shape}")
    if bits.dtype != np.uint8:
        raise ValueError(f"bits must use uint8, got {bits.dtype}")
    if bits.shape[0] <= 0 or bits.shape[1] <= 0:
        raise ValueError(f"bits must be non-empty, got {bits.shape}")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits contains a value outside the binary alphabet {0, 1}")
    if repeat_seeds.dtype != np.uint64 or repeat_seeds.shape != (bits.shape[0],):
        raise ValueError(
            "repeat_seeds must be a uint64 vector with one value per repeat"
        )
    if np.unique(repeat_seeds).size != repeat_seeds.size:
        raise ValueError("repeat_seeds contains duplicate values")
    if batch.bit_generator != BIT_GENERATOR_NAME:
        raise ValueError(
            f"unsupported bit generator: expected {BIT_GENERATOR_NAME}, "
            f"got {batch.bit_generator}"
        )
    _validate_generation_request(
        batch.repeat_count,
        batch.bits_per_repeat,
        int(batch.master_seed),
    )


def generate_binary_sequences(
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    bits_per_repeat: int = DEFAULT_BITS_PER_REPEAT,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> BinarySequenceBatch:
    """Generate independent repeat streams derived from one recorded seed."""

    repeat_count = int(repeat_count)
    bits_per_repeat = int(bits_per_repeat)
    master_seed = int(master_seed)
    _validate_generation_request(repeat_count, bits_per_repeat, master_seed)

    child_sequences = np.random.SeedSequence(master_seed).spawn(repeat_count)
    repeat_seeds = np.asarray(
        [
            child.generate_state(1, dtype=np.uint64)[0]
            for child in child_sequences
        ],
        dtype=np.uint64,
    )
    bits = np.empty((repeat_count, bits_per_repeat), dtype=np.uint8)
    for index, repeat_seed in enumerate(repeat_seeds):
        generator = np.random.Generator(np.random.PCG64(int(repeat_seed)))
        bits[index] = generator.integers(
            0,
            2,
            size=bits_per_repeat,
            dtype=np.uint8,
        )

    batch = BinarySequenceBatch(
        bits=bits,
        repeat_seeds=repeat_seeds,
        master_seed=master_seed,
    )
    validate_binary_sequence_batch(batch)
    return batch


def save_binary_sequence_batch(
    batch: BinarySequenceBatch,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save a batch without object arrays or pickle data."""

    validate_binary_sequence_batch(batch)
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.casefold() != ".npz":
        raise ValueError(f"binary sequence output must use .npz: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite to replace it: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        np.savez(
            temporary,
            bits=batch.bits,
            repeat_seeds=batch.repeat_seeds,
            master_seed=np.asarray(batch.master_seed, dtype=np.uint64),
            repeat_count=np.asarray(batch.repeat_count, dtype=np.int64),
            bits_per_repeat=np.asarray(batch.bits_per_repeat, dtype=np.int64),
            total_bits=np.asarray(batch.total_bits, dtype=np.int64),
            bit_generator=np.asarray(batch.bit_generator),
            bit_sha256=np.asarray(batch.bit_sha256),
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_binary_sequence_batch(path: str | Path) -> BinarySequenceBatch:
    """Load and verify a previously generated sequence batch."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "bits",
            "repeat_seeds",
            "master_seed",
            "repeat_count",
            "bits_per_repeat",
            "total_bits",
            "bit_generator",
            "bit_sha256",
            "schema_version",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"binary sequence archive is missing: {sorted(missing)}")
        if int(archive["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(
                "unsupported binary sequence schema: "
                f"{int(archive['schema_version'])}"
            )
        batch = BinarySequenceBatch(
            bits=np.array(archive["bits"], copy=True),
            repeat_seeds=np.array(archive["repeat_seeds"], copy=True),
            master_seed=int(archive["master_seed"]),
            bit_generator=str(archive["bit_generator"]),
        )
        recorded_shape = (
            int(archive["repeat_count"]),
            int(archive["bits_per_repeat"]),
        )
        if batch.bits.shape != recorded_shape:
            raise ValueError(
                f"recorded shape {recorded_shape} does not match {batch.bits.shape}"
            )
        if batch.total_bits != int(archive["total_bits"]):
            raise ValueError("recorded total_bits does not match the bit array")
        recorded_sha256 = str(archive["bit_sha256"])

    validate_binary_sequence_batch(batch)
    if batch.bit_sha256 != recorded_sha256:
        raise ValueError("binary sequence SHA-256 does not match the archive")
    return batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible shared binary streams for BER experiments."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument(
        "--bits-per-repeat",
        type=int,
        default=DEFAULT_BITS_PER_REPEAT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch = generate_binary_sequences(
        repeat_count=args.repeats,
        bits_per_repeat=args.bits_per_repeat,
        master_seed=args.seed,
    )
    destination = save_binary_sequence_batch(
        batch,
        args.output,
        overwrite=args.overwrite,
    )
    loaded = load_binary_sequence_batch(destination)
    one_fraction = loaded.bits.mean(axis=1)
    print(f"[BER-01] output: {destination}")
    print(
        f"[BER-01] shape: {loaded.repeat_count} x {loaded.bits_per_repeat:,} "
        f"= {loaded.total_bits:,} bits"
    )
    print(
        f"[BER-01] ones fraction: total={loaded.bits.mean():.8f}, "
        f"repeat min={one_fraction.min():.8f}, max={one_fraction.max():.8f}"
    )
    print(f"[BER-01] bit SHA-256: {loaded.bit_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
