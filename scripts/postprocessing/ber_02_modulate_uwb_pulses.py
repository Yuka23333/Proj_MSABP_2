"""Map BER bit batches onto a sparse 3.1--4.8 GHz UWB pulse train.

The full RF waveform is intentionally not materialized: at 5 Mbps and
24 GSa/s the default batch would require about 80 GiB as float32.  The saved
representation is lossless and consists of antipodal symbols, one unit-energy
pulse template, and the integer number of RF samples per bit::

    x_r[n] = sum_k symbols[r, k] * pulse[n - k * samples_per_bit]

A short dense preview is included for visual inspection.  Later BER stages can
render or convolve this representation block by block.
"""

from __future__ import annotations

import argparse
import hashlib
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

from scripts.postprocessing import ber_01_generate_binary_sequences as binary_source  # noqa: E402


# F5 defaults.
DEFAULT_BINARY_INPUT = binary_source.DEFAULT_OUTPUT_PATH
DEFAULT_BIT_RATE_BPS = 5.0e6
DEFAULT_SAMPLE_RATE_HZ = 24.0e9
DEFAULT_BAND_LOW_HZ = 3.1e9
DEFAULT_BAND_HIGH_HZ = 4.8e9
DEFAULT_EDGE_ATTENUATION_DB = 10.0
DEFAULT_TRUNCATION_SIGMA = 6.0
DEFAULT_PREVIEW_BITS = 8
SCHEMA_VERSION = 1
MODULATION_NAME = "antipodal_bpsk_pulse"
REPRESENTATION_NAME = "sparse_symbol_pulse_train"


def _rate_token(bit_rate_bps: float) -> str:
    rate_mbps = float(bit_rate_bps) / 1.0e6
    return format(rate_mbps, ".12g").replace("-", "m").replace(".", "p")


def default_output_path(bit_rate_bps: float = DEFAULT_BIT_RATE_BPS) -> Path:
    return (
        REPOSITORY_ROOT
        / "results"
        / "processed"
        / "propagation_s21_14"
        / "ber_inputs"
        / f"uwb_bpsk_{_rate_token(bit_rate_bps)}Mbps_3p1-4p8GHz.npz"
    )


DEFAULT_OUTPUT_PATH = default_output_path()


@dataclass(frozen=True)
class UwbPulseDesign:
    time_s: np.ndarray
    samples: np.ndarray
    band_low_hz: float
    band_high_hz: float
    center_frequency_hz: float
    gaussian_sigma_s: float
    edge_attenuation_db: float
    truncation_sigma: float
    lower_edge_db: float
    upper_edge_db: float
    in_band_energy_fraction: float

    @property
    def center_index(self) -> int:
        return int(self.samples.size // 2)

    @property
    def discrete_energy(self) -> float:
        return float(np.dot(self.samples, self.samples))


@dataclass(frozen=True)
class UwbModulationBatch:
    symbols: np.ndarray
    repeat_seeds: np.ndarray
    master_seed: int
    source_bit_sha256: str
    source_binary_path: str
    pulse: UwbPulseDesign
    sample_rate_hz: float
    requested_bit_rate_bps: float
    realized_bit_rate_bps: float
    samples_per_bit: int
    preview_time_s: np.ndarray
    preview_waveform: np.ndarray
    preview_bit_count: int

    @property
    def repeat_count(self) -> int:
        return int(self.symbols.shape[0])

    @property
    def bits_per_repeat(self) -> int:
        return int(self.symbols.shape[1])


def _spectral_amplitude(pulse: np.ndarray, time_s: np.ndarray, frequency_hz: float) -> float:
    phase = np.exp(-2j * np.pi * float(frequency_hz) * time_s)
    return float(abs(np.dot(pulse, phase)))


def _in_band_energy_fraction(
    pulse: np.ndarray,
    sample_rate_hz: float,
    band_low_hz: float,
    band_high_hz: float,
) -> float:
    n_fft = 1 << max(16, int(math.ceil(math.log2(pulse.size * 64))))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    spectrum_power = np.abs(np.fft.rfft(pulse, n=n_fft)) ** 2
    selected = (frequencies >= band_low_hz) & (frequencies <= band_high_hz)
    total = float(spectrum_power.sum())
    if total <= 0.0:
        raise RuntimeError("designed UWB pulse has zero spectral energy")
    return float(spectrum_power[selected].sum() / total)


def design_gaussian_uwb_pulse(
    *,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    band_low_hz: float = DEFAULT_BAND_LOW_HZ,
    band_high_hz: float = DEFAULT_BAND_HIGH_HZ,
    edge_attenuation_db: float = DEFAULT_EDGE_ATTENUATION_DB,
    truncation_sigma: float = DEFAULT_TRUNCATION_SIGMA,
) -> UwbPulseDesign:
    """Create a real, discrete-unit-energy Gaussian-windowed RF pulse."""

    sample_rate_hz = float(sample_rate_hz)
    band_low_hz = float(band_low_hz)
    band_high_hz = float(band_high_hz)
    edge_attenuation_db = float(edge_attenuation_db)
    truncation_sigma = float(truncation_sigma)
    if not 0.0 < band_low_hz < band_high_hz:
        raise ValueError("UWB band must satisfy 0 < low < high")
    if sample_rate_hz <= 2.0 * band_high_hz:
        raise ValueError("sample_rate_hz must exceed twice the upper band edge")
    if edge_attenuation_db <= 0.0:
        raise ValueError("edge_attenuation_db must be positive")
    if truncation_sigma < 3.0:
        raise ValueError("truncation_sigma must be at least 3")

    center_frequency_hz = (band_low_hz + band_high_hz) / 2.0
    half_bandwidth_hz = (band_high_hz - band_low_hz) / 2.0
    gaussian_sigma_s = math.sqrt(
        edge_attenuation_db * math.log(10.0)
        / (40.0 * math.pi**2 * half_bandwidth_hz**2)
    )
    half_sample_count = max(
        1,
        int(math.ceil(truncation_sigma * gaussian_sigma_s * sample_rate_hz)),
    )
    sample_indices = np.arange(-half_sample_count, half_sample_count + 1)
    time_s = sample_indices.astype(np.float64) / sample_rate_hz
    envelope = np.exp(-0.5 * (time_s / gaussian_sigma_s) ** 2)
    raw_pulse = envelope * np.cos(2.0 * np.pi * center_frequency_hz * time_s)
    pulse = raw_pulse / np.linalg.norm(raw_pulse)

    center_amplitude = _spectral_amplitude(pulse, time_s, center_frequency_hz)
    lower_edge_db = 20.0 * math.log10(
        _spectral_amplitude(pulse, time_s, band_low_hz) / center_amplitude
    )
    upper_edge_db = 20.0 * math.log10(
        _spectral_amplitude(pulse, time_s, band_high_hz) / center_amplitude
    )
    return UwbPulseDesign(
        time_s=time_s,
        samples=np.asarray(pulse, dtype=np.float64),
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        center_frequency_hz=center_frequency_hz,
        gaussian_sigma_s=gaussian_sigma_s,
        edge_attenuation_db=edge_attenuation_db,
        truncation_sigma=truncation_sigma,
        lower_edge_db=lower_edge_db,
        upper_edge_db=upper_edge_db,
        in_band_energy_fraction=_in_band_energy_fraction(
            pulse,
            sample_rate_hz,
            band_low_hz,
            band_high_hz,
        ),
    )


def bits_to_antipodal_symbols(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.dtype != np.uint8:
        raise ValueError("bits must be a two-dimensional uint8 array")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits contains a value outside {0, 1}")
    return np.asarray(bits * np.int8(2) - np.int8(1), dtype=np.int8)


def render_pulse_train(
    symbols: np.ndarray,
    pulse: np.ndarray,
    samples_per_bit: int,
    *,
    max_output_samples: int = 10_000_000,
) -> np.ndarray:
    """Render a one-dimensional symbol slice, normally only for previews."""

    symbols = np.asarray(symbols)
    pulse = np.asarray(pulse, dtype=np.float64)
    samples_per_bit = int(samples_per_bit)
    if symbols.ndim != 1 or symbols.size <= 0:
        raise ValueError("symbols must be a non-empty one-dimensional array")
    if np.any((symbols != -1) & (symbols != 1)):
        raise ValueError("symbols must contain only -1 and +1")
    if pulse.ndim != 1 or pulse.size <= 0:
        raise ValueError("pulse must be a non-empty one-dimensional array")
    if samples_per_bit <= 0:
        raise ValueError("samples_per_bit must be positive")
    output_samples = (symbols.size - 1) * samples_per_bit + pulse.size
    if output_samples > int(max_output_samples):
        raise ValueError(
            f"dense waveform would contain {output_samples:,} samples; "
            "use block processing instead"
        )
    waveform = np.zeros(output_samples, dtype=np.float64)
    for index, symbol in enumerate(symbols):
        start = index * samples_per_bit
        waveform[start : start + pulse.size] += float(symbol) * pulse
    return waveform


def build_uwb_modulation(
    binary_batch: binary_source.BinarySequenceBatch,
    *,
    source_binary_path: str | Path,
    bit_rate_bps: float = DEFAULT_BIT_RATE_BPS,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    band_low_hz: float = DEFAULT_BAND_LOW_HZ,
    band_high_hz: float = DEFAULT_BAND_HIGH_HZ,
    edge_attenuation_db: float = DEFAULT_EDGE_ATTENUATION_DB,
    truncation_sigma: float = DEFAULT_TRUNCATION_SIGMA,
    preview_bits: int = DEFAULT_PREVIEW_BITS,
) -> UwbModulationBatch:
    binary_source.validate_binary_sequence_batch(binary_batch)
    bit_rate_bps = float(bit_rate_bps)
    sample_rate_hz = float(sample_rate_hz)
    if bit_rate_bps <= 0.0 or bit_rate_bps >= sample_rate_hz:
        raise ValueError("bit_rate_bps must satisfy 0 < rate < sample rate")
    samples_per_bit = max(1, int(round(sample_rate_hz / bit_rate_bps)))
    realized_bit_rate_bps = sample_rate_hz / samples_per_bit
    preview_bit_count = min(int(preview_bits), binary_batch.bits_per_repeat)
    if preview_bit_count <= 0:
        raise ValueError("preview_bits must be positive")

    pulse = design_gaussian_uwb_pulse(
        sample_rate_hz=sample_rate_hz,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        edge_attenuation_db=edge_attenuation_db,
        truncation_sigma=truncation_sigma,
    )
    symbols = bits_to_antipodal_symbols(binary_batch.bits)
    preview_waveform = render_pulse_train(
        symbols[0, :preview_bit_count],
        pulse.samples,
        samples_per_bit,
    )
    preview_time_s = np.arange(preview_waveform.size, dtype=np.float64) / sample_rate_hz
    try:
        source_name = str(
            Path(source_binary_path).expanduser().resolve().relative_to(REPOSITORY_ROOT)
        )
    except ValueError:
        source_name = str(Path(source_binary_path).expanduser().resolve())
    return UwbModulationBatch(
        symbols=symbols,
        repeat_seeds=binary_batch.repeat_seeds.copy(),
        master_seed=binary_batch.master_seed,
        source_bit_sha256=binary_batch.bit_sha256,
        source_binary_path=source_name,
        pulse=pulse,
        sample_rate_hz=sample_rate_hz,
        requested_bit_rate_bps=bit_rate_bps,
        realized_bit_rate_bps=realized_bit_rate_bps,
        samples_per_bit=samples_per_bit,
        preview_time_s=preview_time_s,
        preview_waveform=preview_waveform,
        preview_bit_count=preview_bit_count,
    )


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def validate_uwb_modulation(batch: UwbModulationBatch) -> None:
    symbols = np.asarray(batch.symbols)
    if symbols.ndim != 2 or symbols.dtype != np.int8 or symbols.size <= 0:
        raise ValueError("symbols must be a non-empty two-dimensional int8 array")
    if np.any((symbols != -1) & (symbols != 1)):
        raise ValueError("symbols contains a value outside {-1, +1}")
    repeat_seeds = np.asarray(batch.repeat_seeds)
    if repeat_seeds.dtype != np.uint64 or repeat_seeds.shape != (symbols.shape[0],):
        raise ValueError("repeat_seeds must contain one uint64 value per repeat")
    if batch.samples_per_bit <= 0:
        raise ValueError("samples_per_bit must be positive")
    if not math.isclose(
        batch.realized_bit_rate_bps,
        batch.sample_rate_hz / batch.samples_per_bit,
        rel_tol=1e-12,
    ):
        raise ValueError("realized bit rate is inconsistent with the sampling grid")
    if batch.sample_rate_hz <= 2.0 * batch.pulse.band_high_hz:
        raise ValueError("sample rate does not satisfy Nyquist for the UWB band")

    pulse_time = np.asarray(batch.pulse.time_s)
    pulse_samples = np.asarray(batch.pulse.samples)
    if (
        pulse_time.ndim != 1
        or pulse_samples.ndim != 1
        or pulse_time.shape != pulse_samples.shape
        or pulse_samples.size % 2 != 1
    ):
        raise ValueError("pulse arrays must have the same non-empty odd length")
    if pulse_time.dtype != np.float64 or pulse_samples.dtype != np.float64:
        raise ValueError("pulse arrays must use float64")
    if not np.all(np.diff(pulse_time) > 0.0):
        raise ValueError("pulse time samples must be strictly increasing")
    if not math.isclose(
        float(pulse_time[batch.pulse.center_index]),
        0.0,
        abs_tol=np.finfo(np.float64).eps,
    ):
        raise ValueError("pulse template is not centered at t=0")
    if not math.isclose(batch.pulse.discrete_energy, 1.0, rel_tol=1e-12):
        raise ValueError("pulse template does not have unit discrete energy")
    if not 0.0 < batch.pulse.in_band_energy_fraction <= 1.0:
        raise ValueError("invalid in-band pulse energy fraction")

    preview_time = np.asarray(batch.preview_time_s)
    preview = np.asarray(batch.preview_waveform)
    if preview_time.shape != preview.shape or preview.ndim != 1 or preview.size <= 0:
        raise ValueError("preview time and waveform arrays are inconsistent")
    if not 0 < batch.preview_bit_count <= batch.bits_per_repeat:
        raise ValueError("preview_bit_count is outside the bit sequence")
    expected_preview_samples = (
        (batch.preview_bit_count - 1) * batch.samples_per_bit + pulse_samples.size
    )
    if preview.size != expected_preview_samples:
        raise ValueError("preview waveform length does not match sparse modulation")
    if len(batch.source_bit_sha256) != 64:
        raise ValueError("source bit SHA-256 is malformed")


def save_uwb_modulation(
    batch: UwbModulationBatch,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    validate_uwb_modulation(batch)
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.casefold() != ".npz":
        raise ValueError(f"UWB modulation output must use .npz: {destination}")
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
            symbols=batch.symbols,
            repeat_seeds=batch.repeat_seeds,
            master_seed=np.asarray(batch.master_seed, dtype=np.uint64),
            source_bit_sha256=np.asarray(batch.source_bit_sha256),
            source_binary_path=np.asarray(batch.source_binary_path),
            pulse_time_s=batch.pulse.time_s,
            pulse_samples=batch.pulse.samples,
            pulse_sha256=np.asarray(_array_sha256(batch.pulse.samples)),
            symbols_sha256=np.asarray(_array_sha256(batch.symbols)),
            sample_rate_hz=np.asarray(batch.sample_rate_hz, dtype=np.float64),
            requested_bit_rate_bps=np.asarray(
                batch.requested_bit_rate_bps, dtype=np.float64
            ),
            realized_bit_rate_bps=np.asarray(
                batch.realized_bit_rate_bps, dtype=np.float64
            ),
            samples_per_bit=np.asarray(batch.samples_per_bit, dtype=np.int64),
            band_low_hz=np.asarray(batch.pulse.band_low_hz, dtype=np.float64),
            band_high_hz=np.asarray(batch.pulse.band_high_hz, dtype=np.float64),
            center_frequency_hz=np.asarray(
                batch.pulse.center_frequency_hz, dtype=np.float64
            ),
            gaussian_sigma_s=np.asarray(
                batch.pulse.gaussian_sigma_s, dtype=np.float64
            ),
            edge_attenuation_db=np.asarray(
                batch.pulse.edge_attenuation_db, dtype=np.float64
            ),
            truncation_sigma=np.asarray(
                batch.pulse.truncation_sigma, dtype=np.float64
            ),
            lower_edge_db=np.asarray(batch.pulse.lower_edge_db, dtype=np.float64),
            upper_edge_db=np.asarray(batch.pulse.upper_edge_db, dtype=np.float64),
            in_band_energy_fraction=np.asarray(
                batch.pulse.in_band_energy_fraction, dtype=np.float64
            ),
            pulse_center_index=np.asarray(batch.pulse.center_index, dtype=np.int64),
            preview_time_s=batch.preview_time_s,
            preview_waveform=batch.preview_waveform,
            preview_bit_count=np.asarray(batch.preview_bit_count, dtype=np.int64),
            repeat_count=np.asarray(batch.repeat_count, dtype=np.int64),
            bits_per_repeat=np.asarray(batch.bits_per_repeat, dtype=np.int64),
            modulation=np.asarray(MODULATION_NAME),
            representation=np.asarray(REPRESENTATION_NAME),
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_uwb_modulation(path: str | Path) -> UwbModulationBatch:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        if int(archive["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"unsupported UWB schema: {int(archive['schema_version'])}")
        if str(archive["modulation"]) != MODULATION_NAME:
            raise ValueError(f"unsupported modulation: {str(archive['modulation'])}")
        if str(archive["representation"]) != REPRESENTATION_NAME:
            raise ValueError(
                f"unsupported representation: {str(archive['representation'])}"
            )
        pulse = UwbPulseDesign(
            time_s=np.array(archive["pulse_time_s"], copy=True),
            samples=np.array(archive["pulse_samples"], copy=True),
            band_low_hz=float(archive["band_low_hz"]),
            band_high_hz=float(archive["band_high_hz"]),
            center_frequency_hz=float(archive["center_frequency_hz"]),
            gaussian_sigma_s=float(archive["gaussian_sigma_s"]),
            edge_attenuation_db=float(archive["edge_attenuation_db"]),
            truncation_sigma=float(archive["truncation_sigma"]),
            lower_edge_db=float(archive["lower_edge_db"]),
            upper_edge_db=float(archive["upper_edge_db"]),
            in_band_energy_fraction=float(archive["in_band_energy_fraction"]),
        )
        batch = UwbModulationBatch(
            symbols=np.array(archive["symbols"], copy=True),
            repeat_seeds=np.array(archive["repeat_seeds"], copy=True),
            master_seed=int(archive["master_seed"]),
            source_bit_sha256=str(archive["source_bit_sha256"]),
            source_binary_path=str(archive["source_binary_path"]),
            pulse=pulse,
            sample_rate_hz=float(archive["sample_rate_hz"]),
            requested_bit_rate_bps=float(archive["requested_bit_rate_bps"]),
            realized_bit_rate_bps=float(archive["realized_bit_rate_bps"]),
            samples_per_bit=int(archive["samples_per_bit"]),
            preview_time_s=np.array(archive["preview_time_s"], copy=True),
            preview_waveform=np.array(archive["preview_waveform"], copy=True),
            preview_bit_count=int(archive["preview_bit_count"]),
        )
        recorded_pulse_sha256 = str(archive["pulse_sha256"])
        recorded_symbols_sha256 = str(archive["symbols_sha256"])
        recorded_shape = (
            int(archive["repeat_count"]),
            int(archive["bits_per_repeat"]),
        )
        recorded_center_index = int(archive["pulse_center_index"])

    validate_uwb_modulation(batch)
    if batch.symbols.shape != recorded_shape:
        raise ValueError("recorded symbol shape does not match the symbol array")
    if batch.pulse.center_index != recorded_center_index:
        raise ValueError("recorded pulse center index is inconsistent")
    if _array_sha256(batch.pulse.samples) != recorded_pulse_sha256:
        raise ValueError("pulse SHA-256 does not match the archive")
    if _array_sha256(batch.symbols) != recorded_symbols_sha256:
        raise ValueError("symbol SHA-256 does not match the archive")
    return batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pulse-modulate BER bits onto a sparse 3.1--4.8 GHz UWB signal."
    )
    parser.add_argument("--binary-input", type=Path, default=DEFAULT_BINARY_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bit-rate-mbps", type=float, default=5.0)
    parser.add_argument("--sample-rate-ghz", type=float, default=24.0)
    parser.add_argument("--band-low-ghz", type=float, default=3.1)
    parser.add_argument("--band-high-ghz", type=float, default=4.8)
    parser.add_argument(
        "--edge-attenuation-db",
        type=float,
        default=DEFAULT_EDGE_ATTENUATION_DB,
    )
    parser.add_argument(
        "--truncation-sigma",
        type=float,
        default=DEFAULT_TRUNCATION_SIGMA,
    )
    parser.add_argument("--preview-bits", type=int, default=DEFAULT_PREVIEW_BITS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bit_rate_bps = args.bit_rate_mbps * 1.0e6
    output = args.output or default_output_path(bit_rate_bps)
    binary_batch = binary_source.load_binary_sequence_batch(args.binary_input)
    modulation = build_uwb_modulation(
        binary_batch,
        source_binary_path=args.binary_input,
        bit_rate_bps=bit_rate_bps,
        sample_rate_hz=args.sample_rate_ghz * 1.0e9,
        band_low_hz=args.band_low_ghz * 1.0e9,
        band_high_hz=args.band_high_ghz * 1.0e9,
        edge_attenuation_db=args.edge_attenuation_db,
        truncation_sigma=args.truncation_sigma,
        preview_bits=args.preview_bits,
    )
    destination = save_uwb_modulation(
        modulation,
        output,
        overwrite=args.overwrite,
    )
    verified = load_uwb_modulation(destination)
    dense_gib = (
        verified.repeat_count
        * ((verified.bits_per_repeat - 1) * verified.samples_per_bit + verified.pulse.samples.size)
        * np.dtype(np.float32).itemsize
        / 1024**3
    )
    print(f"[BER-02] output: {destination}")
    print(
        f"[BER-02] modulation: 0 -> -pulse, 1 -> +pulse; "
        f"rate={verified.realized_bit_rate_bps / 1e6:.9g} Mbps"
    )
    print(
        f"[BER-02] band: {verified.pulse.band_low_hz / 1e9:g}--"
        f"{verified.pulse.band_high_hz / 1e9:g} GHz, "
        f"fc={verified.pulse.center_frequency_hz / 1e9:g} GHz"
    )
    print(
        f"[BER-02] pulse: {verified.pulse.samples.size} samples, "
        f"edge={verified.pulse.lower_edge_db:.3f}/"
        f"{verified.pulse.upper_edge_db:.3f} dB, "
        f"in-band energy={verified.pulse.in_band_energy_fraction:.6f}"
    )
    print(
        f"[BER-02] sparse spacing: {verified.samples_per_bit} samples/bit; "
        f"dense float32 expansion avoided: {dense_gib:.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
