"""Read one complex CST S-parameter curve in an isolated Python process.

This module intentionally imports only Python's standard library at module load
time.  ``cst.results`` is loaded lazily inside :func:`load_complex_samples` so
the caller can execute this file in a fresh process, isolated from NumPy,
SciPy, Shapely, ``cst.interface``, and their native DLLs.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


DEFAULT_TREE_PATH = r"1D Results\S-Parameters\S2,1"
CSV_HEADER = (
    "frequency_ghz",
    "s21_real",
    "s21_imag",
    "s21_magnitude",
    "s21_magnitude_db",
    "s21_phase_deg",
)


def validate_complex_samples(rows: Iterable[Sequence[Any]]) -> list[tuple[float, complex]]:
    """Convert CST result rows and enforce the complex-S21 CSV contract."""

    samples: list[tuple[float, complex]] = []
    for index, row in enumerate(rows):
        if len(row) < 2:
            raise ValueError(f"result row {index} contains fewer than two values")
        try:
            frequency = float(row[0])
            value = complex(row[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"result row {index} is not numeric: {row!r}") from exc
        if not (
            math.isfinite(frequency)
            and math.isfinite(value.real)
            and math.isfinite(value.imag)
        ):
            raise ValueError(f"result row {index} contains a non-finite value")
        samples.append((frequency, value))

    if len(samples) < 2:
        raise ValueError("S-parameter result contains fewer than two samples")
    for index, (left, right) in enumerate(zip(samples, samples[1:]), start=1):
        if right[0] <= left[0]:
            raise ValueError(
                "S-parameter frequencies are not strictly increasing at "
                f"rows {index - 1} and {index}: {left[0]!r}, {right[0]!r}"
            )
    return samples


def load_complex_samples(
    project_path: str | Path,
    tree_path: str = DEFAULT_TREE_PATH,
) -> list[tuple[float, complex]]:
    """Read and validate one complex curve through CST's results-only API."""

    project_path = Path(project_path).expanduser().resolve()
    if not project_path.is_file():
        raise FileNotFoundError(f"CST project does not exist: {project_path}")
    if not str(tree_path).strip():
        raise ValueError("tree path must not be empty")

    # Keep this import inside the function.  The production caller launches this
    # worker in a new process specifically to isolate CST's native result DLLs.
    cst_results = importlib.import_module("cst.results")
    project_file = cst_results.ProjectFile(
        str(project_path),
        allow_interactive=True,
    )
    result_item = project_file.get_3d().get_result_item(str(tree_path))
    return validate_complex_samples(result_item.get_data())


def write_complex_csv_atomic(
    samples: Iterable[Sequence[Any]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write complex samples using the established S21 CSV schema."""

    validated = validate_complex_samples(samples)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite to replace it: {output_path}"
        )
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_HEADER)
            for frequency_ghz, value in validated:
                magnitude = abs(value)
                magnitude_db = (
                    20.0 * math.log10(magnitude) if magnitude > 0.0 else -math.inf
                )
                writer.writerow(
                    (
                        format(frequency_ghz, ".17g"),
                        format(value.real, ".17g"),
                        format(value.imag, ".17g"),
                        format(magnitude, ".17g"),
                        format(magnitude_db, ".17g"),
                        format(math.degrees(math.atan2(value.imag, value.real)), ".17g"),
                    )
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return output_path


def export_complex_s21(
    project_path: str | Path,
    output_path: str | Path,
    tree_path: str = DEFAULT_TREE_PATH,
    *,
    overwrite: bool = False,
) -> tuple[Path, int]:
    """Read one CST curve and atomically export it to CSV."""

    samples = load_complex_samples(project_path, tree_path)
    destination = write_complex_csv_atomic(samples, output_path, overwrite=overwrite)
    return destination, len(samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export complex CST S2,1 in a results-only child process."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-path", default=DEFAULT_TREE_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        destination, sample_count = export_complex_s21(
            args.project,
            args.output,
            args.tree_path,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(
            f"[complex-s21-worker] ERROR {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"[complex-s21-worker] exported {sample_count} samples: {destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
