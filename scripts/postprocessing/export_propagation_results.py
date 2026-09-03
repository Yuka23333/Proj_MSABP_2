"""Export complex S21 and native CST E-field monitor results.

The S21 table is written as an analysis-friendly CSV with the original complex
samples.  E-field monitor data are copied without resampling from CST's native
``.m3d`` result files together with their ``.rex`` metadata sidecars.

Running this file directly uses the propagation baseline project and exports
port-1-excited fields, which is the field state corresponding to S(2,1).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation.cst_generate_polygen import open_cst_project  # noqa: E402


DEFAULT_PROJECT_PATH = (
    REPOSITORY_ROOT / "simulations" / "models" / "msa-bp-propagation.cst"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "raw"
    / "msa-bp-propagation-baseline-001"
)
S21_TREE_PATH = r"1D Results\S-Parameters\S2,1"
E_FIELD_TREE_PATTERN = re.compile(
    r"^2D/3D Results\\E-Field\\e-field \(f=([0-9]+(?:\.[0-9]+)?)\) \[([0-9]+)\]$"
)
COPY_CHUNK_BYTES = 16 * 1024 * 1024
PROGRESS_INTERVAL_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class ExportedFile:
    kind: str
    output_path: str
    size_bytes: int
    source_path: str | None = None
    result_tree_path: str | None = None
    frequency_ghz: float | None = None
    excitation_port: int | None = None


@dataclass(frozen=True)
class PropagationExportReport:
    project_path: str
    output_directory: str
    created_at_utc: str
    s21_tree_path: str
    s21_sample_count: int
    e_field_monitor_count: int
    excitation_port: int
    files: tuple[ExportedFile, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_s21_csv(
    samples: Sequence[tuple[float, complex]],
    output_path: Path,
) -> None:
    temporary = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frequency_ghz",
                "s21_real",
                "s21_imag",
                "s21_magnitude",
                "s21_magnitude_db",
                "s21_phase_deg",
            )
        )
        for frequency_ghz, value in samples:
            magnitude = abs(value)
            magnitude_db = (
                20.0 * math.log10(magnitude) if magnitude > 0.0 else -math.inf
            )
            writer.writerow(
                (
                    format(float(frequency_ghz), ".17g"),
                    format(float(value.real), ".17g"),
                    format(float(value.imag), ".17g"),
                    format(float(magnitude), ".17g"),
                    format(float(magnitude_db), ".17g"),
                    format(float(math.degrees(math.atan2(value.imag, value.real))), ".17g"),
                )
            )
    os.replace(temporary, output_path)


def _load_complex_s21(project_path: Path) -> list[tuple[float, complex]]:
    import cst.results

    project_file = cst.results.ProjectFile(
        str(project_path),
        allow_interactive=True,
    )
    result = project_file.get_3d().get_result_item(S21_TREE_PATH)
    samples = [
        (float(frequency), complex(value))
        for frequency, value in result.get_data()
    ]
    if len(samples) < 2:
        raise RuntimeError(f"S21 contains fewer than two samples: {project_path}")
    if any(
        not (
            math.isfinite(frequency)
            and math.isfinite(value.real)
            and math.isfinite(value.imag)
        )
        for frequency, value in samples
    ):
        raise RuntimeError("S21 contains a non-finite frequency or complex value")
    if any(right[0] <= left[0] for left, right in zip(samples, samples[1:])):
        raise RuntimeError("S21 frequencies are not strictly increasing")
    return samples


def _field_tree_items(
    project_path: Path,
    excitation_port: int,
    requested_frequencies: Sequence[float] | None,
    timeout: float,
    project: Any | None = None,
) -> list[tuple[float, str]]:
    if project is None:
        project = open_cst_project(project_path)
    model3d = project.model3d
    if model3d is None:
        raise RuntimeError("CST project does not expose a 3D modeler")

    matches: list[tuple[float, str]] = []
    for value in model3d.get_tree_items(timeout=timeout):
        tree_path = str(value)
        match = E_FIELD_TREE_PATTERN.fullmatch(tree_path)
        if match is None or int(match.group(2)) != excitation_port:
            continue
        matches.append((float(match.group(1)), tree_path))
    matches.sort(key=lambda item: item[0])

    if requested_frequencies:
        selected: list[tuple[float, str]] = []
        for requested in requested_frequencies:
            candidates = [
                item for item in matches if math.isclose(item[0], requested, abs_tol=1e-9)
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected one port-{excitation_port} E-field result at "
                    f"{requested:g} GHz, found {len(candidates)}"
                )
            selected.append(candidates[0])
        matches = selected

    if not matches:
        raise RuntimeError(
            f"no E-field results were found for excitation port {excitation_port}"
        )
    return matches


def _frequency_token(frequency_ghz: float) -> str:
    text = format(frequency_ghz, ".12g")
    return text.replace("-", "m").replace(".", "p")


def _copy_atomic(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists; pass --overwrite to replace it: {destination}"
            )
        destination.unlink()

    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.partial")
    if temporary.exists():
        temporary.unlink()

    copied = 0
    next_progress = PROGRESS_INTERVAL_BYTES
    total = source.stat().st_size
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            while True:
                block = input_stream.read(COPY_CHUNK_BYTES)
                if not block:
                    break
                output_stream.write(block)
                copied += len(block)
                if copied >= next_progress:
                    print(
                        f"    copied {copied / 1024**3:.2f}/{total / 1024**3:.2f} GiB",
                        flush=True,
                    )
                    next_progress += PROGRESS_INTERVAL_BYTES
        shutil.copystat(source, temporary)
        if copied != total:
            raise RuntimeError(
                f"incomplete copy for {source}: copied={copied}, expected={total}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _native_field_sources(
    project_path: Path,
    frequency_ghz: float,
    excitation_port: int,
) -> tuple[Path, Path]:
    result_directory = project_path.with_suffix("") / "Result"
    filename_pattern = re.compile(
        rf"^e-field \(f=([0-9]+(?:\.[0-9]+)?)\)_{excitation_port},1\.m3d$"
    )
    candidates = []
    for path in result_directory.glob("e-field (f=*)_*,1.m3d"):
        match = filename_pattern.fullmatch(path.name)
        if match is not None and math.isclose(
            float(match.group(1)),
            frequency_ghz,
            abs_tol=1e-9,
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one native port-{excitation_port} E-field file at "
            f"{frequency_ghz:g} GHz, found {len(candidates)}"
        )
    field_path = candidates[0]
    stem = field_path.stem
    metadata_path = result_directory / f"{stem}_m3d.rex"
    for path in (field_path, metadata_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"native CST E-field result is missing: {path}")
    return field_path, metadata_path


def export_propagation_results(
    project_path: str | Path = DEFAULT_PROJECT_PATH,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    excitation_port: int = 1,
    field_frequencies_ghz: Sequence[float] | None = None,
    overwrite: bool = False,
    timeout: float = 60.0,
    project: Any | None = None,
) -> PropagationExportReport:
    """Export complex S21 and exact native E-field monitor data.

    A long-lived Maid may pass its already-open ``project``.  This avoids a
    second CST control connection while retaining the standalone project path
    used by ``cst.results`` and the native ``Result`` files.
    """

    project_path = Path(project_path).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    if not project_path.is_file():
        raise FileNotFoundError(f"CST project does not exist: {project_path}")
    if excitation_port <= 0:
        raise ValueError("excitation_port must be positive")

    output_directory.mkdir(parents=True, exist_ok=True)
    field_directory = output_directory / "e_field_native"
    field_directory.mkdir(parents=True, exist_ok=True)
    s21_path = output_directory / "S21_complex.csv"
    manifest_path = output_directory / "export_manifest.json"
    if not overwrite:
        existing = [path for path in (s21_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"output already exists; pass --overwrite to replace it: {existing[0]}"
            )

    print(f"[export] reading complex S21: {S21_TREE_PATH}")
    s21_samples = _load_complex_s21(project_path)
    _atomic_write_s21_csv(s21_samples, s21_path)
    exported: list[ExportedFile] = [
        ExportedFile(
            kind="complex_s21_csv",
            output_path=str(s21_path),
            size_bytes=s21_path.stat().st_size,
            result_tree_path=S21_TREE_PATH,
        )
    ]

    field_items = _field_tree_items(
        project_path,
        excitation_port,
        field_frequencies_ghz,
        timeout,
        project=project,
    )
    for frequency_ghz, tree_path in field_items:
        source_field, source_metadata = _native_field_sources(
            project_path,
            frequency_ghz,
            excitation_port,
        )
        token = _frequency_token(frequency_ghz)
        output_field = (
            field_directory / f"EField_f{token}GHz_port{excitation_port}.m3d"
        )
        output_metadata = (
            field_directory / f"EField_f{token}GHz_port{excitation_port}_m3d.rex"
        )
        print(f"[export] copying native field: {tree_path}", flush=True)
        _copy_atomic(source_field, output_field, overwrite)
        _copy_atomic(source_metadata, output_metadata, overwrite)
        exported.extend(
            (
                ExportedFile(
                    kind="native_e_field_m3d",
                    output_path=str(output_field),
                    size_bytes=output_field.stat().st_size,
                    source_path=str(source_field),
                    result_tree_path=tree_path,
                    frequency_ghz=frequency_ghz,
                    excitation_port=excitation_port,
                ),
                ExportedFile(
                    kind="native_e_field_metadata",
                    output_path=str(output_metadata),
                    size_bytes=output_metadata.stat().st_size,
                    source_path=str(source_metadata),
                    result_tree_path=tree_path,
                    frequency_ghz=frequency_ghz,
                    excitation_port=excitation_port,
                ),
            )
        )

    report = PropagationExportReport(
        project_path=str(project_path),
        output_directory=str(output_directory),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        s21_tree_path=S21_TREE_PATH,
        s21_sample_count=len(s21_samples),
        e_field_monitor_count=len(field_items),
        excitation_port=excitation_port,
        files=tuple(exported),
    )
    payload = asdict(report)
    payload["s21_sha256"] = _sha256(s21_path)
    _atomic_write_json(payload, manifest_path)
    print(f"[export] manifest: {manifest_path}")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export complex S21 and native CST E-field monitors."
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--excitation-port", type=int, default=1)
    parser.add_argument(
        "--field-frequency-ghz",
        type=float,
        action="append",
        dest="field_frequencies_ghz",
        help="Export only this monitor frequency; repeat for multiple values.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = export_propagation_results(
        args.project,
        args.output_dir,
        excitation_port=args.excitation_port,
        field_frequencies_ghz=args.field_frequencies_ghz,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )
    print(
        f"[export] complete: S21 samples={report.s21_sample_count}, "
        f"E-field monitors={report.e_field_monitor_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
