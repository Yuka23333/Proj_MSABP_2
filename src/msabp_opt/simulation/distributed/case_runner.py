"""Execute one sampler CSV row in a long-lived CST worker session.

This module deliberately owns only the per-case pipeline.  Scheduling, leases,
SSH transport, worker restart policy, and result collection belong to Princess
and Maid.  A caller can pass an already connected CST ``project`` so one Maid
does not reopen the project for every assigned case.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Maid runs without an interactive desktop; geometry preflight never needs a GUI.
os.environ.setdefault("MPLBACKEND", "Agg")

from scripts.automation import antenna_sampler  # noqa: E402
from scripts.automation import cst_build_msabp_geometry  # noqa: E402
from scripts.automation import cst_run_and_export_s11  # noqa: E402


MANIFEST_SCHEMA_VERSION = 1
S11_FILENAME = "S11.csv"
RAD_EFF_FILENAME = cst_run_and_export_s11.RAD_EFF_FILENAME
TOT_EFF_FILENAME = cst_run_and_export_s11.TOT_EFF_FILENAME
FARFIELD_SOURCE_FILENAME = "Farfield Source [1].ffs"
MANIFEST_FILENAME = "manifest.json"
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

StageCallback = Callable[[str], None]


class CaseRunError(RuntimeError):
    """Identify the failed pipeline stage while retaining the original cause."""

    def __init__(self, case_id: str, stage: str, message: str):
        super().__init__(f"case {case_id} failed during {stage}: {message}")
        self.case_id = case_id
        self.stage = stage


@dataclass(frozen=True)
class CaseRunResult:
    case_id: str
    case_directory: Path
    manifest_path: Path
    s11_path: Path | None
    farfield_source_path: Path | None
    dry_run: bool
    elapsed_seconds: float
    rad_eff_path: Path | None = None
    tot_eff_path: Path | None = None


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one result artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_farfield_source_path(project_path: str | Path) -> Path:
    """Return CST's generated FFS location beside a standalone project file."""

    project_path = Path(project_path).expanduser().resolve()
    return project_path.with_suffix("") / "Result" / FARFIELD_SOURCE_FILENAME


def _file_generation_signature(path: Path) -> tuple[int, int, int] | None:
    """Return cheap metadata used to reject a stale CST result file."""

    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _parse_csv_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number) and number in {0.0, 1.0}:
            return bool(number)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{label} must be a boolean, got {value!r}")


def _case_id_from_row(row: Mapping[str, Any], case_id: str | int | None) -> str:
    raw_case_id = row.get("sample_id") if case_id is None else case_id
    if raw_case_id is None or str(raw_case_id).strip() == "":
        raise ValueError("sample row requires a non-empty sample_id")
    resolved = str(raw_case_id).strip()
    if not _SAFE_CASE_ID.fullmatch(resolved) or resolved in {".", ".."}:
        raise ValueError(f"unsafe sample_id for a case directory: {resolved!r}")
    return resolved


def _case_directory_name(case_id: str, id_width: int) -> str:
    if id_width < 1:
        raise ValueError("id_width must be positive")
    if case_id.isdecimal():
        return f"case_{int(case_id):0{id_width}d}"
    return f"case_{case_id}"


def _notify(callback: StageCallback | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _artifact_record(path: Path, case_directory: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(case_directory)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_csv_row(
    row: Mapping[str, Any],
    *,
    project_path: str | Path,
    output_root: str | Path,
    project: Any | None = None,
    case_id: str | int | None = None,
    id_width: int = 4,
    coordinate_quantum_mm: float = 0.01,
    allow_disconnected_conductor: bool = False,
    command_timeout: float | None = 15.0,
    overwrite: bool = False,
    save_project_after_case: bool = False,
    dry_run: bool = False,
    stage_callback: StageCallback | None = None,
) -> CaseRunResult:
    """Run one CSV row through build, solver, 1D/FFS exports, and manifest.

    ``dry_run`` performs the same typed row parsing and quantized geometry
    preflight, then writes a manifest without importing or connecting to CST.
    """

    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    resolved_case_id = _case_id_from_row(row, case_id)

    try:
        if "geometry_valid" in row and not _parse_csv_bool(
            row["geometry_valid"], "geometry_valid"
        ):
            detail = str(row.get("geometry_error", "")).strip()
            raise ValueError(detail or "sampler marked this geometry invalid")
        parameters = antenna_sampler.parameters_from_csv_row(row)
        _notify(stage_callback, "prechecking_geometry")
        _, preflight_report = cst_build_msabp_geometry.build_sampled_polygon_specs(
            parameters,
            coordinate_quantum_mm=coordinate_quantum_mm,
        )
    except Exception as exc:
        raise CaseRunError(resolved_case_id, "precheck", str(exc)) from exc

    project_path = Path(project_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    case_directory = output_root / _case_directory_name(resolved_case_id, id_width)
    s11_path = case_directory / S11_FILENAME
    rad_eff_path = case_directory / RAD_EFF_FILENAME
    tot_eff_path = case_directory / TOT_EFF_FILENAME
    farfield_destination = case_directory / FARFIELD_SOURCE_FILENAME
    manifest_path = case_directory / MANIFEST_FILENAME

    collisions = [
        path
        for path in (
            s11_path,
            rad_eff_path,
            tot_eff_path,
            farfield_destination,
            manifest_path,
        )
        if path.exists()
    ]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise CaseRunError(
            resolved_case_id,
            "prepare_output",
            f"case artifacts already exist: {names}",
        )
    case_directory.mkdir(parents=True, exist_ok=True)

    build_report = preflight_report
    if dry_run:
        completed_at = datetime.now(timezone.utc).isoformat()
        elapsed_seconds = time.perf_counter() - started_clock
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "case_id": resolved_case_id,
            "status": "dry_run",
            "dry_run": True,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "elapsed_seconds": elapsed_seconds,
            "project_path": str(project_path),
            "parameters": asdict(parameters),
            "geometry": asdict(build_report),
            "artifacts": {},
        }
        _notify(stage_callback, "writing_manifest")
        _write_manifest(manifest_path, payload)
        _notify(stage_callback, "completed")
        return CaseRunResult(
            case_id=resolved_case_id,
            case_directory=case_directory,
            manifest_path=manifest_path,
            s11_path=None,
            farfield_source_path=None,
            dry_run=True,
            elapsed_seconds=elapsed_seconds,
        )

    if not project_path.is_file():
        raise CaseRunError(
            resolved_case_id,
            "open_project",
            f"CST project does not exist: {project_path}",
        )

    if project is None:
        try:
            _notify(stage_callback, "opening_project")
            project = cst_run_and_export_s11.open_cst_project(str(project_path))
        except Exception as exc:
            raise CaseRunError(resolved_case_id, "open_project", str(exc)) from exc

    try:
        _notify(stage_callback, "clearing_results")
        cst_run_and_export_s11.clear_results_on_project(
            project,
            timeout=command_timeout,
        )
    except Exception as exc:
        raise CaseRunError(resolved_case_id, "clear_results", str(exc)) from exc

    try:
        _notify(stage_callback, "building_geometry")
        build_report = cst_build_msabp_geometry.build_msabp_in_cst(
            project_path=project_path,
            parameters=parameters,
            coordinate_quantum_mm=coordinate_quantum_mm,
            allow_disconnected_conductor=allow_disconnected_conductor,
            timeout=command_timeout,
            project=project,
            save_project=False,
        )
    except Exception as exc:
        raise CaseRunError(resolved_case_id, "build", str(exc)) from exc

    try:
        _notify(stage_callback, "restoring_simulation_setup")
        cst_run_and_export_s11.restore_recorded_simulation_setup(
            project,
            timeout=command_timeout,
        )
    except Exception as exc:
        raise CaseRunError(
            resolved_case_id,
            "restore_simulation_setup",
            str(exc),
        ) from exc

    farfield_source = project_farfield_source_path(project_path)
    farfield_before = _file_generation_signature(farfield_source)
    try:
        cst_run_and_export_s11.solve_and_export_s11_on_project(
            project,
            s11_path,
            overwrite=overwrite,
            command_timeout=command_timeout,
            save_project=save_project_after_case,
            clear_results=False,
            stage_callback=stage_callback,
        )
    except Exception as exc:
        raise CaseRunError(resolved_case_id, "solver_or_1d_export", str(exc)) from exc

    try:
        _notify(stage_callback, "copying_farfield_source")
        if not farfield_source.is_file() or farfield_source.stat().st_size == 0:
            raise FileNotFoundError(
                f"CST farfield source is missing or empty: {farfield_source}"
            )
        farfield_after = _file_generation_signature(farfield_source)
        if farfield_before is not None and farfield_after == farfield_before:
            raise RuntimeError(
                "CST farfield source was not regenerated by the current solver run: "
                f"{farfield_source}"
            )
        shutil.copy2(farfield_source, farfield_destination)
        if farfield_destination.stat().st_size == 0:
            raise RuntimeError(f"copied farfield source is empty: {farfield_destination}")
    except Exception as exc:
        raise CaseRunError(resolved_case_id, "copy_farfield_source", str(exc)) from exc

    elapsed_seconds = time.perf_counter() - started_clock
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_id": resolved_case_id,
        "status": "completed",
        "dry_run": False,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "project_path": str(project_path),
        "parameters": asdict(parameters),
        "geometry": asdict(build_report),
        "artifacts": {
            "s11": _artifact_record(s11_path, case_directory),
            "rad_eff": _artifact_record(rad_eff_path, case_directory),
            "tot_eff": _artifact_record(tot_eff_path, case_directory),
            "farfield_source": _artifact_record(
                farfield_destination, case_directory
            ),
        },
    }
    try:
        _notify(stage_callback, "writing_manifest")
        _write_manifest(manifest_path, payload)
    except Exception as exc:
        raise CaseRunError(resolved_case_id, "write_manifest", str(exc)) from exc

    _notify(stage_callback, "completed")
    return CaseRunResult(
        case_id=resolved_case_id,
        case_directory=case_directory,
        manifest_path=manifest_path,
        s11_path=s11_path,
        farfield_source_path=farfield_destination,
        dry_run=False,
        elapsed_seconds=elapsed_seconds,
        rad_eff_path=rad_eff_path,
        tot_eff_path=tot_eff_path,
    )
