"""Run one two-antenna propagation case without touching CST infrastructure.

The device-local project template owns the SMA connectors, ports, Muscle
phantom, solver setup, and E-field monitors.  This runner only rebuilds the two
managed antenna components (metal, substrate, and reflector), solves, returns
complex S21, and retains native E-field files on the Maid host.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.automation import antenna_sampler
from scripts.automation import cst_build_msabp_geometry
from scripts.automation import cst_run_and_export_s11
from scripts.automation.cst_generate_polygen import execute_project_vba
from scripts.postprocessing import export_propagation_results


SIMULATION_MODE = "propagation_s21"
SOURCE_COMPONENT = "component1"
SECOND_COMPONENT = "component1_1"
SECOND_ANTENNA_OFFSET_Y_MM = 300.0
S21_FILENAME = "S21.csv"
MIN_E_FIELD_MONITORS = 3


def _transform_second_antenna_vba() -> str:
    return f'''Sub Main()
    With Transform
        .Reset
        .Name "{SECOND_COMPONENT}"
        .Origin "Free"
        .Center "0", "0", "0"
        .PlaneNormal "0", "1", "0"
        .MultipleObjects "False"
        .GroupObjects "False"
        .Repetitions "1"
        .MultipleSelection "False"
        .Destination ""
        .Material ""
        .AutoDestination "True"
        .Transform "Shape", "Mirror"
    End With

    With Transform
        .Reset
        .Name "{SECOND_COMPONENT}"
        .Vector "0", "{SECOND_ANTENNA_OFFSET_Y_MM:g}", "0"
        .UsePickedPoints "False"
        .InvertPickedPoints "False"
        .MultipleObjects "False"
        .GroupObjects "False"
        .Repetitions "1"
        .MultipleSelection "False"
        .AutoDestination "True"
        .Transform "Shape", "Translate"
    End With
End Sub'''


def inspect_propagation_infrastructure(
    project: Any,
    timeout: float | None,
) -> dict[str, Any]:
    """Read-only validation of infrastructure that this runner must preserve."""

    model3d = project.model3d
    if model3d is None:
        raise RuntimeError("CST project does not expose a 3D modeler")
    tree_items = tuple(str(item) for item in model3d.get_tree_items(timeout=timeout))
    folded = {item.casefold(): item for item in tree_items}
    missing_ports = [
        port for port in (r"Ports\port1", r"Ports\port2")
        if port.casefold() not in folded
    ]
    if missing_ports:
        raise RuntimeError(
            "propagation template is missing manually configured ports: "
            + ", ".join(missing_ports)
        )
    monitors = tuple(
        item
        for item in tree_items
        if item.casefold().startswith("field monitors\\")
        and "e-field" in item.casefold()
    )
    if len(monitors) < MIN_E_FIELD_MONITORS:
        raise RuntimeError(
            "propagation template must retain at least "
            f"{MIN_E_FIELD_MONITORS} E-field monitors; found {len(monitors)}"
        )
    solver_name = str(model3d.get_active_solver_name(timeout=timeout))
    if solver_name != cst_run_and_export_s11.EXPECTED_SOLVER_NAME:
        raise RuntimeError(
            "unexpected active CST solver: "
            f"expected={cst_run_and_export_s11.EXPECTED_SOLVER_NAME}, "
            f"actual={solver_name}"
        )
    if bool(model3d.is_solver_running(timeout=timeout)):
        raise RuntimeError("CST solver is already running")
    return {
        "ports": (folded[r"ports\port1"], folded[r"ports\port2"]),
        "e_field_monitors": monitors,
        "solver_name": solver_name,
    }


def run_csv_row(
    row: Mapping[str, Any],
    *,
    project_path: str | Path,
    output_root: str | Path,
    local_artifact_root: str | Path | None,
    project: Any | None = None,
    case_id: str | int | None = None,
    id_width: int = 4,
    coordinate_quantum_mm: float = 0.01,
    allow_disconnected_conductor: bool = False,
    command_timeout: float | None = 15.0,
    overwrite: bool = False,
    save_project_after_case: bool = False,
    dry_run: bool = False,
    stage_callback: Any | None = None,
) -> Any:
    """Execute one propagation row and return a ``CaseRunResult``."""

    # Local import avoids a module cycle while sharing the established manifest
    # and result protocol with the ordinary antenna-characterization runner.
    from . import case_runner

    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    resolved_case_id = case_runner._case_id_from_row(row, case_id)
    try:
        if "geometry_valid" in row and not case_runner._parse_csv_bool(
            row["geometry_valid"], "geometry_valid"
        ):
            detail = str(row.get("geometry_error", "")).strip()
            raise ValueError(detail or "sampler marked this geometry invalid")
        parameters = antenna_sampler.parameters_from_csv_row(row)
        substrate_material_name = str(
            row.get(
                case_runner.SUBSTRATE_MATERIAL_COLUMN,
                cst_build_msabp_geometry.DEFAULT_SUBSTRATE_MATERIAL_NAME,
            )
        ).strip()
        case_runner._notify(stage_callback, "prechecking_geometry")
        specs, preflight_report = cst_build_msabp_geometry.build_sampled_polygon_specs(
            parameters,
            coordinate_quantum_mm=coordinate_quantum_mm,
        )
        _, preflight_report = cst_build_msabp_geometry.apply_substrate_material(
            specs,
            preflight_report,
            substrate_material_name,
        )
    except Exception as exc:
        raise case_runner.CaseRunError(resolved_case_id, "precheck", str(exc)) from exc

    project_path = Path(project_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    case_directory = output_root / case_runner._case_directory_name(
        resolved_case_id,
        id_width,
    )
    s21_path = case_directory / S21_FILENAME
    manifest_path = case_directory / case_runner.MANIFEST_FILENAME
    collisions = [path for path in (s21_path, manifest_path) if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise case_runner.CaseRunError(
            resolved_case_id,
            "prepare_output",
            f"case artifacts already exist: {names}",
        )
    case_directory.mkdir(parents=True, exist_ok=True)

    if dry_run:
        elapsed_seconds = time.perf_counter() - started_clock
        payload = {
            "schema_version": case_runner.MANIFEST_SCHEMA_VERSION,
            "simulation_mode": SIMULATION_MODE,
            "case_id": resolved_case_id,
            "status": "dry_run",
            "dry_run": True,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "project_path": str(project_path),
            "parameters": asdict(parameters),
            "geometry": asdict(preflight_report),
            "artifacts": {},
        }
        case_runner._notify(stage_callback, "writing_manifest")
        case_runner._write_manifest(manifest_path, payload)
        case_runner._notify(stage_callback, "completed")
        return case_runner.CaseRunResult(
            case_id=resolved_case_id,
            case_directory=case_directory,
            manifest_path=manifest_path,
            s11_path=None,
            farfield_source_path=None,
            dry_run=True,
            elapsed_seconds=elapsed_seconds,
            simulation_mode=SIMULATION_MODE,
        )

    if local_artifact_root is None:
        raise case_runner.CaseRunError(
            resolved_case_id,
            "prepare_output",
            "propagation mode requires a Maid-local artifact root",
        )
    local_case_directory = (
        Path(local_artifact_root).expanduser().resolve()
        / case_runner._case_directory_name(resolved_case_id, id_width)
    )
    if not project_path.is_file():
        raise case_runner.CaseRunError(
            resolved_case_id,
            "open_project",
            f"CST project does not exist: {project_path}",
        )
    if project is None:
        try:
            case_runner._notify(stage_callback, "opening_project")
            project = cst_run_and_export_s11.open_cst_project(str(project_path))
        except Exception as exc:
            raise case_runner.CaseRunError(
                resolved_case_id, "open_project", str(exc)
            ) from exc

    try:
        case_runner._notify(stage_callback, "checking_propagation_infrastructure")
        infrastructure = inspect_propagation_infrastructure(project, command_timeout)
        case_runner._notify(stage_callback, "clearing_results")
        cst_run_and_export_s11.clear_results_on_project(
            project,
            timeout=command_timeout,
        )
    except Exception as exc:
        raise case_runner.CaseRunError(
            resolved_case_id, "check_or_clear_infrastructure", str(exc)
        ) from exc

    try:
        case_runner._notify(stage_callback, "building_source_antenna")
        build_report = cst_build_msabp_geometry.build_msabp_in_cst(
            project_path=project_path,
            component_name=SOURCE_COMPONENT,
            parameters=parameters,
            coordinate_quantum_mm=coordinate_quantum_mm,
            substrate_material_name=substrate_material_name,
            allow_disconnected_conductor=allow_disconnected_conductor,
            timeout=command_timeout,
            project=project,
            save_project=False,
            define_substrate_material=True,
        )
        case_runner._notify(stage_callback, "building_second_antenna")
        second_report = cst_build_msabp_geometry.build_msabp_in_cst(
            project_path=project_path,
            component_name=SECOND_COMPONENT,
            parameters=parameters,
            coordinate_quantum_mm=coordinate_quantum_mm,
            substrate_material_name=substrate_material_name,
            allow_disconnected_conductor=allow_disconnected_conductor,
            timeout=command_timeout,
            project=project,
            save_project=False,
            define_substrate_material=False,
        )
        if second_report != build_report:
            raise RuntimeError("source and second antenna build reports differ")
        case_runner._notify(stage_callback, "positioning_second_antenna")
        execute_project_vba(
            project,
            "mirror and translate second managed antenna",
            _transform_second_antenna_vba(),
            timeout=command_timeout,
        )
    except Exception as exc:
        raise case_runner.CaseRunError(resolved_case_id, "build_pair", str(exc)) from exc

    try:
        case_runner._notify(stage_callback, "solving")
        project.model3d.run_solver(timeout=None)
        case_runner._notify(stage_callback, "retaining_local_e_fields")
        report = export_propagation_results.export_propagation_results(
            project_path,
            local_case_directory,
            excitation_port=1,
            overwrite=True,
            timeout=float(command_timeout or 60.0),
            project=project,
        )
        local_s21_path = local_case_directory / "S21_complex.csv"
        shutil.copy2(local_s21_path, s21_path)
        if save_project_after_case:
            cst_run_and_export_s11.execute_save_project(
                project,
                timeout=command_timeout,
            )
    except Exception as exc:
        raise case_runner.CaseRunError(
            resolved_case_id, "solver_or_propagation_export", str(exc)
        ) from exc

    elapsed_seconds = time.perf_counter() - started_clock
    payload = {
        "schema_version": case_runner.MANIFEST_SCHEMA_VERSION,
        "simulation_mode": SIMULATION_MODE,
        "case_id": resolved_case_id,
        "status": "completed",
        "dry_run": False,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "project_path": str(project_path),
        "parameters": asdict(parameters),
        "geometry": asdict(build_report),
        "infrastructure": infrastructure,
        "local_only": {
            "retained_on_maid": True,
            "transferred_to_princess": False,
            "directory": str(local_case_directory),
            "e_field_monitor_count": report.e_field_monitor_count,
        },
        "artifacts": {
            "s21": case_runner._artifact_record(s21_path, case_directory),
        },
    }
    try:
        case_runner._notify(stage_callback, "writing_manifest")
        case_runner._write_manifest(manifest_path, payload)
    except Exception as exc:
        raise case_runner.CaseRunError(
            resolved_case_id, "write_manifest", str(exc)
        ) from exc
    case_runner._notify(stage_callback, "completed")
    return case_runner.CaseRunResult(
        case_id=resolved_case_id,
        case_directory=case_directory,
        manifest_path=manifest_path,
        s11_path=None,
        farfield_source_path=None,
        dry_run=False,
        elapsed_seconds=elapsed_seconds,
        s21_path=s21_path,
        local_e_field_directory=local_case_directory / "e_field_native",
        simulation_mode=SIMULATION_MODE,
    )
