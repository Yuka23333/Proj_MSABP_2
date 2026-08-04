"""Rebuild the managed MSA-BP geometry and recorded CST solve setup.

The standalone ``.cst`` keeps the connector and general model data, but a bare
copy does not reliably restore picks or field monitors.  The pick-dependent
port, monitors, mesh, and solver settings are therefore recreated from the
recorded CST 2025 history before every solve.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from .cst_build_msabp_geometry import DEFAULT_PROJECT_PATH, build_msabp_in_cst
    from .cst_generate_polygen import (
        execute_project_vba,
        execute_save_project,
        open_cst_project,
    )
except ImportError:
    from cst_build_msabp_geometry import (  # type: ignore[no-redef]
        DEFAULT_PROJECT_PATH,
        build_msabp_in_cst,
    )
    from cst_generate_polygen import (  # type: ignore[no-redef]
        execute_project_vba,
        execute_save_project,
        open_cst_project,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "results" / "raw" / "baseline" / "S11.csv"
S11_TREE_PATH = r"1D Results\S-Parameters\S1,1"
DEFAULT_COMMAND_TIMEOUT = 60.0
RECORDED_PORT_NUMBER = 1
RECORDED_PORT_TREE_ITEM = rf"Ports\port{RECORDED_PORT_NUMBER}"
RECORDED_FARFIELD_MIN_GHZ = "2"
RECORDED_FARFIELD_MAX_GHZ = "7"
RECORDED_FARFIELD_STEP_GHZ = "0.1"


def _format_frequency_tenths(value: int) -> str:
    whole, tenths = divmod(value, 10)
    return str(whole) if tenths == 0 else f"{whole}.{tenths}"


RECORDED_FARFIELD_MONITOR_NAMES = tuple(
    f"farfield (f={_format_frequency_tenths(value)})"
    for value in range(20, 71)
)
RECORDED_FARFIELD_MONITOR_TREE_ITEMS = tuple(
    rf"Field Monitors\{name}" for name in RECORDED_FARFIELD_MONITOR_NAMES
)


@dataclass(frozen=True)
class ProjectPrerequisites:
    """Reusable CST-template objects required before starting the solver."""

    solver_name: str
    solver_running: bool
    ports: tuple[str, ...]
    farfield_monitors: tuple[str, ...]


@dataclass(frozen=True)
class S11CurveSummary:
    point_count: int
    frequency_min_ghz: float
    frequency_max_ghz: float
    minimum_db: float
    minimum_frequency_ghz: float


def _vba_string(value: object) -> str:
    return str(value).replace('"', '""')


def build_clear_results_vba() -> str:
    return "Sub Main()\n    DeleteResults\nEnd Sub"


def build_delete_recorded_setup_vba() -> str:
    """Delete the objects recreated from ``History_list_record.txt``."""

    monitor_deletes = "\n".join(
        f'    Monitor.Delete "{name}"'
        for name in RECORDED_FARFIELD_MONITOR_NAMES
    )
    return f'''\
Sub Main()
    On Error Resume Next
    Port.Delete {RECORDED_PORT_NUMBER}
{monitor_deletes}
    On Error GoTo 0
    Pick.ClearAllPicks
End Sub
'''


def build_recorded_port_vba() -> str:
    """Recreate the connector-edge pick and Port 1 exactly as recorded."""

    return '''\
Sub Main()
    Pick.ClearAllPicks
    Pick.PickEdgeFromId "Connector:ConFace", "30", "22"

    With Port
        .Reset
        .PortNumber "1"
        .Label ""
        .Folder ""
        .NumberOfModes "1"
        .AdjustPolarization "False"
        .PolarizationAngle "0.0"
        .ReferencePlaneDistance "0"
        .TextSize "50"
        .TextMaxLimit "0"
        .Coordinates "Picks"
        .Orientation "positive"
        .PortOnBound "False"
        .ClipPickedPortToBound "False"
        .Xrange "-1.76825", "1.76825"
        .Yrange "-8.89", "-8.89"
        .Zrange "-1.38725", "2.14925"
        .XrangeAdd "0.0", "0.0"
        .YrangeAdd "0.0", "0.0"
        .ZrangeAdd "0.0", "0.0"
        .SingleEnded "False"
        .WaveguideMonitor "False"
        .Create
    End With
End Sub
'''


def build_recorded_frequency_range_vba() -> str:
    return '''\
Sub Main()
    Solver.FrequencyRange "2", "7"
End Sub
'''


def build_recorded_mesh_vba() -> str:
    """Return the recorded CST 2025 Hexahedral FIT mesh configuration."""

    return '''\
Sub Main()
    With Mesh
        .MeshType "PBA"
        .SetCreator "High Frequency"
    End With
    With MeshSettings
        .SetMeshType "Hex"
        .Set "Version", 1%
        .Set "StepsPerWaveNear", "20"
        .Set "StepsPerWaveFar", "15"
        .Set "WavelengthRefinementSameAsNear", "0"
        .Set "StepsPerBoxNear", "20"
        .Set "StepsPerBoxFar", "10"
        .Set "MaxStepNear", "0"
        .Set "MaxStepFar", "0"
        .Set "ModelBoxDescrNear", "maxedge"
        .Set "ModelBoxDescrFar", "maxedge"
        .Set "UseMaxStepAbsolute", "0"
        .Set "GeometryRefinementSameAsNear", "0"
        .Set "UseRatioLimitGeometry", "1"
        .Set "RatioLimitGeometry", "20"
        .Set "MinStepGeometryX", "0"
        .Set "MinStepGeometryY", "0"
        .Set "MinStepGeometryZ", "0"
        .Set "UseSameMinStepGeometryXYZ", "1"
    End With
    With MeshSettings
        .Set "PlaneMergeVersion", "2"
    End With
    With MeshSettings
        .SetMeshType "Hex"
        .Set "FaceRefinementType", "NONE"
        .Set "FaceRefinementRatio", "2"
        .Set "FaceRefinementStep", "0"
        .Set "FaceRefinementNSteps", "2"
        .Set "EllipseRefinementType", "NONE"
        .Set "EllipseRefinementRatio", "2"
        .Set "EllipseRefinementStep", "0"
        .Set "EllipseRefinementNSteps", "2"
        .Set "FaceRefinementBufferLines", "3"
        .Set "EdgeRefinementType", "RATIO"
        .Set "EdgeRefinementRatio", "6"
        .Set "EdgeRefinementStep", "0"
        .Set "EdgeRefinementBufferLines", "3"
        .Set "RefineEdgeMaterialGlobal", "0"
        .Set "RefineAxialEdgeGlobal", "0"
        .Set "BufferLinesNear", "3"
        .Set "UseDielectrics", "1"
        .Set "EquilibrateOn", "1"
        .Set "Equilibrate", "1.5"
        .Set "IgnoreThinPanelMaterial", "0"
    End With
    With MeshSettings
        .SetMeshType "Hex"
        .Set "SnapToAxialEdges", "0"
        .Set "SnapToPlanes", "1"
        .Set "SnapToSpheres", "1"
        .Set "SnapToEllipses", "0"
        .Set "SnapToCylinders", "1"
        .Set "SnapToCylinderCenters", "1"
        .Set "SnapToEllipseCenters", "1"
        .Set "SnapToTori", "1"
        .Set "SnapXYZ", "1", "1", "1"
    End With
    With Mesh
        .ConnectivityCheck "True"
        .UsePecEdgeModel "True"
        .PointAccEnhancement "0"
        .TSTVersion "0"
        .PBAVersion "2024121625"
        .SetCADProcessingMethod "MultiThread22", "-1"
        .SetGPUForMatrixCalculationDisabled "False"
    End With
End Sub
'''


def build_recorded_solver_acceleration_vba() -> str:
    return '''\
Sub Main()
    With Solver
        .UseParallelization "True"
        .MaximumNumberOfThreads "1024"
        .MaximumNumberOfCPUDevices "2"
        .RemoteCalculation "False"
        .UseDistributedComputing "False"
        .MaxNumberOfDistributedComputingPorts "64"
        .DistributeMatrixCalculation "True"
        .MPIParallelization "False"
        .AutomaticMPI "False"
        .ConsiderOnly0D1DResultsForMPI "False"
        .HardwareAcceleration "True"
        .MaximumNumberOfGPUs "1"
    End With
    UseDistributedComputingForParameters "False"
    MaxNumberOfDistributedComputingParameters "2"
    UseDistributedComputingMemorySetting "False"
    MinDistributedComputingMemoryLimit "0"
    UseDistributedComputingSharedDirectory "False"
    OnlyConsider0D1DResultsForDC "False"
End Sub
'''


def build_recorded_solver_parameters_vba() -> str:
    return '''\
Sub Main()
    Mesh.SetCreator "High Frequency"

    With Solver
        .Method "Hexahedral"
        .CalculationType "TD-S"
        .StimulationPort "All"
        .StimulationMode "All"
        .SteadyStateLimit "-35"
        .MeshAdaption "False"
        .AutoNormImpedance "False"
        .NormingImpedance "50"
        .CalculateModesOnly "False"
        .SParaSymmetry "False"
        .StoreTDResultsInCache "False"
        .RunDiscretizerOnly "False"
        .FullDeembedding "False"
        .SuperimposePLWExcitation "False"
        .UseSensitivityAnalysis "False"
    End With
End Sub
'''


def build_recorded_farfield_monitors_vba() -> str:
    return '''\
Sub Main()
    With Monitor
        .Reset
        .Domain "Frequency"
        .FieldType "Farfield"
        .ExportFarfieldSource "True"
        .UseSubvolume "False"
        .Coordinates "Structure"
        .SetSubvolume "-4.76", "4.76", "-8.89", "4.5", "-3.96", "3.96"
        .SetSubvolumeOffset "10", "10", "10", "10", "10", "10"
        .SetSubvolumeInflateWithOffset "False"
        .SetSubvolumeOffsetType "FractionOfWavelength"
        .EnableNearfieldCalculation "True"
        .CreateUsingLinearStep "2", "7", "0.1"
    End With
End Sub
'''


def clear_results_on_project(
    project: Any,
    *,
    timeout: float | None = DEFAULT_COMMAND_TIMEOUT,
) -> None:
    """Clear stale solver results before changing geometry or setup."""

    execute_project_vba(
        project,
        "clear previous simulation results",
        build_clear_results_vba(),
        timeout=timeout,
    )


def build_export_s11_vba(output_path: Path) -> str:
    tree_path = _vba_string(S11_TREE_PATH)
    output = _vba_string(output_path.resolve())
    return f'''\
Sub Main()
    If Not SelectTreeItem("{tree_path}") Then
        Err.Raise vbObjectError + 1300, , "S11 result tree item does not exist"
    End If
    With ASCIIExport
        .Reset
        .FileName "{output}"
        .Execute
    End With
End Sub
'''


def plot_s11_ascii(input_path: Path, output_path: Path) -> S11CurveSummary:
    """Plot CST's two-column ASCII S11 export and return basic curve metrics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequencies: list[float] = []
    values_db: list[float] = []
    for line in input_path.read_text(encoding="utf-8-sig").splitlines():
        columns = line.split()
        if len(columns) < 2:
            continue
        try:
            frequency = float(columns[0])
            value_db = float(columns[1])
        except ValueError:
            continue
        frequencies.append(frequency)
        values_db.append(value_db)

    if len(frequencies) < 2:
        raise ValueError(f"S11 export contains fewer than two numeric rows: {input_path}")

    minimum_index = min(range(len(values_db)), key=values_db.__getitem__)
    summary = S11CurveSummary(
        point_count=len(frequencies),
        frequency_min_ghz=min(frequencies),
        frequency_max_ghz=max(frequencies),
        minimum_db=values_db[minimum_index],
        minimum_frequency_ghz=frequencies[minimum_index],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(frequencies, values_db, color="#0068b5", linewidth=1.4)
    axis.axhline(-10.0, color="#b91c1c", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Frequency (GHz)")
    axis.set_ylabel("S11 (dB)")
    axis.set_title("MSA-BP baseline S11")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return summary


def inspect_project(project: Any, timeout: float | None) -> ProjectPrerequisites:
    """Read and validate the current port, monitor, and solver setup."""

    model3d = project.model3d
    if model3d is None:
        raise RuntimeError("CST project does not expose a 3D modeler")

    tree_items = tuple(str(item) for item in model3d.get_tree_items(timeout=timeout))
    ports = tuple(
        sorted(
            item
            for item in tree_items
            if item.startswith("Ports\\") and item.count("\\") == 1
        )
    )
    if not ports:
        raise RuntimeError("CST project does not contain a reusable excitation port")

    farfield_monitors = tuple(
        sorted(
            item
            for item in tree_items
            if item.startswith("Field Monitors\\farfield")
        )
    )
    if not farfield_monitors:
        raise RuntimeError("CST project does not contain a reusable farfield monitor")

    return ProjectPrerequisites(
        solver_name=str(model3d.get_active_solver_name(timeout=timeout)),
        solver_running=bool(model3d.is_solver_running(timeout=timeout)),
        ports=ports,
        farfield_monitors=farfield_monitors,
    )


def inspect_recorded_simulation_setup(
    project: Any,
    timeout: float | None = DEFAULT_COMMAND_TIMEOUT,
) -> ProjectPrerequisites:
    """Require Port 1 and every monitor from the recorded 2--7 GHz sweep."""

    prerequisites = inspect_project(project, timeout)
    expected_port = RECORDED_PORT_TREE_ITEM
    if expected_port not in prerequisites.ports:
        raise RuntimeError(
            "recorded excitation port is missing: "
            f"expected={expected_port}, actual={prerequisites.ports}"
        )

    actual_monitors = set(prerequisites.farfield_monitors)
    missing_monitors = [
        item
        for item in RECORDED_FARFIELD_MONITOR_TREE_ITEMS
        if item not in actual_monitors
    ]
    if missing_monitors:
        preview = ", ".join(missing_monitors[:3])
        if len(missing_monitors) > 3:
            preview += f", ... ({len(missing_monitors)} missing)"
        raise RuntimeError(f"recorded farfield monitor sweep is incomplete: {preview}")
    return prerequisites


def restore_recorded_simulation_setup(
    project: Any,
    *,
    timeout: float | None = DEFAULT_COMMAND_TIMEOUT,
) -> ProjectPrerequisites:
    """Recreate the solver setup recorded from the repaired CST project."""

    commands = (
        (
            "delete recorded port and farfield monitors",
            build_delete_recorded_setup_vba(),
        ),
        ("recreate connector pick and Port 1", build_recorded_port_vba()),
        ("restore solver frequency range", build_recorded_frequency_range_vba()),
        ("restore Hexahedral FIT mesh", build_recorded_mesh_vba()),
        (
            "restore time-domain solver acceleration",
            build_recorded_solver_acceleration_vba(),
        ),
        (
            "restore time-domain solver parameters",
            build_recorded_solver_parameters_vba(),
        ),
        (
            "recreate 2 to 7 GHz farfield monitors",
            build_recorded_farfield_monitors_vba(),
        ),
    )
    for label, vba in commands:
        execute_project_vba(project, label, vba, timeout=timeout)

    prerequisites = inspect_recorded_simulation_setup(project, timeout)
    print(
        "[CST] recorded simulation setup restored: "
        f"ports={len(prerequisites.ports)}, "
        f"farfield_monitors={len(prerequisites.farfield_monitors)}",
        flush=True,
    )
    return prerequisites


def export_s11_from_project(
    project: Any,
    output_path: Path,
    *,
    overwrite: bool = False,
    command_timeout: float | None = DEFAULT_COMMAND_TIMEOUT,
    save_project: bool = False,
    export_label: str = "export S11",
) -> Path:
    """Export the current S11 result from an already connected project."""

    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"S11 output already exists: {output_path}; enable overwrite to replace it"
        )

    tree_items = set(
        str(item) for item in project.model3d.get_tree_items(timeout=command_timeout)
    )
    if S11_TREE_PATH not in tree_items:
        raise RuntimeError(f"solver completed without result: {S11_TREE_PATH}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    execute_project_vba(
        project,
        export_label,
        build_export_s11_vba(output_path),
        timeout=command_timeout,
    )
    if save_project:
        execute_save_project(project, timeout=command_timeout)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"CST reported export success but output is missing: {output_path}")
    print(f"S11 exported: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def solve_and_export_s11_on_project(
    project: Any,
    output_path: Path,
    *,
    overwrite: bool = False,
    command_timeout: float | None = DEFAULT_COMMAND_TIMEOUT,
    save_project: bool = False,
    clear_results: bool = True,
    stage_callback: Callable[[str], None] | None = None,
) -> Path:
    """Optionally clear, solve synchronously, and export S11."""

    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"S11 output already exists: {output_path}; enable overwrite to replace it"
        )

    if clear_results:
        if stage_callback is not None:
            stage_callback("clearing_results")
        clear_results_on_project(project, timeout=command_timeout)
    if stage_callback is not None:
        stage_callback("solving")
    print("[CST] solver started", flush=True)
    project.model3d.run_solver(timeout=None)
    print("[CST] solver completed", flush=True)
    if stage_callback is not None:
        stage_callback("exporting_s11")
    return export_s11_from_project(
        project,
        output_path,
        overwrite=overwrite,
        command_timeout=command_timeout,
        save_project=save_project,
        export_label="export case S11",
    )


def run_and_export_s11(
    project_path: Path = DEFAULT_PROJECT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    rebuild_geometry: bool = True,
    overwrite: bool = False,
    resume_running: bool = False,
    poll_interval: float = 10.0,
    command_timeout: float | None = DEFAULT_COMMAND_TIMEOUT,
    check_only: bool = False,
) -> Path | None:
    """Run one synchronous solve and export ``1D Results\\S-Parameters\\S1,1``."""

    project_path = project_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not project_path.is_file():
        raise FileNotFoundError(f"CST project does not exist: {project_path}")
    if poll_interval <= 0.0:
        raise ValueError("poll interval must be positive")

    project = open_cst_project(str(project_path))
    print(f"CST project: {project_path}")
    if check_only:
        prerequisites = inspect_recorded_simulation_setup(project, command_timeout)
        print(f"active solver: {prerequisites.solver_name}")
        print(f"solver running: {prerequisites.solver_running}")
        print(f"ports: {prerequisites.ports}")
        print(f"farfield monitors: {len(prerequisites.farfield_monitors)}")
        print("check-only: no geometry, results, solver, or files were changed")
        return None

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"S11 output already exists: {output_path}; use --overwrite to replace it"
        )

    if resume_running:
        prerequisites = inspect_recorded_simulation_setup(project, command_timeout)
        print(f"active solver: {prerequisites.solver_name}")
        print(f"ports: {prerequisites.ports}")
        print(f"farfield monitors: {len(prerequisites.farfield_monitors)}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print("[CST] attaching to the existing solver run", flush=True)
        while project.model3d.is_solver_running(timeout=command_timeout):
            print("[CST] solver is still running", flush=True)
            time.sleep(poll_interval)
        print("[CST] existing solver run completed", flush=True)
        return export_s11_from_project(
            project,
            output_path,
            overwrite=overwrite,
            command_timeout=command_timeout,
            save_project=True,
            export_label="export baseline S11",
        )

    clear_results_on_project(project, timeout=command_timeout)
    if rebuild_geometry:
        print("[CST] rebuilding script-managed geometry")
        build_msabp_in_cst(
            project_path=project_path,
            timeout=command_timeout,
            project=project,
            save_project=False,
        )
    prerequisites = restore_recorded_simulation_setup(
        project,
        timeout=command_timeout,
    )
    print(f"active solver: {prerequisites.solver_name}")
    print(f"ports: {prerequisites.ports}")
    print(f"farfield monitors: {len(prerequisites.farfield_monitors)}")
    return solve_and_export_s11_on_project(
        project,
        output_path,
        overwrite=overwrite,
        command_timeout=command_timeout,
        save_project=True,
        clear_results=False,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild MSA-BP, run CST once, and export S11."
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument(
        "--resume-running",
        action="store_true",
        help="Wait for an already-running solver and export without clearing or restarting.",
    )
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="Plot an existing --output ASCII file without connecting to CST.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "figures" / "S11_baseline.png",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT,
        help="Timeout for short CST commands; the solver itself has no timeout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plot_existing:
        summary = plot_s11_ascii(args.output.resolve(), args.plot_output.resolve())
        print(f"S11 plot: {args.plot_output.resolve()}")
        print(
            f"points={summary.point_count}, "
            f"range={summary.frequency_min_ghz:g}..{summary.frequency_max_ghz:g} GHz, "
            f"minimum={summary.minimum_db:.6g} dB "
            f"at {summary.minimum_frequency_ghz:g} GHz"
        )
        return 0

    run_and_export_s11(
        project_path=args.project,
        output_path=args.output,
        rebuild_geometry=not args.skip_rebuild and not args.resume_running,
        overwrite=args.overwrite,
        resume_running=args.resume_running,
        poll_interval=args.poll_interval,
        command_timeout=args.command_timeout,
        check_only=args.check_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
