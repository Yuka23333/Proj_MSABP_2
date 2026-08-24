"""Rebuild the managed MSA-BP geometry and run the CST solver.

The maintained ``msa-bp.cst`` template owns the excitation port, farfield
monitors, mesh, and solver settings. Geometry rebuilds preserve those objects;
the automation validates the template setup before solving but never recreates
or silently repairs it.
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
RAD_EFF_TREE_PATH = r"1D Results\Efficiencies\Rad. Efficiency [1]"
TOT_EFF_TREE_PATH = r"1D Results\Efficiencies\Tot. Efficiency [1]"
RAD_EFF_FILENAME = "Rad_Eff.csv"
TOT_EFF_FILENAME = "Tot_Eff.csv"
DEFAULT_COMMAND_TIMEOUT = 60.0
RECORDED_PORT_NUMBER = 1
RECORDED_PORT_TREE_ITEM = rf"Ports\port{RECORDED_PORT_NUMBER}"
EXPECTED_SOLVER_NAME = "HF Time Domain"
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


def build_export_1d_results_vba(
    exports: Sequence[tuple[str, Path]],
) -> str:
    """Build one VBA command that exports multiple CST 1D result curves."""

    if not exports:
        raise ValueError("at least one 1D result export is required")
    lines = ["Sub Main()"]
    for index, (tree_path, output_path) in enumerate(exports):
        tree_path_vba = _vba_string(tree_path)
        output_path_vba = _vba_string(output_path.resolve())
        error_message = _vba_string(f"result tree item does not exist: {tree_path}")
        lines.extend(
            (
                f'    If Not SelectTreeItem("{tree_path_vba}") Then',
                f'        Err.Raise vbObjectError + {1300 + index}, , '
                f'"{error_message}"',
                "    End If",
                "    With ASCIIExport",
                "        .Reset",
                f'        .FileName "{output_path_vba}"',
                "        .Execute",
                "    End With",
            )
        )
    lines.append("End Sub")
    return "\n".join(lines)


def build_export_s11_vba(output_path: Path) -> str:
    """Build the legacy-compatible single S11 export command."""

    return build_export_1d_results_vba(((S11_TREE_PATH, output_path),))


def one_d_result_output_paths(s11_output_path: Path) -> dict[str, Path]:
    """Derive all per-case 1D output paths from the requested S11 path."""

    s11_path = s11_output_path.expanduser().resolve()
    return {
        "s11": s11_path,
        "rad_eff": s11_path.with_name(RAD_EFF_FILENAME),
        "tot_eff": s11_path.with_name(TOT_EFF_FILENAME),
    }


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
    if prerequisites.solver_name != EXPECTED_SOLVER_NAME:
        raise RuntimeError(
            "unexpected active CST solver: "
            f"expected={EXPECTED_SOLVER_NAME}, actual={prerequisites.solver_name}"
        )
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


def export_1d_results_from_project(
    project: Any,
    s11_output_path: Path,
    *,
    overwrite: bool = False,
    command_timeout: float | None = DEFAULT_COMMAND_TIMEOUT,
    save_project: bool = False,
    export_label: str = "export 1D results",
) -> dict[str, Path]:
    """Export S11, radiation efficiency, and total efficiency curves."""

    output_paths = one_d_result_output_paths(s11_output_path)
    collisions = [path for path in output_paths.values() if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise FileExistsError(
            f"1D result outputs already exist: {names}; enable overwrite to replace them"
        )

    tree_items = set(
        str(item) for item in project.model3d.get_tree_items(timeout=command_timeout)
    )
    result_tree_paths = {
        "s11": S11_TREE_PATH,
        "rad_eff": RAD_EFF_TREE_PATH,
        "tot_eff": TOT_EFF_TREE_PATH,
    }
    missing_results = [
        tree_path
        for tree_path in result_tree_paths.values()
        if tree_path not in tree_items
    ]
    if missing_results:
        raise RuntimeError(
            "solver completed without 1D results: " + ", ".join(missing_results)
        )

    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    execute_project_vba(
        project,
        export_label,
        build_export_1d_results_vba(
            tuple(
                (result_tree_paths[name], output_paths[name])
                for name in ("s11", "rad_eff", "tot_eff")
            )
        ),
        timeout=command_timeout,
    )
    if save_project:
        execute_save_project(project, timeout=command_timeout)

    for name, output_path in output_paths.items():
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"CST reported {name} export success but output is missing: "
                f"{output_path}"
            )
        print(
            f"{name} exported: {output_path} ({output_path.stat().st_size} bytes)"
        )
    return output_paths


def export_s11_from_project(
    project: Any,
    output_path: Path,
    *,
    overwrite: bool = False,
    command_timeout: float | None = DEFAULT_COMMAND_TIMEOUT,
    save_project: bool = False,
    export_label: str = "export 1D results",
) -> Path:
    """Export all standard 1D curves and return the S11 path for compatibility."""

    return export_1d_results_from_project(
        project,
        output_path,
        overwrite=overwrite,
        command_timeout=command_timeout,
        save_project=save_project,
        export_label=export_label,
    )["s11"]


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
    """Optionally clear, solve synchronously, and export standard 1D curves."""

    output_path = output_path.expanduser().resolve()
    collisions = [
        path
        for path in one_d_result_output_paths(output_path).values()
        if path.exists()
    ]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise FileExistsError(
            f"1D result outputs already exist: {names}; enable overwrite to replace them"
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
        stage_callback("exporting_1d_results")
    return export_s11_from_project(
        project,
        output_path,
        overwrite=overwrite,
        command_timeout=command_timeout,
        save_project=save_project,
        export_label="export case 1D results",
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

    collisions = [
        path
        for path in one_d_result_output_paths(output_path).values()
        if path.exists()
    ]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise FileExistsError(
            f"1D result outputs already exist: {names}; use --overwrite to replace them"
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
            export_label="export baseline 1D results",
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
    prerequisites = inspect_recorded_simulation_setup(project, command_timeout)
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
        description="Rebuild MSA-BP, run CST once, and export standard 1D results."
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
