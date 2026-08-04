"""Rebuild the managed MSA-BP geometry, solve, and export one S11 curve.

The geometry builder deletes only its own ``component1:msabp_*`` objects and
matching source curves.  Existing connectors, ports, boundaries, solver setup,
and field monitors are treated as reusable CST-template objects.
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
    """Read and validate the reusable port, monitor, and solver setup."""

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
    stage_callback: Callable[[str], None] | None = None,
) -> Path:
    """Clear, solve synchronously, and export S11 on an existing CST project."""

    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"S11 output already exists: {output_path}; enable overwrite to replace it"
        )

    if stage_callback is not None:
        stage_callback("clearing_results")
    execute_project_vba(
        project,
        "clear previous simulation results",
        build_clear_results_vba(),
        timeout=command_timeout,
    )
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
    prerequisites = inspect_project(project, command_timeout)
    print(f"CST project: {project_path}")
    print(f"active solver: {prerequisites.solver_name}")
    print(f"solver running: {prerequisites.solver_running}")
    print(f"ports: {prerequisites.ports}")
    print(f"farfield monitors: {len(prerequisites.farfield_monitors)}")
    if check_only:
        print("check-only: no geometry, results, solver, or files were changed")
        return None

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"S11 output already exists: {output_path}; use --overwrite to replace it"
        )

    if rebuild_geometry and not resume_running:
        print("[CST] rebuilding script-managed geometry")
        build_msabp_in_cst(
            project_path=project_path,
            timeout=command_timeout,
        )
        project = open_cst_project(str(project_path))
        prerequisites = inspect_project(project, command_timeout)
        print(
            "[CST] reusable template objects retained: "
            f"ports={len(prerequisites.ports)}, "
            f"farfield_monitors={len(prerequisites.farfield_monitors)}"
        )

    if resume_running:
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
    else:
        return solve_and_export_s11_on_project(
            project,
            output_path,
            overwrite=overwrite,
            command_timeout=command_timeout,
            save_project=True,
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
