"""Build one propagation case in a host-local CST session and keep it alive.

Run this script from an interactive desktop on each simulation host.  Geometry
created through ``execute_vba_code`` is intentionally session-local: saving and
reopening a standalone CST file can discard it because it is absent from the
History Tree.  Therefore this script does not transfer or persist generated
geometry.  It keeps the owning Python/CST session open while the user starts
the solver manually in the visible CST window.
"""

from __future__ import annotations

import argparse
import csv
import socket
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from msabp_opt.simulation.distributed.maid import OwnedCstSession  # noqa: E402
from msabp_opt.simulation.distributed.propagation_case_runner import (  # noqa: E402
    SECOND_COMPONENT,
    SOURCE_COMPONENT,
    _transform_second_antenna_vba,
    inspect_propagation_infrastructure,
)
from scripts.automation import antenna_sampler  # noqa: E402
from scripts.automation import cst_build_msabp_geometry  # noqa: E402
from scripts.automation import cst_run_and_export_s11  # noqa: E402
from scripts.automation.cst_generate_polygen import execute_project_vba  # noqa: E402


DEBUG_ID = "manual-propagation-export-debug-rank01-002"
PROJECT_PATH = (
    REPOSITORY_ROOT
    / "simulations"
    / "runs"
    / DEBUG_ID
    / "model"
    / "msa-bp-propagation.cst"
)
WORKLIST_PATH = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_selected_13.csv"
)
CANDIDATE_RANK = 1
COMMAND_TIMEOUT_SECONDS = 60.0


def load_candidate_row(
    worklist: str | Path,
    candidate_rank: int,
) -> dict[str, str]:
    source = Path(worklist).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        matches = [
            row
            for row in csv.DictReader(stream)
            if int(row["candidate_rank"]) == int(candidate_rank)
        ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one candidate_rank={candidate_rank}, "
            f"found {len(matches)} in {source}"
        )
    return matches[0]


def build_pair_in_open_session(
    project: object,
    project_path: Path,
    row: dict[str, str],
    *,
    timeout: float,
) -> None:
    """Clear results and create both managed antennas without solving/saving."""

    parameters = antenna_sampler.parameters_from_csv_row(row)
    substrate_material = str(
        row.get(
            "substrate_material",
            cst_build_msabp_geometry.DEFAULT_SUBSTRATE_MATERIAL_NAME,
        )
    ).strip()

    before = inspect_propagation_infrastructure(project, timeout)
    print(
        "[host-local] infrastructure before build: "
        f"ports={before['ports']}, "
        f"E-field monitors={len(before['e_field_monitors'])}, "
        f"solver={before['solver_name']}",
        flush=True,
    )
    cst_run_and_export_s11.clear_results_on_project(project, timeout=timeout)
    source_report = cst_build_msabp_geometry.build_msabp_in_cst(
        project_path=project_path,
        component_name=SOURCE_COMPONENT,
        parameters=parameters,
        substrate_material_name=substrate_material,
        timeout=timeout,
        project=project,
        save_project=False,
        define_substrate_material=True,
    )
    second_report = cst_build_msabp_geometry.build_msabp_in_cst(
        project_path=project_path,
        component_name=SECOND_COMPONENT,
        parameters=parameters,
        substrate_material_name=substrate_material,
        timeout=timeout,
        project=project,
        save_project=False,
        define_substrate_material=False,
    )
    if source_report != second_report:
        raise RuntimeError("the two antenna build reports differ")
    execute_project_vba(
        project,
        "mirror and translate second managed antenna",
        _transform_second_antenna_vba(),
        timeout=timeout,
    )
    after = inspect_propagation_infrastructure(project, timeout)
    print(
        "[host-local] infrastructure after build: "
        f"ports={after['ports']}, "
        f"E-field monitors={len(after['e_field_monitors'])}, "
        f"solver={after['solver_name']}",
        flush=True,
    )


def hold_session_open() -> None:
    print("", flush=True)
    print("=" * 72, flush=True)
    print("READY FOR MANUAL SOLVE", flush=True)
    print("Start the solver in this CST window now.", flush=True)
    print("Do NOT close CST or this Python process before export debugging.", flush=True)
    print("Return here and type RELEASE only when the session may be closed.", flush=True)
    print("=" * 72, flush=True)
    while True:
        try:
            command = input("[host-local] command: ").strip().upper()
        except EOFError as exc:
            raise RuntimeError(
                "stdin closed while CST must remain alive; run this script from "
                "an interactive PowerShell/IDE terminal"
            ) from exc
        if command == "RELEASE":
            return
        print("[host-local] session retained; type RELEASE to close it", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a propagation antenna pair locally and hold CST open."
    )
    parser.add_argument("--project", type=Path, default=PROJECT_PATH)
    parser.add_argument("--worklist", type=Path, default=WORKLIST_PATH)
    parser.add_argument("--candidate-rank", type=int, default=CANDIDATE_RANK)
    parser.add_argument("--timeout", type=float, default=COMMAND_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_path = args.project.expanduser().resolve()
    if not project_path.is_file():
        raise FileNotFoundError(f"debug CST project is missing: {project_path}")
    row = load_candidate_row(args.worklist, args.candidate_rank)
    print(f"[host-local] host: {socket.gethostname()}", flush=True)
    print(f"[host-local] project: {project_path}", flush=True)
    print(
        f"[host-local] candidate: rank #{args.candidate_rank}, "
        f"sample={row['sample_id']}",
        flush=True,
    )

    session = OwnedCstSession(project_path)
    try:
        project = session.open()
        build_pair_in_open_session(
            project,
            project_path,
            row,
            timeout=args.timeout,
        )
        hold_session_open()
    finally:
        print("[host-local] releasing owned CST session", flush=True)
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
