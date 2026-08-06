"""Build one reproducible slot branch that leaves the Patch but not substrate."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from shapely.geometry import LineString, MultiLineString, Point, Polygon


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import cst_build_msabp_geometry  # noqa: E402
from scripts.geometry import antenna_outline  # noqa: E402


DEFAULT_TEST_PROJECT = (
    REPOSITORY_ROOT / "simulations" / "models" / "MSA-BP_outgrown-branch-test.cst"
)
DEFAULT_RANDOM_SEED = 20260803
DEFAULT_BRANCH_WIDTH_MM = 1.0

BRANCH_PARAMETER_FIELDS = {
    "reserved_up_1": (
        "inner_slot_order2_reserved_up_enabled",
        "inner_slot_order2_reserved_up_length_mm",
        "inner_slot_order2_reserved_up_width_mm",
    ),
    "reserved_down_1": (
        "inner_slot_order2_reserved_down1_enabled",
        "inner_slot_order2_reserved_down1_length_mm",
        "inner_slot_order2_reserved_down1_width_mm",
    ),
    "reserved_down_2": (
        "inner_slot_order2_reserved_down2_enabled",
        "inner_slot_order2_reserved_down2_length_mm",
        "inner_slot_order2_reserved_down2_width_mm",
    ),
}


@dataclass(frozen=True)
class OutgrownBranchTrial:
    name: str
    anchor: antenna_outline.Point2D
    growth_direction: antenna_outline.Point2D
    patch_exit_length_mm: float
    substrate_limit_length_mm: float
    trial_length_mm: float
    width_mm: float
    slot_outside_patch_area_mm2: float
    final_conductor_geometry_type: str
    parameters: antenna_outline.AntennaOutlineParameters


def _line_component_from_anchor(
    geometry: LineString | MultiLineString,
    anchor: antenna_outline.Point2D,
) -> LineString:
    components = (geometry,) if isinstance(geometry, LineString) else geometry.geoms
    anchor_point = Point(anchor)
    matching = [component for component in components if component.distance(anchor_point) < 1e-9]
    if len(matching) != 1:
        raise ValueError("could not identify one Patch segment connected to branch anchor")
    return matching[0]


def select_outgrown_branch_trial(
    seed: int = DEFAULT_RANDOM_SEED,
    width_mm: float = DEFAULT_BRANCH_WIDTH_MM,
) -> OutgrownBranchTrial:
    """Randomly select a branch with room to cross Patch before substrate edge."""

    if width_mm <= 0.0:
        raise ValueError("trial branch width must be positive")
    default_parameters = antenna_outline.DEFAULT_ANTENNA_PARAMETERS
    patch = antenna_outline.build_antenna_closed_polygons(default_parameters)[0]
    reservations = antenna_outline.generate_inner_slot_order2_reservations(
        default_parameters
    )
    eligible: list[tuple[antenna_outline.InnerSlotBranchReservation, float, float]] = []
    for reservation in reservations:
        anchor_x, anchor_y = reservation.anchor
        direction_x, direction_y = reservation.growth_direction
        distances_to_edges = []
        if direction_x > 0.0:
            distances_to_edges.append(default_parameters.rectangle_length_mm / 2.0 - anchor_x)
        elif direction_x < 0.0:
            distances_to_edges.append(anchor_x + default_parameters.rectangle_length_mm / 2.0)
        if direction_y > 0.0:
            distances_to_edges.append(default_parameters.rectangle_width_mm - anchor_y)
        elif direction_y < 0.0:
            distances_to_edges.append(anchor_y)
        if len(distances_to_edges) != 1:
            raise ValueError(f"{reservation.name} must grow along one cardinal axis")
        substrate_limit = float(distances_to_edges[0])
        ray_end = (
            anchor_x + direction_x * substrate_limit,
            anchor_y + direction_y * substrate_limit,
        )
        inside = LineString([reservation.anchor, ray_end]).intersection(patch)
        if not isinstance(inside, (LineString, MultiLineString)):
            continue
        patch_exit = float(
            _line_component_from_anchor(inside, reservation.anchor).length
        )
        if patch_exit < substrate_limit - 1e-6:
            eligible.append((reservation, patch_exit, substrate_limit))

    if not eligible:
        raise ValueError("no reserved branch can leave Patch while staying in substrate")
    reservation, patch_exit, substrate_limit = random.Random(seed).choice(eligible)
    trial_length = (patch_exit + substrate_limit) / 2.0
    enabled_field, length_field, width_field = BRANCH_PARAMETER_FIELDS[reservation.name]
    parameters = replace(
        default_parameters,
        **{
            enabled_field: True,
            length_field: trial_length,
            width_field: width_mm,
        },
    )
    trial_patch, trial_slot, trial_guide = antenna_outline.build_antenna_closed_polygons(
        parameters
    )
    substrate = Polygon(antenna_outline.generate_rectangle(parameters))
    if not substrate.covers(trial_slot):
        raise ValueError("selected trial slot unexpectedly leaves the substrate")
    slot_outside_patch_area = float(trial_slot.difference(trial_patch).area)
    if slot_outside_patch_area <= 0.0:
        raise ValueError("selected trial slot did not actually leave the Patch")
    final_conductor = trial_patch.difference(trial_slot).union(trial_guide)
    return OutgrownBranchTrial(
        name=reservation.name,
        anchor=reservation.anchor,
        growth_direction=reservation.growth_direction,
        patch_exit_length_mm=patch_exit,
        substrate_limit_length_mm=substrate_limit,
        trial_length_mm=trial_length,
        width_mm=width_mm,
        slot_outside_patch_area_mm2=slot_outside_patch_area,
        final_conductor_geometry_type=final_conductor.geom_type,
        parameters=parameters,
    )


def run_trial(
    project_path: Path = DEFAULT_TEST_PROJECT,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    width_mm: float = DEFAULT_BRANCH_WIDTH_MM,
    timeout: float | None = 60.0,
    dry_run: bool = False,
) -> OutgrownBranchTrial:
    trial = select_outgrown_branch_trial(seed=seed, width_mm=width_mm)
    print(f"selected branch: {trial.name}")
    print(f"anchor: {trial.anchor}, direction: {trial.growth_direction}")
    print(f"Patch exit length: {trial.patch_exit_length_mm:g} mm")
    print(f"substrate limit length: {trial.substrate_limit_length_mm:g} mm")
    print(f"trial length x width: {trial.trial_length_mm:g} x {trial.width_mm:g} mm")
    print(f"slot outside Patch: {trial.slot_outside_patch_area_mm2:g} mm^2")
    print(f"final Shapely type: {trial.final_conductor_geometry_type}")
    cst_build_msabp_geometry.build_msabp_in_cst(
        project_path=project_path,
        parameters=trial.parameters,
        allow_disconnected_conductor=True,
        timeout=timeout,
        dry_run=dry_run,
    )
    return trial


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test one reserved branch beyond Patch but inside substrate."
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_TEST_PROJECT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--width", type=float, default=DEFAULT_BRANCH_WIDTH_MM)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_trial(
        project_path=args.project,
        seed=args.seed,
        width_mm=args.width,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
