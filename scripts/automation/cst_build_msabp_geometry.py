"""Build the current MSA-BP antenna in a CST project.

The substrate, conductor, and reflector source polygons are kept in CST's
Curves tree.  The conductor and reflector tool solids are combined as::

    final conductor = Patch - symmetric slot + symmetric guide
    reflector = substrate-sized rectangle - connector clearance

Run the live CST operation with the ``cstpy`` Conda environment.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from shapely.geometry import MultiPolygon, Polygon


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.geometry import antenna_closed_curves  # noqa: E402
from scripts.geometry import antenna_outline  # noqa: E402
from scripts.geometry import antenna_polygon_export  # noqa: E402
from scripts.geometry import shapely_antenna_model  # noqa: E402

try:
    from .cst_generate_polygen import (
        build_extrude_curve_vba,
        build_polygon_vba,
        execute_project_vba,
        execute_save_project,
        open_cst_project,
    )
except ImportError:
    from cst_generate_polygen import (  # type: ignore[no-redef]
        build_extrude_curve_vba,
        build_polygon_vba,
        execute_project_vba,
        execute_save_project,
        open_cst_project,
    )


DEFAULT_PROJECT_PATH = REPOSITORY_ROOT / "simulations" / "models" / "MSA-BP.cst"
DEFAULT_POLYGON_JSON_PATH = antenna_polygon_export.DEFAULT_EXPORT_PATH
DEFAULT_COMPONENT_NAME = "component1"
DEFAULT_COPPER_MATERIAL_NAME = "Copper (annealed)"
DEFAULT_TOOL_MATERIAL_NAME = "Vacuum"
DEFAULT_COPPER_THICKNESS_MM = 0.035
DEFAULT_SUBSTRATE_MATERIAL_NAME = "Rogers AD 350A (lossy)"
DEFAULT_SUBSTRATE_RELATIVE_PERMITTIVITY = 3.5
VACUUM_SUBSTRATE_MATERIAL_NAME = "Vacuum"
VACUUM_SUBSTRATE_RELATIVE_PERMITTIVITY = 1.0
DEFAULT_SUBSTRATE_THICKNESS_MM = -4.7
DEFAULT_REFLECTOR_CONNECTOR_BOARD_THICKNESS_MM = (
    antenna_outline.REFLECTOR_CONNECTOR_BOARD_THICKNESS_FIXED_MM
)
DEFAULT_REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_MM = (
    antenna_outline.REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_FIXED_MM
)
DEFAULT_REFLECTOR_CUTOUT_DEPTH_MM = antenna_outline.REFLECTOR_CUTOUT_DEPTH_FIXED_MM

SUBSTRATE_CURVE_NAME = "msabp_substrate_curve"
SUBSTRATE_POLYGON_NAME = "msabp_substrate_polygon"
SUBSTRATE_SOLID_NAME = "msabp_substrate_solid"

PATCH_CURVE_NAME = "msabp_patch_curve"
PATCH_POLYGON_NAME = "msabp_patch_polygon"
PATCH_SOLID_NAME = "msabp_patch_solid"

SLOT_CURVE_NAME = "msabp_symmetric_slot_curve"
SLOT_POLYGON_NAME = "msabp_symmetric_slot_polygon"
SLOT_SOLID_NAME = "msabp_symmetric_slot_solid"

GUIDE_CURVE_NAME = "msabp_symmetric_guide_curve"
GUIDE_POLYGON_NAME = "msabp_symmetric_guide_polygon"
GUIDE_SOLID_NAME = "msabp_symmetric_guide_solid"

REFLECTOR_CURVE_NAME = "msabp_reflector_curve"
REFLECTOR_POLYGON_NAME = "msabp_reflector_polygon"
REFLECTOR_SOLID_NAME = "msabp_reflector_solid"

REFLECTOR_CUTOUT_CURVE_NAME = "msabp_reflector_connector_clearance_curve"
REFLECTOR_CUTOUT_POLYGON_NAME = "msabp_reflector_connector_clearance_polygon"
REFLECTOR_CUTOUT_SOLID_NAME = "msabp_reflector_connector_clearance_solid"


Point2D = tuple[float, float]


@dataclass(frozen=True)
class CstPolygonSpec:
    """Names, material, and ordered points for one CST polygon extrusion."""

    label: str
    curve_name: str
    polygon_name: str
    solid_name: str
    material_name: str
    thickness_mm: float
    points: list[Point2D]


@dataclass(frozen=True)
class GeometryBuildReport:
    """Expected planar and volumetric properties used for live verification."""

    point_counts: tuple[int, ...]
    substrate_material_name: str
    substrate_relative_permittivity: float
    substrate_area_mm2: float
    substrate_volume_mm3: float
    patch_area_mm2: float
    slot_area_mm2: float
    guide_area_mm2: float
    final_conductor_area_mm2: float
    final_conductor_volume_mm3: float
    final_conductor_component_count: int
    coordinate_quantum_mm: float
    reflector_cutout_width_mm: float
    reflector_cutout_depth_mm: float
    reflector_area_mm2: float
    reflector_volume_mm3: float
    reflector_z_min_mm: float
    reflector_z_max_mm: float


def _solid_ref(component_name: str, solid_name: str) -> str:
    return f"{component_name}:{solid_name}"


def _curve_item_ref(curve_name: str, polygon_name: str) -> str:
    return f"{curve_name}:{polygon_name}"


def apply_substrate_material(
    specs: Sequence[CstPolygonSpec],
    report: GeometryBuildReport,
    substrate_material_name: str = DEFAULT_SUBSTRATE_MATERIAL_NAME,
) -> tuple[tuple[CstPolygonSpec, ...], GeometryBuildReport]:
    """Apply one supported substrate material without changing geometry.

    ``Vacuum`` is a built-in CST material and must not be deleted or recreated.
    The Rogers baseline remains the default for every row without an explicit
    override.
    """

    requested = str(substrate_material_name).strip()
    supported = {
        DEFAULT_SUBSTRATE_MATERIAL_NAME.casefold(): (
            DEFAULT_SUBSTRATE_MATERIAL_NAME,
            DEFAULT_SUBSTRATE_RELATIVE_PERMITTIVITY,
        ),
        VACUUM_SUBSTRATE_MATERIAL_NAME.casefold(): (
            VACUUM_SUBSTRATE_MATERIAL_NAME,
            VACUUM_SUBSTRATE_RELATIVE_PERMITTIVITY,
        ),
    }
    try:
        canonical_name, relative_permittivity = supported[requested.casefold()]
    except KeyError as exc:
        names = ", ".join(item[0] for item in supported.values())
        raise ValueError(
            f"unsupported substrate material {requested!r}; expected one of: {names}"
        ) from exc

    substrate_matches = [
        spec for spec in specs if spec.solid_name == SUBSTRATE_SOLID_NAME
    ]
    if len(substrate_matches) != 1:
        raise ValueError(
            "CST polygon specs must contain exactly one managed substrate solid"
        )
    updated_specs = tuple(
        replace(spec, material_name=canonical_name)
        if spec.solid_name == SUBSTRATE_SOLID_NAME
        else spec
        for spec in specs
    )
    updated_report = replace(
        report,
        substrate_material_name=canonical_name,
        substrate_relative_permittivity=relative_permittivity,
    )
    return updated_specs, updated_report


def _polygon_signed_area(points: Sequence[Point2D]) -> float:
    if len(points) < 3:
        return 0.0
    following = (*points[1:], points[0])
    return 0.5 * sum(
        x_current * y_next - x_next * y_current
        for (x_current, y_current), (x_next, y_next) in zip(
            points,
            following,
            strict=True,
        )
    )


def _polygon_area(points: Sequence[Point2D]) -> float:
    return abs(_polygon_signed_area(points))


def _build_direct_polygon_specs(
    exported: antenna_polygon_export.AntennaPolygonExport,
    *,
    copper_thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
    substrate_thickness_mm: float = DEFAULT_SUBSTRATE_THICKNESS_MM,
    reflector_connector_board_thickness_mm: float = (
        DEFAULT_REFLECTOR_CONNECTOR_BOARD_THICKNESS_MM
    ),
    reflector_cutout_width_adjustment_mm: float = (
        DEFAULT_REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_MM
    ),
    reflector_cutout_depth_mm: float = DEFAULT_REFLECTOR_CUTOUT_DEPTH_MM,
) -> tuple[tuple[CstPolygonSpec, ...], GeometryBuildReport]:
    """Build CST specs directly from the three exported vertex arrays.

    No Shapely reconstruction or geometric legality check is performed here.
    The exported point order and floating-point values are passed directly to
    the CST Polygon writer, which adds only the closing edge.
    """

    copper_thickness_mm = float(copper_thickness_mm)
    substrate_thickness_mm = float(substrate_thickness_mm)
    reflector_connector_board_thickness_mm = float(
        reflector_connector_board_thickness_mm
    )
    reflector_cutout_width_adjustment_mm = float(reflector_cutout_width_adjustment_mm)
    reflector_cutout_depth_mm = float(reflector_cutout_depth_mm)
    if not math.isfinite(copper_thickness_mm) or copper_thickness_mm <= 0.0:
        raise ValueError("copper thickness must be a finite positive number")
    if not math.isfinite(substrate_thickness_mm) or substrate_thickness_mm >= 0.0:
        raise ValueError("substrate thickness must be finite and negative")
    reflector_values = (
        reflector_connector_board_thickness_mm,
        reflector_cutout_width_adjustment_mm,
        reflector_cutout_depth_mm,
    )
    if not all(math.isfinite(value) for value in reflector_values):
        raise ValueError("reflector clearance dimensions must be finite")

    patch_points = exported.points(antenna_polygon_export.PATCH_KEY)
    slot_points = exported.points(antenna_polygon_export.SLOT_KEY)
    guide_points = exported.points(antenna_polygon_export.FEED_PIN_KEY)
    substrate_points = exported.substrate_rectangle_points()
    reflector_points = list(substrate_points)
    substrate_width_mm, substrate_height_mm = exported.substrate_size_mm
    substrate_min_y = exported.substrate_bounds_mm[1]
    reflector_cutout_half_width_mm = (
        substrate_width_mm / 2.0
        - reflector_connector_board_thickness_mm
        + reflector_cutout_width_adjustment_mm
    )
    reflector_cutout_points = [
        (-reflector_cutout_half_width_mm, substrate_min_y),
        (reflector_cutout_half_width_mm, substrate_min_y),
        (
            reflector_cutout_half_width_mm,
            substrate_min_y + reflector_cutout_depth_mm,
        ),
        (
            -reflector_cutout_half_width_mm,
            substrate_min_y + reflector_cutout_depth_mm,
        ),
        (-reflector_cutout_half_width_mm, substrate_min_y),
    ]
    point_lists = (
        substrate_points,
        patch_points,
        slot_points,
        guide_points,
        reflector_points,
        reflector_cutout_points,
    )
    specs = (
        CstPolygonSpec(
            "substrate",
            SUBSTRATE_CURVE_NAME,
            SUBSTRATE_POLYGON_NAME,
            SUBSTRATE_SOLID_NAME,
            DEFAULT_SUBSTRATE_MATERIAL_NAME,
            substrate_thickness_mm,
            substrate_points,
        ),
        CstPolygonSpec(
            "Patch",
            PATCH_CURVE_NAME,
            PATCH_POLYGON_NAME,
            PATCH_SOLID_NAME,
            DEFAULT_COPPER_MATERIAL_NAME,
            copper_thickness_mm,
            patch_points,
        ),
        CstPolygonSpec(
            "Slot",
            SLOT_CURVE_NAME,
            SLOT_POLYGON_NAME,
            SLOT_SOLID_NAME,
            DEFAULT_TOOL_MATERIAL_NAME,
            copper_thickness_mm,
            slot_points,
        ),
        CstPolygonSpec(
            "CPW_Feed_Pin",
            GUIDE_CURVE_NAME,
            GUIDE_POLYGON_NAME,
            GUIDE_SOLID_NAME,
            DEFAULT_COPPER_MATERIAL_NAME,
            copper_thickness_mm,
            guide_points,
        ),
        CstPolygonSpec(
            "reflector",
            REFLECTOR_CURVE_NAME,
            REFLECTOR_POLYGON_NAME,
            REFLECTOR_SOLID_NAME,
            DEFAULT_COPPER_MATERIAL_NAME,
            -copper_thickness_mm,
            reflector_points,
        ),
        CstPolygonSpec(
            "reflector connector clearance",
            REFLECTOR_CUTOUT_CURVE_NAME,
            REFLECTOR_CUTOUT_POLYGON_NAME,
            REFLECTOR_CUTOUT_SOLID_NAME,
            DEFAULT_TOOL_MATERIAL_NAME,
            -copper_thickness_mm,
            reflector_cutout_points,
        ),
    )

    substrate_area_mm2 = substrate_width_mm * substrate_height_mm
    patch_area_mm2 = _polygon_area(patch_points)
    slot_area_mm2 = _polygon_area(slot_points)
    guide_area_mm2 = _polygon_area(guide_points)
    final_conductor_area_mm2 = patch_area_mm2 - slot_area_mm2 + guide_area_mm2
    cutout_area_mm2 = 2.0 * reflector_cutout_half_width_mm * reflector_cutout_depth_mm
    reflector_area_mm2 = substrate_area_mm2 - cutout_area_mm2
    report = GeometryBuildReport(
        point_counts=tuple(len(points) for points in point_lists),
        substrate_material_name=DEFAULT_SUBSTRATE_MATERIAL_NAME,
        substrate_relative_permittivity=DEFAULT_SUBSTRATE_RELATIVE_PERMITTIVITY,
        substrate_area_mm2=substrate_area_mm2,
        substrate_volume_mm3=substrate_area_mm2 * abs(substrate_thickness_mm),
        patch_area_mm2=patch_area_mm2,
        slot_area_mm2=slot_area_mm2,
        guide_area_mm2=guide_area_mm2,
        final_conductor_area_mm2=final_conductor_area_mm2,
        final_conductor_volume_mm3=(final_conductor_area_mm2 * copper_thickness_mm),
        final_conductor_component_count=1,
        coordinate_quantum_mm=exported.quantize_step_mm,
        reflector_cutout_width_mm=2.0 * reflector_cutout_half_width_mm,
        reflector_cutout_depth_mm=reflector_cutout_depth_mm,
        reflector_area_mm2=reflector_area_mm2,
        reflector_volume_mm3=reflector_area_mm2 * copper_thickness_mm,
        reflector_z_min_mm=substrate_thickness_mm - copper_thickness_mm,
        reflector_z_max_mm=substrate_thickness_mm,
    )
    return specs, report


def build_exported_polygon_specs(
    polygon_json_path: str | Path = DEFAULT_POLYGON_JSON_PATH,
    copper_thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
    substrate_thickness_mm: float = DEFAULT_SUBSTRATE_THICKNESS_MM,
    reflector_connector_board_thickness_mm: float = (
        DEFAULT_REFLECTOR_CONNECTOR_BOARD_THICKNESS_MM
    ),
    reflector_cutout_width_adjustment_mm: float = (
        DEFAULT_REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_MM
    ),
    reflector_cutout_depth_mm: float = DEFAULT_REFLECTOR_CUTOUT_DEPTH_MM,
) -> tuple[tuple[CstPolygonSpec, ...], GeometryBuildReport]:
    """Load the established JSON exchange file and create direct CST specs."""

    exported = antenna_polygon_export.load_antenna_polygon_export(polygon_json_path)
    return _build_direct_polygon_specs(
        exported,
        copper_thickness_mm=copper_thickness_mm,
        substrate_thickness_mm=substrate_thickness_mm,
        reflector_connector_board_thickness_mm=(reflector_connector_board_thickness_mm),
        reflector_cutout_width_adjustment_mm=(reflector_cutout_width_adjustment_mm),
        reflector_cutout_depth_mm=reflector_cutout_depth_mm,
    )


def build_sampled_polygon_specs(
    parameters: shapely_antenna_model.ShapelyAntennaParameters,
    copper_thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
    substrate_thickness_mm: float = DEFAULT_SUBSTRATE_THICKNESS_MM,
    reflector_connector_board_thickness_mm: float = (
        DEFAULT_REFLECTOR_CONNECTOR_BOARD_THICKNESS_MM
    ),
    reflector_cutout_width_adjustment_mm: float = (
        DEFAULT_REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_MM
    ),
    reflector_cutout_depth_mm: float = DEFAULT_REFLECTOR_CUTOUT_DEPTH_MM,
    coordinate_quantum_mm: float = shapely_antenna_model.QUANTIZE_STEP_MM,
) -> tuple[tuple[CstPolygonSpec, ...], GeometryBuildReport]:
    """Build one sampled 23-variable design without a shared JSON file."""

    payload = shapely_antenna_model.polygon_export_payload(
        parameters,
        quantize_step_mm=coordinate_quantum_mm,
    )
    exported = antenna_polygon_export.AntennaPolygonExport(
        source_path=Path("<sampled-in-memory>"),
        quantize_step_mm=float(payload["meta"]["quantize_step"]),
        vertices={
            name: tuple(
                (float(point[0]), float(point[1]))
                for point in payload["vertices"][name]
            )
            for name in antenna_polygon_export.REQUIRED_VERTEX_KEYS
        },
    )
    return _build_direct_polygon_specs(
        exported,
        copper_thickness_mm=copper_thickness_mm,
        substrate_thickness_mm=substrate_thickness_mm,
        reflector_connector_board_thickness_mm=(reflector_connector_board_thickness_mm),
        reflector_cutout_width_adjustment_mm=(reflector_cutout_width_adjustment_mm),
        reflector_cutout_depth_mm=reflector_cutout_depth_mm,
    )


def build_polygon_specs(
    parameters: antenna_outline.AntennaOutlineParameters | None = None,
    copper_thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
    substrate_thickness_mm: float = DEFAULT_SUBSTRATE_THICKNESS_MM,
    reflector_connector_board_thickness_mm: float | None = None,
    reflector_cutout_width_adjustment_mm: float | None = None,
    reflector_cutout_depth_mm: float | None = None,
    coordinate_quantum_mm: float = antenna_outline.COORDINATE_QUANTUM_MM,
    allow_disconnected_conductor: bool = False,
) -> tuple[tuple[CstPolygonSpec, ...], GeometryBuildReport]:
    """Build and validate all substrate, conductor, and reflector polygons."""

    if parameters is None:
        parameters = antenna_outline.DEFAULT_ANTENNA_PARAMETERS

    if reflector_connector_board_thickness_mm is None:
        reflector_connector_board_thickness_mm = (
            parameters.reflector_connector_board_thickness_mm
        )
    if reflector_cutout_width_adjustment_mm is None:
        reflector_cutout_width_adjustment_mm = (
            parameters.reflector_cutout_width_adjustment_mm
        )
    if reflector_cutout_depth_mm is None:
        reflector_cutout_depth_mm = parameters.reflector_cutout_depth_mm

    copper_thickness_mm = float(copper_thickness_mm)
    substrate_thickness_mm = float(substrate_thickness_mm)
    reflector_connector_board_thickness_mm = float(
        reflector_connector_board_thickness_mm
    )
    reflector_cutout_width_adjustment_mm = float(reflector_cutout_width_adjustment_mm)
    reflector_cutout_depth_mm = float(reflector_cutout_depth_mm)
    coordinate_quantum_mm = float(coordinate_quantum_mm)
    if not math.isfinite(coordinate_quantum_mm) or coordinate_quantum_mm <= 0.0:
        raise ValueError("coordinate quantum must be finite and positive")
    if not math.isfinite(copper_thickness_mm) or copper_thickness_mm <= 0.0:
        raise ValueError("copper thickness must be a finite positive number")
    if not math.isfinite(substrate_thickness_mm) or substrate_thickness_mm >= 0.0:
        raise ValueError("substrate thickness must be finite and negative")
    reflector_values = (
        reflector_connector_board_thickness_mm,
        reflector_cutout_width_adjustment_mm,
        reflector_cutout_depth_mm,
    )
    if not all(math.isfinite(value) for value in reflector_values):
        raise ValueError("reflector clearance dimensions must be finite")
    if reflector_connector_board_thickness_mm <= 0.0:
        raise ValueError("reflector connector board thickness must be positive")
    if reflector_cutout_width_adjustment_mm < 0.0:
        raise ValueError("reflector cutout width adjustment must be non-negative")
    if reflector_cutout_depth_mm <= 0.0:
        raise ValueError("reflector cutout depth must be positive")

    substrate_points = antenna_outline.generate_rectangle(parameters)
    conductor_point_lists = antenna_closed_curves.generate_closed_curve_point_lists(
        parameters,
        quantum_mm=coordinate_quantum_mm,
    )
    patch_points, slot_points, guide_points = conductor_point_lists
    reflector_points = list(substrate_points)
    reflector_cutout_half_width_mm = (
        parameters.rectangle_length_mm / 2.0
        - reflector_connector_board_thickness_mm
        + reflector_cutout_width_adjustment_mm
    )
    if (
        not 0.0
        < reflector_cutout_half_width_mm
        < (parameters.rectangle_length_mm / 2.0)
    ):
        raise ValueError("reflector cutout width does not fit inside the reflector")
    if reflector_cutout_depth_mm >= parameters.rectangle_width_mm:
        raise ValueError("reflector cutout depth must be smaller than the reflector")
    reflector_cutout_points = [
        (-reflector_cutout_half_width_mm, 0.0),
        (reflector_cutout_half_width_mm, 0.0),
        (reflector_cutout_half_width_mm, reflector_cutout_depth_mm),
        (-reflector_cutout_half_width_mm, reflector_cutout_depth_mm),
        (-reflector_cutout_half_width_mm, 0.0),
    ]
    raw_point_lists = [
        substrate_points,
        patch_points,
        slot_points,
        guide_points,
        reflector_points,
        reflector_cutout_points,
    ]
    curve_labels = (
        "substrate",
        *antenna_closed_curves.CURVE_NAMES,
        "reflector",
        "reflector connector clearance",
    )
    point_lists = [
        antenna_outline.quantize_and_validate_closed_polygon_points(
            points,
            curve_name=name,
            quantum_mm=coordinate_quantum_mm,
        )
        for name, points in zip(curve_labels, raw_point_lists, strict=True)
    ]
    (
        substrate_points,
        patch_points,
        slot_points,
        guide_points,
        reflector_points,
        reflector_cutout_points,
    ) = point_lists
    polygons = tuple(Polygon(points) for points in point_lists)
    (
        substrate_polygon,
        patch_polygon,
        slot_polygon,
        guide_polygon,
        reflector_polygon,
        reflector_cutout_polygon,
    ) = polygons

    for name, points, polygon in zip(
        curve_labels,
        point_lists,
        polygons,
        strict=True,
    ):
        if points[0] != points[-1]:
            raise ValueError(f"{name} point list is not explicitly closed")
        if _polygon_signed_area(points) <= 0.0:
            raise ValueError(
                f"{name} must retain counterclockwise point order; "
                "the thickness sign controls its Z direction"
            )
        if not polygon.is_valid or len(polygon.interiors) != 0:
            raise ValueError(f"{name} is not one valid hole-free Polygon")

    if not substrate_polygon.covers(slot_polygon):
        raise ValueError("the symmetric slot leaves the substrate")
    if not patch_polygon.intersects(slot_polygon):
        raise ValueError("the symmetric slot does not intersect the Patch")
    if not slot_polygon.covers(guide_polygon):
        raise ValueError("the symmetric guide is not fully contained in the slot")
    if not reflector_polygon.covers(reflector_cutout_polygon):
        raise ValueError("the connector clearance lies outside the reflector")

    final_conductor = patch_polygon.difference(slot_polygon).union(guide_polygon)
    if (
        not isinstance(final_conductor, (Polygon, MultiPolygon))
        or not final_conductor.is_valid
    ):
        raise ValueError(
            "Patch - slot + guide did not produce valid polygonal geometry"
        )
    final_conductor_component_count = (
        1 if isinstance(final_conductor, Polygon) else len(final_conductor.geoms)
    )
    if final_conductor_component_count > 1 and not allow_disconnected_conductor:
        raise ValueError(
            "Patch - slot + guide produced disconnected conductor geometry; "
            "set allow_disconnected_conductor=true only for an intentional trial"
        )
    final_reflector = reflector_polygon.difference(reflector_cutout_polygon)
    if not isinstance(final_reflector, Polygon) or not final_reflector.is_valid:
        raise ValueError(
            "reflector - connector clearance did not produce one valid Polygon"
        )

    specs = (
        CstPolygonSpec(
            label="substrate",
            curve_name=SUBSTRATE_CURVE_NAME,
            polygon_name=SUBSTRATE_POLYGON_NAME,
            solid_name=SUBSTRATE_SOLID_NAME,
            material_name=DEFAULT_SUBSTRATE_MATERIAL_NAME,
            thickness_mm=substrate_thickness_mm,
            points=substrate_points,
        ),
        CstPolygonSpec(
            label="Patch",
            curve_name=PATCH_CURVE_NAME,
            polygon_name=PATCH_POLYGON_NAME,
            solid_name=PATCH_SOLID_NAME,
            material_name=DEFAULT_COPPER_MATERIAL_NAME,
            thickness_mm=copper_thickness_mm,
            points=patch_points,
        ),
        CstPolygonSpec(
            label="symmetric slot",
            curve_name=SLOT_CURVE_NAME,
            polygon_name=SLOT_POLYGON_NAME,
            solid_name=SLOT_SOLID_NAME,
            material_name=DEFAULT_TOOL_MATERIAL_NAME,
            thickness_mm=copper_thickness_mm,
            points=slot_points,
        ),
        CstPolygonSpec(
            label="symmetric guide",
            curve_name=GUIDE_CURVE_NAME,
            polygon_name=GUIDE_POLYGON_NAME,
            solid_name=GUIDE_SOLID_NAME,
            material_name=DEFAULT_COPPER_MATERIAL_NAME,
            thickness_mm=copper_thickness_mm,
            points=guide_points,
        ),
        CstPolygonSpec(
            label="reflector",
            curve_name=REFLECTOR_CURVE_NAME,
            polygon_name=REFLECTOR_POLYGON_NAME,
            solid_name=REFLECTOR_SOLID_NAME,
            material_name=DEFAULT_COPPER_MATERIAL_NAME,
            thickness_mm=-copper_thickness_mm,
            points=reflector_points,
        ),
        CstPolygonSpec(
            label="reflector connector clearance",
            curve_name=REFLECTOR_CUTOUT_CURVE_NAME,
            polygon_name=REFLECTOR_CUTOUT_POLYGON_NAME,
            solid_name=REFLECTOR_CUTOUT_SOLID_NAME,
            material_name=DEFAULT_TOOL_MATERIAL_NAME,
            thickness_mm=-copper_thickness_mm,
            points=reflector_cutout_points,
        ),
    )
    report = GeometryBuildReport(
        point_counts=tuple(len(points) for points in point_lists),
        substrate_material_name=DEFAULT_SUBSTRATE_MATERIAL_NAME,
        substrate_relative_permittivity=(DEFAULT_SUBSTRATE_RELATIVE_PERMITTIVITY),
        substrate_area_mm2=float(substrate_polygon.area),
        substrate_volume_mm3=float(
            substrate_polygon.area * abs(substrate_thickness_mm)
        ),
        patch_area_mm2=float(patch_polygon.area),
        slot_area_mm2=float(slot_polygon.area),
        guide_area_mm2=float(guide_polygon.area),
        final_conductor_area_mm2=float(final_conductor.area),
        final_conductor_volume_mm3=float(final_conductor.area * copper_thickness_mm),
        final_conductor_component_count=final_conductor_component_count,
        coordinate_quantum_mm=coordinate_quantum_mm,
        reflector_cutout_width_mm=float(
            reflector_cutout_polygon.bounds[2] - reflector_cutout_polygon.bounds[0]
        ),
        reflector_cutout_depth_mm=float(
            reflector_cutout_polygon.bounds[3] - reflector_cutout_polygon.bounds[1]
        ),
        reflector_area_mm2=float(final_reflector.area),
        reflector_volume_mm3=float(final_reflector.area * copper_thickness_mm),
        reflector_z_min_mm=substrate_thickness_mm - copper_thickness_mm,
        reflector_z_max_mm=substrate_thickness_mm,
    )
    return specs, report


def build_prepare_project_vba(
    specs: Sequence[CstPolygonSpec],
    component_name: str = DEFAULT_COMPONENT_NAME,
) -> str:
    """Set millimetre units and remove only objects managed by this script."""

    solid_refs = [_solid_ref(component_name, spec.solid_name) for spec in specs]
    curve_names = [spec.curve_name for spec in specs]
    delete_solid_lines = "\n".join(
        f'    Solid.Delete "{solid_ref}"' for solid_ref in solid_refs
    )
    delete_curve_lines = "\n".join(
        f'    Curve.DeleteCurve "{curve_name}"' for curve_name in curve_names
    )
    return f"""
Sub Main()
    With Units
        .Geometry "mm"
    End With

    On Error Resume Next
    Component.New "{component_name}"
{delete_solid_lines}
{delete_curve_lines}
    On Error GoTo 0
End Sub
"""


def build_substrate_material_vba() -> str:
    """Create the selected CST-library laminate as a project material."""

    return f"""
Sub Main()
    On Error Resume Next
    Material.Delete "{DEFAULT_SUBSTRATE_MATERIAL_NAME}"
    On Error GoTo 0

    With Material
        .Reset
        .Name "{DEFAULT_SUBSTRATE_MATERIAL_NAME}"
        .FrqType "all"
        .Type "Normal"
        .SetMaterialUnit "GHz", "mm"
        .Epsilon "{DEFAULT_SUBSTRATE_RELATIVE_PERMITTIVITY:g}"
        .Mu "1.0"
        .Kappa "0.0"
        .TanD "0.003"
        .TanDFreq "10.0"
        .TanDGiven "True"
        .TanDModel "ConstTanD"
        .KappaM "0.0"
        .TanDM "0.0"
        .TanDMFreq "0.0"
        .TanDMGiven "False"
        .TanDMModel "ConstKappa"
        .DispModelEps "None"
        .DispModelMu "None"
        .DispersiveFittingSchemeEps "General 1st"
        .DispersiveFittingSchemeMu "General 1st"
        .UseGeneralDispersionEps "False"
        .UseGeneralDispersionMu "False"
        .Rho "0.0"
        .ThermalType "Normal"
        .ThermalConductivity "0.44"
        .SetActiveMaterial "all"
        .Colour "0.94", "0.82", "0.76"
        .Wireframe "False"
        .Transparency "0"
        .Create
    End With
End Sub
"""


def build_boolean_vba(
    component_name: str = DEFAULT_COMPONENT_NAME,
) -> str:
    """Subtract the slot tool and then add the guide to the Patch."""

    patch_ref = _solid_ref(component_name, PATCH_SOLID_NAME)
    slot_ref = _solid_ref(component_name, SLOT_SOLID_NAME)
    guide_ref = _solid_ref(component_name, GUIDE_SOLID_NAME)
    return f"""
Sub Main()
    Solid.Subtract "{patch_ref}", "{slot_ref}"
    Solid.Add "{patch_ref}", "{guide_ref}"
End Sub
"""


def build_reflector_boolean_vba(
    component_name: str = DEFAULT_COMPONENT_NAME,
) -> str:
    """Subtract the bottom-edge connector clearance from the reflector."""

    reflector_ref = _solid_ref(component_name, REFLECTOR_SOLID_NAME)
    cutout_ref = _solid_ref(component_name, REFLECTOR_CUTOUT_SOLID_NAME)
    return f"""
Sub Main()
    Solid.Subtract "{reflector_ref}", "{cutout_ref}"
End Sub
"""


def build_reflector_translation_vba(
    substrate_thickness_mm: float,
    component_name: str = DEFAULT_COMPONENT_NAME,
) -> str:
    """Move the reflector from Z=0 to the substrate's opposite face."""

    reflector_ref = _solid_ref(component_name, REFLECTOR_SOLID_NAME)
    return f"""
Sub Main()
    With Transform
        .Reset
        .Name "{reflector_ref}"
        .Vector "0", "0", "{substrate_thickness_mm:.15g}"
        .UsePickedPoints "False"
        .InvertPickedPoints "False"
        .MultipleObjects "False"
        .GroupObjects "False"
        .Repetitions "1"
        .MultipleSelection "False"
        .Component ""
        .Material ""
        .TranslateAdvanced
    End With
End Sub
"""


def build_verification_vba(
    specs: Sequence[CstPolygonSpec],
    report: GeometryBuildReport,
    component_name: str = DEFAULT_COMPONENT_NAME,
) -> str:
    """Assert substrate/conductor topology, materials, curves, and volumes."""

    substrate_ref = _solid_ref(component_name, SUBSTRATE_SOLID_NAME)
    patch_ref = _solid_ref(component_name, PATCH_SOLID_NAME)
    slot_ref = _solid_ref(component_name, SLOT_SOLID_NAME)
    guide_ref = _solid_ref(component_name, GUIDE_SOLID_NAME)
    reflector_ref = _solid_ref(component_name, REFLECTOR_SOLID_NAME)
    reflector_cutout_ref = _solid_ref(
        component_name,
        REFLECTOR_CUTOUT_SOLID_NAME,
    )
    expected_volume = report.final_conductor_volume_mm3
    volume_tolerance = max(1e-6, abs(expected_volume) * 1e-6)
    expected_substrate_volume = report.substrate_volume_mm3
    substrate_volume_tolerance = max(
        1e-6,
        abs(expected_substrate_volume) * 1e-6,
    )
    expected_reflector_volume = report.reflector_volume_mm3
    reflector_volume_tolerance = max(
        1e-6,
        abs(expected_reflector_volume) * 1e-6,
    )
    closed_curve_lines: list[str] = []
    for index, spec in enumerate(specs, start=1):
        closed_curve_lines.extend(
            (
                f'    If Not Curve.IsClosed("{_curve_item_ref(spec.curve_name, spec.polygon_name)}") Then',
                f'        Err.Raise vbObjectError + {1100 + index}, , "{spec.label} source curve is not closed"',
                "    End If",
            )
        )
    closed_curve_checks = "\n".join(closed_curve_lines)
    return f"""
Sub Main()
    Dim actualVolume As Double
    Dim actualSubstrateVolume As Double
    Dim actualSubstrateMaterial As String
    Dim actualReflectorVolume As Double
    Dim actualReflectorMaterial As String
    Dim reflectorXMin As Double
    Dim reflectorXMax As Double
    Dim reflectorYMin As Double
    Dim reflectorYMax As Double
    Dim reflectorZMin As Double
    Dim reflectorZMax As Double

    If Not Solid.DoesExist("{substrate_ref}") Then
        Err.Raise vbObjectError + 1000, , "MSA-BP substrate does not exist"
    End If
    If Not Solid.DoesExist("{patch_ref}") Then
        Err.Raise vbObjectError + 1001, , "final MSA-BP conductor does not exist"
    End If
    If Solid.DoesExist("{slot_ref}") Then
        Err.Raise vbObjectError + 1002, , "slot tool still exists after subtraction"
    End If
    If Solid.DoesExist("{guide_ref}") Then
        Err.Raise vbObjectError + 1003, , "guide tool still exists after unite"
    End If
    If Not Solid.DoesExist("{reflector_ref}") Then
        Err.Raise vbObjectError + 1007, , "MSA-BP reflector does not exist"
    End If
    If Solid.DoesExist("{reflector_cutout_ref}") Then
        Err.Raise vbObjectError + 1008, , "reflector clearance tool still exists"
    End If

{closed_curve_checks}

    actualSubstrateMaterial = Solid.GetMaterialNameForShape("{substrate_ref}")
    If StrComp(actualSubstrateMaterial, "{report.substrate_material_name}", vbTextCompare) <> 0 Then
        Err.Raise vbObjectError + 1005, , "substrate material mismatch"
    End If

    actualSubstrateVolume = Solid.GetVolume("{substrate_ref}")
    If Abs(actualSubstrateVolume - {expected_substrate_volume:.15g}) > {substrate_volume_tolerance:.15g} Then
        Err.Raise vbObjectError + 1006, , "substrate volume mismatch"
    End If

    actualVolume = Solid.GetVolume("{patch_ref}")
    If Abs(actualVolume - {expected_volume:.15g}) > {volume_tolerance:.15g} Then
        Err.Raise vbObjectError + 1004, , "final conductor volume mismatch"
    End If

    actualReflectorMaterial = Solid.GetMaterialNameForShape("{reflector_ref}")
    If StrComp(actualReflectorMaterial, "{DEFAULT_COPPER_MATERIAL_NAME}", vbTextCompare) <> 0 Then
        Err.Raise vbObjectError + 1009, , "reflector material mismatch"
    End If

    actualReflectorVolume = Solid.GetVolume("{reflector_ref}")
    If Abs(actualReflectorVolume - {expected_reflector_volume:.15g}) > {reflector_volume_tolerance:.15g} Then
        Err.Raise vbObjectError + 1010, , "reflector volume mismatch"
    End If

    If Not Solid.GetLooseBoundingBoxOfShape("{reflector_ref}", reflectorXMin, reflectorXMax, reflectorYMin, reflectorYMax, reflectorZMin, reflectorZMax) Then
        Err.Raise vbObjectError + 1011, , "could not query reflector bounding box"
    End If
    If Abs(reflectorZMin - {report.reflector_z_min_mm:.15g}) > 0.000001 Or Abs(reflectorZMax - {report.reflector_z_max_mm:.15g}) > 0.000001 Then
        Err.Raise vbObjectError + 1012, , "reflector Z placement mismatch"
    End If
End Sub
"""


def _build_live_vba_sequence(
    specs: Sequence[CstPolygonSpec],
    report: GeometryBuildReport,
    component_name: str,
) -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = [
        ("prepare project", build_prepare_project_vba(specs, component_name)),
    ]
    if report.substrate_material_name != VACUUM_SUBSTRATE_MATERIAL_NAME:
        commands.append(("define substrate material", build_substrate_material_vba()))
    for spec in specs:
        commands.extend(
            (
                (
                    f"create {spec.label} curve",
                    build_polygon_vba(
                        spec.points,
                        polygon_name=spec.polygon_name,
                        curve_name=spec.curve_name,
                    ),
                ),
                (
                    f"extrude {spec.label}",
                    build_extrude_curve_vba(
                        solid_name=spec.solid_name,
                        component_name=component_name,
                        material_name=spec.material_name,
                        thickness=spec.thickness_mm,
                        curve_name=spec.curve_name,
                        polygon_name=spec.polygon_name,
                    ),
                ),
            )
        )
    commands.extend(
        (
            ("boolean Patch - slot + guide", build_boolean_vba(component_name)),
            (
                "subtract reflector connector clearance",
                build_reflector_boolean_vba(component_name),
            ),
            (
                "move reflector to substrate back face",
                build_reflector_translation_vba(
                    report.reflector_z_max_mm,
                    component_name,
                ),
            ),
        )
    )
    for spec in specs:
        commands.append(
            (
                f"restore {spec.label} source curve",
                build_polygon_vba(
                    spec.points,
                    polygon_name=spec.polygon_name,
                    curve_name=spec.curve_name,
                ),
            )
        )
    commands.extend(
        (
            (
                "verify geometry",
                build_verification_vba(specs, report, component_name),
            ),
        )
    )
    return commands


def build_msabp_in_cst(
    project_path: Path = DEFAULT_PROJECT_PATH,
    component_name: str = DEFAULT_COMPONENT_NAME,
    parameters: (
        antenna_outline.AntennaOutlineParameters
        | shapely_antenna_model.ShapelyAntennaParameters
        | None
    ) = None,
    polygon_json_path: str | Path = DEFAULT_POLYGON_JSON_PATH,
    thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
    substrate_thickness_mm: float = DEFAULT_SUBSTRATE_THICKNESS_MM,
    substrate_material_name: str = DEFAULT_SUBSTRATE_MATERIAL_NAME,
    reflector_connector_board_thickness_mm: float | None = None,
    reflector_cutout_width_adjustment_mm: float | None = None,
    reflector_cutout_depth_mm: float | None = None,
    coordinate_quantum_mm: float = antenna_outline.COORDINATE_QUANTUM_MM,
    allow_disconnected_conductor: bool = False,
    timeout: float | None = 60.0,
    dry_run: bool = False,
    project: Any | None = None,
    save_project: bool = True,
) -> GeometryBuildReport:
    """Build and verify the antenna in a target or already connected project.

    The default behavior remains the CLI-compatible open/build/save workflow.
    Long-lived automation workers may pass an existing ``project`` and set
    ``save_project=False`` so one CST session can serve multiple cases without
    reopening or persisting every intermediate geometry.
    """

    if parameters is None:
        specs, report = build_exported_polygon_specs(
            polygon_json_path=polygon_json_path,
            copper_thickness_mm=thickness_mm,
            substrate_thickness_mm=substrate_thickness_mm,
            reflector_connector_board_thickness_mm=(
                DEFAULT_REFLECTOR_CONNECTOR_BOARD_THICKNESS_MM
                if reflector_connector_board_thickness_mm is None
                else reflector_connector_board_thickness_mm
            ),
            reflector_cutout_width_adjustment_mm=(
                DEFAULT_REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_MM
                if reflector_cutout_width_adjustment_mm is None
                else reflector_cutout_width_adjustment_mm
            ),
            reflector_cutout_depth_mm=(
                DEFAULT_REFLECTOR_CUTOUT_DEPTH_MM
                if reflector_cutout_depth_mm is None
                else reflector_cutout_depth_mm
            ),
        )
    elif isinstance(parameters, shapely_antenna_model.ShapelyAntennaParameters):
        specs, report = build_sampled_polygon_specs(
            parameters,
            copper_thickness_mm=thickness_mm,
            substrate_thickness_mm=substrate_thickness_mm,
            reflector_connector_board_thickness_mm=(
                DEFAULT_REFLECTOR_CONNECTOR_BOARD_THICKNESS_MM
                if reflector_connector_board_thickness_mm is None
                else reflector_connector_board_thickness_mm
            ),
            reflector_cutout_width_adjustment_mm=(
                DEFAULT_REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_MM
                if reflector_cutout_width_adjustment_mm is None
                else reflector_cutout_width_adjustment_mm
            ),
            reflector_cutout_depth_mm=(
                DEFAULT_REFLECTOR_CUTOUT_DEPTH_MM
                if reflector_cutout_depth_mm is None
                else reflector_cutout_depth_mm
            ),
            coordinate_quantum_mm=coordinate_quantum_mm,
        )
    else:
        # Transitional compatibility: distributed DoE rows still describe the
        # former parameterized geometry.  Keeping that explicit path prevents
        # those parameters from being silently ignored until the new producer
        # itself accepts sampled inputs.
        specs, report = build_polygon_specs(
            parameters=parameters,
            copper_thickness_mm=thickness_mm,
            substrate_thickness_mm=substrate_thickness_mm,
            reflector_connector_board_thickness_mm=(
                reflector_connector_board_thickness_mm
            ),
            reflector_cutout_width_adjustment_mm=(reflector_cutout_width_adjustment_mm),
            reflector_cutout_depth_mm=reflector_cutout_depth_mm,
            coordinate_quantum_mm=coordinate_quantum_mm,
            allow_disconnected_conductor=allow_disconnected_conductor,
        )

    specs, report = apply_substrate_material(
        specs,
        report,
        substrate_material_name,
    )
    commands = _build_live_vba_sequence(
        specs,
        report,
        component_name,
    )
    print(f"CST project: {Path(project_path).resolve()}")
    print(
        "source point counts "
        "(substrate, Patch, slot, guide, reflector, reflector clearance): "
        f"{report.point_counts}"
    )
    print(
        f"coordinate quantum: {report.coordinate_quantum_mm:g} mm; "
        f"final conductor components: {report.final_conductor_component_count}"
    )
    print(
        "substrate: "
        f"{report.substrate_material_name}, "
        f"epsilon_r={report.substrate_relative_permittivity:g}, "
        f"thickness={float(substrate_thickness_mm):g} mm, "
        f"area={report.substrate_area_mm2:.9f} mm^2, "
        f"volume={report.substrate_volume_mm3:.9f} mm^3"
    )
    print(
        "areas [mm^2]: "
        f"Patch={report.patch_area_mm2:.9f}, "
        f"slot={report.slot_area_mm2:.9f}, "
        f"guide={report.guide_area_mm2:.9f}, "
        f"final={report.final_conductor_area_mm2:.9f}"
    )
    print(f"expected final volume: {report.final_conductor_volume_mm3:.9f} mm^3")
    print(
        "reflector: "
        f"clearance={report.reflector_cutout_width_mm:g} x "
        f"{report.reflector_cutout_depth_mm:g} mm, "
        f"z={report.reflector_z_min_mm:g}..{report.reflector_z_max_mm:g} mm, "
        f"area={report.reflector_area_mm2:.9f} mm^2, "
        f"volume={report.reflector_volume_mm3:.9f} mm^3"
    )

    if dry_run:
        for label, vba in commands:
            print(f"\n--- {label} ---")
            print(vba.strip())
        return report

    project_path = Path(project_path).resolve()
    if project is None:
        if not project_path.is_file():
            raise FileNotFoundError(f"CST project does not exist: {project_path}")
        project = open_cst_project(str(project_path))

    for label, vba in commands:
        print(f"Executing: {label}")
        execute_project_vba(project, label, vba, timeout=timeout)

    if save_project:
        print("Executing: save project")
        execute_save_project(project, timeout=timeout)
        completion = "verified and saved"
    else:
        completion = "verified"
    print(f"CST build {completion}: {_solid_ref(component_name, PATCH_SOLID_NAME)}")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the current MSA-BP antenna geometry in CST."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT_PATH,
        help="Target CST project.",
    )
    parser.add_argument(
        "--component",
        default=DEFAULT_COMPONENT_NAME,
        help="CST component name.",
    )
    parser.add_argument(
        "--polygon-json",
        type=Path,
        default=DEFAULT_POLYGON_JSON_PATH,
        help="Three-curve JSON exported by shapely_rectangle_test.py.",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=DEFAULT_COPPER_THICKNESS_MM,
        help="Positive +Z copper extrusion thickness in mm.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for each execute_vba_code call.",
    )
    parser.add_argument(
        "--substrate-thickness",
        type=float,
        default=DEFAULT_SUBSTRATE_THICKNESS_MM,
        help="Negative -Z substrate extrusion thickness in mm.",
    )
    parser.add_argument(
        "--substrate-material",
        default=DEFAULT_SUBSTRATE_MATERIAL_NAME,
        choices=(DEFAULT_SUBSTRATE_MATERIAL_NAME, VACUUM_SUBSTRATE_MATERIAL_NAME),
        help="Substrate material; Vacuum is intended for controlled comparison runs.",
    )
    parser.add_argument(
        "--reflector-connector-board-thickness",
        type=float,
        default=None,
        help="Legacy BoardThick term used to derive reflector clearance width.",
    )
    parser.add_argument(
        "--reflector-cutout-width-adjustment",
        type=float,
        default=None,
        help="Legacy bplate_w_cut adjustment used for reflector clearance width.",
    )
    parser.add_argument(
        "--reflector-cutout-depth",
        type=float,
        default=None,
        help="Bottom-edge reflector connector-clearance depth in mm.",
    )
    parser.add_argument(
        "--coordinate-quantum",
        type=float,
        default=antenna_outline.COORDINATE_QUANTUM_MM,
        help="Coordinate grid applied before individual CST curve validation.",
    )
    parser.add_argument(
        "--allow-disconnected-conductor",
        action="store_true",
        help="Allow Patch-slot+guide to contain multiple metal components.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print VBA without opening CST.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_msabp_in_cst(
        project_path=args.project,
        component_name=args.component,
        polygon_json_path=args.polygon_json,
        thickness_mm=args.thickness,
        substrate_thickness_mm=args.substrate_thickness,
        substrate_material_name=args.substrate_material,
        reflector_connector_board_thickness_mm=(
            args.reflector_connector_board_thickness
        ),
        reflector_cutout_width_adjustment_mm=(args.reflector_cutout_width_adjustment),
        reflector_cutout_depth_mm=args.reflector_cutout_depth,
        coordinate_quantum_mm=args.coordinate_quantum,
        allow_disconnected_conductor=args.allow_disconnected_conductor,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
