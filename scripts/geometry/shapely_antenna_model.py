"""Parameterized Shapely model behind the 23-variable MSA-BP design space.

This module is the import-safe counterpart of ``shapely_rectangle_test.py``.
It contains no plotting, printing, or shared-file writes, so samplers and Maid
workers can build independent geometries in memory.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from shapely.affinity import scale
from shapely.geometry import LinearRing, LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_PATH = (
    REPOSITORY_ROOT / "results" / "processed" / "antenna_polygon_vertices.json"
)
QUANTIZE_STEP_MM = 0.01

# Fixed construction dimensions, deliberately excluded from optimization.
FIXED_OFFSET_MM = 1.0
PATCH_BRICK_2_WIDTH_MM = 12.0
PATCH_BRICK_4_FIXED_MM = 13.0
CPW_FEED_SLOT_WIDE_WIDTH_MM = 2.4
CPW_FEED_SLOT_NARROW_WIDTH_MM = 1.7
CPW_KEEPOUT_MARGIN_MM = 2.0
MATCHING_STUB_1_HALF_WIDTH_MM = 3.5
MATCHING_STUB_1_LOWER_OFFSET_MM = 5.5
MATCHING_STUB_1_UPPER_OFFSET_MM = 6.5
MATCHING_STUB_2_HALF_WIDTH_MM = 3.0
MATCHING_STUB_2_LOWER_OFFSET_MM = 8.0
MATCHING_STUB_2_UPPER_OFFSET_MM = 8.9
CPW_FEED_PIN_BASE_WIDTH_MM = 0.5
CPW_FEED_PIN_WIDE_WIDTH_MM = 1.375
CPW_FEED_PIN_CHAMFER_HEIGHT_MM = 0.3

Point2D = tuple[float, float]


@dataclass(frozen=True)
class ShapelyAntennaParameters:
    """Exactly seven millimetre variables and sixteen unit-ratio variables."""

    SLOT_MAIN_LENGTH: float = 53.0
    SLOT_MAIN_HEIGHT: float = 2.0
    PATCH_BRICK_1_SIDE_MARGIN: float = 6.0
    PATCH_BRICK_1_TOP_MARGIN: float = 2.6
    PATCH_BRICK_3_BOTTOM_MARGIN: float = 2.0
    PATCH_BRICK_2_HEIGHT_MARGIN: float = 15.0
    PATCH_BRICK_4_MARGIN: float = 4.0

    UPPER_CORNER_NOTCH_1_K1: float = 17.0 / 27.5
    UPPER_CORNER_NOTCH_1_K2: float = 14.0 / 15.0
    UPPER_CORNER_EAR_1_K1: float = 7.0 / 17.0
    UPPER_CORNER_EAR_1_K2: float = 1.0 / 14.0

    LOWER_CORNER_NOTCH_1_K1: float = 21.3 / 27.5
    LOWER_CORNER_NOTCH_1_K2: float = 12.0 / 17.0
    LOWER_CORNER_EAR_1_K1: float = 5.0 / 21.3
    LOWER_CORNER_EAR_1_K2: float = 4.0 / 6.0
    LOWER_CORNER_EAR_2_K1: float = 4.0 / 16.3
    LOWER_CORNER_EAR_2_K2: float = 1.5 / 6.0

    BRANCH_UP_1_K: float = 0.5
    BRANCH_UP_1_K2: float = 0.5
    BRANCH_UP_1_K3: float = 0.5
    BRANCH_DOWN_1_K: float = 0.5
    BRANCH_DOWN_1_K2: float = 0.5
    BRANCH_DOWN_1_K3: float = 0.0


DEFAULT_PARAMETERS = ShapelyAntennaParameters()
DEFAULT_ANTENNA_PARAMETERS = DEFAULT_PARAMETERS
ABSOLUTE_PARAMETER_NAMES = (
    "SLOT_MAIN_LENGTH",
    "SLOT_MAIN_HEIGHT",
    "PATCH_BRICK_1_SIDE_MARGIN",
    "PATCH_BRICK_1_TOP_MARGIN",
    "PATCH_BRICK_3_BOTTOM_MARGIN",
    "PATCH_BRICK_2_HEIGHT_MARGIN",
    "PATCH_BRICK_4_MARGIN",
)
RATIO_PARAMETER_NAMES = (
    "UPPER_CORNER_NOTCH_1_K1",
    "UPPER_CORNER_NOTCH_1_K2",
    "UPPER_CORNER_EAR_1_K1",
    "UPPER_CORNER_EAR_1_K2",
    "LOWER_CORNER_NOTCH_1_K1",
    "LOWER_CORNER_NOTCH_1_K2",
    "LOWER_CORNER_EAR_1_K1",
    "LOWER_CORNER_EAR_1_K2",
    "LOWER_CORNER_EAR_2_K1",
    "LOWER_CORNER_EAR_2_K2",
    "BRANCH_UP_1_K",
    "BRANCH_UP_1_K2",
    "BRANCH_UP_1_K3",
    "BRANCH_DOWN_1_K",
    "BRANCH_DOWN_1_K2",
    "BRANCH_DOWN_1_K3",
)
PARAMETER_NAMES = (*ABSOLUTE_PARAMETER_NAMES, *RATIO_PARAMETER_NAMES)


@dataclass(frozen=True)
class ShapelyAntennaGeometry:
    slot: Polygon
    patch: Polygon
    cpw_feed_pin: Polygon
    upper_substrate: Polygon
    lower_substrate: Polygon


def parameters_from_mapping(values: Mapping[str, Any]) -> ShapelyAntennaParameters:
    """Construct strongly typed parameters from one complete name/value mapping."""

    missing = set(PARAMETER_NAMES) - set(values)
    if missing:
        raise ValueError(f"missing antenna parameters: {sorted(missing)}")
    return ShapelyAntennaParameters(
        **{name: float(values[name]) for name in PARAMETER_NAMES}
    )


def _boundary_hit_ys(geometry: Any) -> list[float]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [float(geometry.y)]
    if geometry.geom_type == "LineString":
        return [float(coordinate[1]) for coordinate in geometry.coords]
    if hasattr(geometry, "geoms"):
        return [
            y_value
            for part in geometry.geoms
            for y_value in _boundary_hit_ys(part)
        ]
    return []


def _ray_distance(
    x_value: float,
    base_y: float,
    shape: Polygon,
    *,
    direction: int,
) -> float:
    ray_end = shape.bounds[3] + 10.0 if direction > 0 else shape.bounds[1] - 10.0
    ray = LineString([(x_value, base_y), (x_value, ray_end)])
    hit_ys = _boundary_hit_ys(ray.intersection(shape.boundary))
    if direction > 0:
        candidates = [value for value in hit_ys if value > base_y + 1e-9]
        if not candidates:
            raise ValueError("upward branch ray does not reach the upper substrate")
        return min(candidates) - base_y
    candidates = [value for value in hit_ys if value < base_y - 1e-9]
    if not candidates:
        raise ValueError("downward branch ray does not reach the lower substrate")
    return base_y - max(candidates)


def _build_branch_pair(
    *,
    k1: float,
    k2: float,
    k3: float,
    slot_min_x: float,
    slot_max_x: float,
    base_y: float,
    x_low: float,
    near_x: float,
    substrate: Polygon,
    direction: int,
) -> tuple[Polygon, Polygon]:
    x_high = slot_max_x - FIXED_OFFSET_MM
    midpoint_x = k1 * (x_high - x_low) + x_low
    max_half_width = min(slot_max_x - midpoint_x, midpoint_x - near_x)
    left_x = midpoint_x - k2 * max_half_width
    right_x = midpoint_x + k2 * max_half_width
    max_length = min(
        _ray_distance(left_x, base_y, substrate, direction=direction),
        _ray_distance(right_x, base_y, substrate, direction=direction),
    )
    length = k3 * max_length
    if direction > 0:
        branch = box(left_x, base_y, right_x, base_y + length)
    else:
        branch = box(left_x, base_y - length, right_x, base_y)
    return branch, scale(branch, xfact=-1, yfact=1, origin=(0, 0))


def build_antenna_geometry(
    parameters: ShapelyAntennaParameters = DEFAULT_PARAMETERS,
) -> ShapelyAntennaGeometry:
    """Build the three final polygons using the redesigned construction rules."""

    p = parameters
    values = asdict(p)
    non_finite = [name for name, value in values.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"antenna parameters must be finite: {non_finite}")
    non_positive = [
        name for name in ABSOLUTE_PARAMETER_NAMES if values[name] <= 0.0
    ]
    if non_positive:
        raise ValueError(f"millimetre parameters must be positive: {non_positive}")
    outside_unit_interval = [
        name
        for name in RATIO_PARAMETER_NAMES
        if not 0.0 <= values[name] <= 1.0
    ]
    if outside_unit_interval:
        raise ValueError(
            "ratio parameters must lie in [0, 1]: "
            f"{outside_unit_interval}"
        )

    slot_main = box(
        -p.SLOT_MAIN_LENGTH / 2.0,
        -p.SLOT_MAIN_HEIGHT / 2.0,
        p.SLOT_MAIN_LENGTH / 2.0,
        p.SLOT_MAIN_HEIGHT / 2.0,
    )
    slot_min_x, slot_min_y, slot_max_x, slot_max_y = slot_main.bounds
    extent_side = FIXED_OFFSET_MM + p.PATCH_BRICK_1_SIDE_MARGIN
    extent_up = FIXED_OFFSET_MM + p.PATCH_BRICK_1_TOP_MARGIN
    extent_down = FIXED_OFFSET_MM + p.PATCH_BRICK_3_BOTTOM_MARGIN

    patch_brick_1 = box(
        slot_min_x - extent_side,
        slot_min_y,
        slot_max_x + extent_side,
        slot_max_y + extent_up,
    )
    patch_brick_3 = box(
        slot_min_x - extent_side,
        slot_min_y - extent_down,
        slot_max_x + extent_side,
        slot_max_y,
    )
    patch_brick_2_height = extent_up + p.PATCH_BRICK_2_HEIGHT_MARGIN
    patch_brick_2 = box(
        -PATCH_BRICK_2_WIDTH_MM / 2.0,
        slot_max_y,
        PATCH_BRICK_2_WIDTH_MM / 2.0,
        slot_max_y + patch_brick_2_height,
    )
    patch_brick_4_height = (
        extent_down + p.PATCH_BRICK_4_MARGIN + PATCH_BRICK_4_FIXED_MM
    )
    patch_brick_4 = box(
        -PATCH_BRICK_2_WIDTH_MM / 2.0,
        slot_min_y - patch_brick_4_height,
        PATCH_BRICK_2_WIDTH_MM / 2.0,
        slot_min_y,
    )

    upper_base_shapes = (slot_main, patch_brick_1, patch_brick_2)
    sub_min_x = min(shape.bounds[0] for shape in upper_base_shapes)
    sub_min_y = min(shape.bounds[1] for shape in upper_base_shapes)
    sub_max_x = max(shape.bounds[2] for shape in upper_base_shapes)
    sub_max_y = max(shape.bounds[3] for shape in upper_base_shapes)
    substrate_top = box(sub_min_x, sub_min_y, sub_max_x, sub_max_y)
    substrate_width = sub_max_x - sub_min_x
    upper_side_span = (substrate_width - PATCH_BRICK_2_WIDTH_MM) / 2.0
    upper_notch_width = upper_side_span * p.UPPER_CORNER_NOTCH_1_K1
    upper_notch_height = (
        p.PATCH_BRICK_2_HEIGHT_MARGIN * p.UPPER_CORNER_NOTCH_1_K2
    )
    upper_notch_1 = box(
        sub_max_x - upper_notch_width,
        sub_max_y - upper_notch_height,
        sub_max_x,
        sub_max_y,
    )
    upper_ear_1 = box(
        upper_notch_1.bounds[0],
        upper_notch_1.bounds[1],
        upper_notch_1.bounds[0] + p.UPPER_CORNER_EAR_1_K1 * upper_notch_width,
        upper_notch_1.bounds[1] + p.UPPER_CORNER_EAR_1_K2 * upper_notch_height,
    )
    upper_notch_2 = scale(upper_notch_1, xfact=-1, yfact=1, origin=(0, 0))
    upper_ear_2 = scale(upper_ear_1, xfact=-1, yfact=1, origin=(0, 0))
    upper_substrate = (
        substrate_top.difference(upper_notch_1)
        .difference(upper_notch_2)
        .union(upper_ear_1)
        .union(upper_ear_2)
    )

    lower_base_shapes = (patch_brick_3, patch_brick_4)
    lower_min_x = min(shape.bounds[0] for shape in lower_base_shapes)
    lower_min_y = min(shape.bounds[1] for shape in lower_base_shapes)
    lower_max_x = max(shape.bounds[2] for shape in lower_base_shapes)
    lower_max_y = max(shape.bounds[3] for shape in lower_base_shapes)
    substrate_bottom = box(lower_min_x, lower_min_y, lower_max_x, lower_max_y)
    lower_notch_width = upper_side_span * p.LOWER_CORNER_NOTCH_1_K1
    lower_notch_height = (
        (p.PATCH_BRICK_4_MARGIN + PATCH_BRICK_4_FIXED_MM)
        * p.LOWER_CORNER_NOTCH_1_K2
    )
    lower_notch_1 = box(
        lower_max_x - lower_notch_width,
        lower_min_y,
        lower_max_x,
        lower_min_y + lower_notch_height,
    )
    lower_ear_1_width = p.LOWER_CORNER_EAR_1_K1 * lower_notch_width
    lower_ear_1_height = p.LOWER_CORNER_EAR_1_K2 * lower_notch_height / 2.0
    lower_ear_1 = box(
        lower_notch_1.bounds[0],
        lower_notch_1.bounds[3] - lower_ear_1_height,
        lower_notch_1.bounds[0] + lower_ear_1_width,
        lower_notch_1.bounds[3],
    )

    ear_2_x1, ear_2_y1 = lower_ear_1.bounds[2], lower_ear_1.bounds[3]
    ear_2_x2, ear_2_y2 = lower_notch_1.bounds[2], lower_notch_1.bounds[1]
    ear_2_available_width = ear_2_x2 - ear_2_x1
    ear_2_available_half_height = (ear_2_y1 - ear_2_y2) / 2.0
    lower_ear_2 = box(
        ear_2_x1,
        ear_2_y1 - p.LOWER_CORNER_EAR_2_K2 * ear_2_available_half_height,
        ear_2_x1 + p.LOWER_CORNER_EAR_2_K1 * ear_2_available_width,
        ear_2_y1,
    )
    lower_mid_y = (lower_notch_1.bounds[1] + lower_notch_1.bounds[3]) / 2.0
    lower_ear_3 = scale(lower_ear_1, xfact=1, yfact=-1, origin=(0, lower_mid_y))
    lower_ear_4 = scale(lower_ear_2, xfact=1, yfact=-1, origin=(0, lower_mid_y))
    right_lower_features = (lower_ear_1, lower_ear_2, lower_ear_3, lower_ear_4)
    lower_notch_2 = scale(lower_notch_1, xfact=-1, yfact=1, origin=(0, 0))
    left_lower_features = tuple(
        scale(feature, xfact=-1, yfact=1, origin=(0, 0))
        for feature in right_lower_features
    )
    lower_substrate = substrate_bottom.difference(lower_notch_1).difference(
        lower_notch_2
    )
    for feature in (*right_lower_features, *left_lower_features):
        lower_substrate = lower_substrate.union(feature)

    cpw_slot_1 = Polygon(
        [
            (0.0, lower_min_y),
            (CPW_FEED_SLOT_WIDE_WIDTH_MM, lower_min_y),
            (CPW_FEED_SLOT_WIDE_WIDTH_MM, lower_min_y + 11.0),
            (CPW_FEED_SLOT_NARROW_WIDTH_MM, lower_min_y + 12.0),
            (CPW_FEED_SLOT_NARROW_WIDTH_MM, slot_min_y),
            (0.0, slot_min_y),
        ]
    )
    cpw_slot_2 = scale(cpw_slot_1, xfact=-1, yfact=1, origin=(0, 0))
    matching_stub_1 = box(
        -MATCHING_STUB_1_HALF_WIDTH_MM,
        lower_min_y + MATCHING_STUB_1_LOWER_OFFSET_MM,
        MATCHING_STUB_1_HALF_WIDTH_MM,
        lower_min_y + MATCHING_STUB_1_UPPER_OFFSET_MM,
    )
    matching_stub_2 = box(
        -MATCHING_STUB_2_HALF_WIDTH_MM,
        lower_min_y + MATCHING_STUB_2_LOWER_OFFSET_MM,
        MATCHING_STUB_2_HALF_WIDTH_MM,
        lower_min_y + MATCHING_STUB_2_UPPER_OFFSET_MM,
    )
    cpw_pin_1 = Polygon(
        [
            (0.0, lower_min_y),
            (CPW_FEED_PIN_BASE_WIDTH_MM, lower_min_y),
            (CPW_FEED_PIN_WIDE_WIDTH_MM, lower_min_y + CPW_FEED_PIN_CHAMFER_HEIGHT_MM),
            (CPW_FEED_PIN_WIDE_WIDTH_MM, lower_min_y + 11.0),
            (CPW_FEED_PIN_BASE_WIDTH_MM, lower_min_y + 12.0),
            (CPW_FEED_PIN_BASE_WIDTH_MM, slot_max_y),
            (0.0, slot_max_y),
        ]
    )
    cpw_pin_2 = scale(cpw_pin_1, xfact=-1, yfact=1, origin=(0, 0))

    branch_up = _build_branch_pair(
        k1=p.BRANCH_UP_1_K,
        k2=p.BRANCH_UP_1_K2,
        k3=p.BRANCH_UP_1_K3,
        slot_min_x=slot_min_x,
        slot_max_x=slot_max_x,
        base_y=slot_max_y,
        x_low=2.0,
        near_x=1.0,
        substrate=upper_substrate,
        direction=1,
    )
    keepout_x = CPW_FEED_SLOT_WIDE_WIDTH_MM + CPW_KEEPOUT_MARGIN_MM
    branch_down = _build_branch_pair(
        k1=p.BRANCH_DOWN_1_K,
        k2=p.BRANCH_DOWN_1_K2,
        k3=p.BRANCH_DOWN_1_K3,
        slot_min_x=slot_min_x,
        slot_max_x=slot_max_x,
        base_y=slot_min_y,
        x_low=keepout_x,
        near_x=keepout_x,
        substrate=lower_substrate,
        direction=-1,
    )
    slot = unary_union(
        [
            slot_main,
            *branch_up,
            *branch_down,
            cpw_slot_1,
            cpw_slot_2,
            matching_stub_1,
            matching_stub_2,
        ]
    )
    patch = unary_union([upper_substrate, lower_substrate])
    cpw_feed_pin = unary_union([cpw_pin_1, cpw_pin_2])
    for name, geometry in {
        "Slot": slot,
        "Patch": patch,
        "CPW_Feed_Pin": cpw_feed_pin,
    }.items():
        if not isinstance(geometry, Polygon):
            raise ValueError(f"{name} construction returned {geometry.geom_type}")
    return ShapelyAntennaGeometry(
        slot=slot,
        patch=patch,
        cpw_feed_pin=cpw_feed_pin,
        upper_substrate=upper_substrate,
        lower_substrate=lower_substrate,
    )


def polygon_export_payload(
    parameters: ShapelyAntennaParameters = DEFAULT_PARAMETERS,
    *,
    quantize_step_mm: float = QUANTIZE_STEP_MM,
) -> dict[str, Any]:
    """Return the same three-curve JSON payload as the exploratory script."""

    geometry = build_antenna_geometry(parameters)
    polygons = {
        "Slot": geometry.slot,
        "Patch": geometry.patch,
        "CPW_Feed_Pin": geometry.cpw_feed_pin,
    }
    quantized_vertices: dict[str, list[Point2D]] = {}
    for name, polygon in polygons.items():
        coordinates = list(orient(polygon, sign=1.0).exterior.coords)[:-1]
        quantized_vertices[name] = [
            (
                round(x_value / quantize_step_mm) * quantize_step_mm,
                round(y_value / quantize_step_mm) * quantize_step_mm,
            )
            for x_value, y_value in coordinates
        ]
    global_min_y = min(
        y_value
        for points in quantized_vertices.values()
        for _, y_value in points
    )
    shifted_vertices = {
        name: [
            (x_value, round(y_value - global_min_y, 2))
            for x_value, y_value in points
        ]
        for name, points in quantized_vertices.items()
    }
    checks = {}
    for name, points in shifted_vertices.items():
        ring = LinearRing([*points, points[0]])
        polygon = Polygon(points)
        checks[name] = {
            "ring_is_simple": bool(ring.is_simple),
            "polygon_is_valid": bool(polygon.is_valid),
        }
    return {
        "meta": {
            "quantize_step": quantize_step_mm,
            "global_min_y_before_shift": global_min_y,
            "parameters": asdict(parameters),
            "self_intersection_check": checks,
        },
        "vertices": shifted_vertices,
    }


def write_polygon_export(
    parameters: ShapelyAntennaParameters = DEFAULT_PARAMETERS,
    path: str | Path = DEFAULT_EXPORT_PATH,
) -> Path:
    """Atomically write one sampled design using the established JSON format."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(polygon_export_payload(parameters), indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output
