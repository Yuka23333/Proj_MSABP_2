"""Minimal shapely smoke test: draw a 53 x 2 rectangle."""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from shapely.affinity import scale
from shapely.geometry import LinearRing, LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

SLOT_MAIN_LENGTH = 53  # optimizable
SLOT_MAIN_HEIGHT = 2  # optimizable

slot_main = box(
    -SLOT_MAIN_LENGTH / 2, -SLOT_MAIN_HEIGHT / 2,
    SLOT_MAIN_LENGTH / 2, SLOT_MAIN_HEIGHT / 2,
)

# Fixed clearance that always remains even if the optimizable margins below go to 0.
FIXED_OFFSET = 1

PATCH_BRICK_1_SIDE_MARGIN = 6  # optimizable
PATCH_BRICK_1_TOP_MARGIN = 2.6  # optimizable

EXT_SIDE = FIXED_OFFSET + PATCH_BRICK_1_SIDE_MARGIN
EXT_UP = FIXED_OFFSET + PATCH_BRICK_1_TOP_MARGIN

slot_min_x, slot_min_y, slot_max_x, slot_max_y = slot_main.bounds
Patch_Brick_1 = box(
    slot_min_x - EXT_SIDE, slot_min_y,
    slot_max_x + EXT_SIDE, slot_max_y + EXT_UP,
)

PATCH_BRICK_3_BOTTOM_MARGIN = 2  # optimizable
EXT_DOWN_3 = FIXED_OFFSET + PATCH_BRICK_3_BOTTOM_MARGIN

Patch_Brick_3 = box(
    slot_min_x - EXT_SIDE, slot_min_y - EXT_DOWN_3,
    slot_max_x + EXT_SIDE, slot_max_y,
)

PATCH_BRICK_2_WIDTH = 12  # constant
PATCH_BRICK_2_HEIGHT_MARGIN = 15  # optimizable

Patch_Brick_2_height = EXT_UP + PATCH_BRICK_2_HEIGHT_MARGIN
Patch_Brick_2 = box(
    -PATCH_BRICK_2_WIDTH / 2, slot_max_y,
    PATCH_BRICK_2_WIDTH / 2, slot_max_y + Patch_Brick_2_height,
)

PATCH_BRICK_4_MARGIN = 4  # optimizable
PATCH_BRICK_4_FIXED = 13  # fixed, not optimized

Patch_Brick_4_height = EXT_DOWN_3 + PATCH_BRICK_4_MARGIN + PATCH_BRICK_4_FIXED
Patch_Brick_4 = box(
    -PATCH_BRICK_2_WIDTH / 2, slot_min_y - Patch_Brick_4_height,
    PATCH_BRICK_2_WIDTH / 2, slot_min_y,
)

all_shapes = (slot_main, Patch_Brick_1, Patch_Brick_2)
sub_min_x = min(s.bounds[0] for s in all_shapes)
sub_min_y = min(s.bounds[1] for s in all_shapes)
sub_max_x = max(s.bounds[2] for s in all_shapes)
sub_max_y = max(s.bounds[3] for s in all_shapes)
Substrate_Top = box(sub_min_x, sub_min_y, sub_max_x, sub_max_y)

substrate_width = sub_max_x - sub_min_x

UPPER_CORNER_NOTCH_1_K1 = 17 / ((substrate_width - PATCH_BRICK_2_WIDTH) / 2)  # optimizable
UPPER_CORNER_NOTCH_1_K2 = 14 / PATCH_BRICK_2_HEIGHT_MARGIN  # optimizable

UPPER_CORNER_NOTCH_1_WIDTH = -(substrate_width - PATCH_BRICK_2_WIDTH) / 2 * UPPER_CORNER_NOTCH_1_K1
UPPER_CORNER_NOTCH_1_HEIGHT = -PATCH_BRICK_2_HEIGHT_MARGIN * UPPER_CORNER_NOTCH_1_K2

Upper_Corner_Notch_1 = box(
    sub_max_x + UPPER_CORNER_NOTCH_1_WIDTH, sub_max_y + UPPER_CORNER_NOTCH_1_HEIGHT,
    sub_max_x, sub_max_y,
)

X1, Y1 = Upper_Corner_Notch_1.bounds[0], Upper_Corner_Notch_1.bounds[1]
L = -UPPER_CORNER_NOTCH_1_WIDTH  # Upper_Corner_Notch_1 side length along X
H = -UPPER_CORNER_NOTCH_1_HEIGHT  # Upper_Corner_Notch_1 side length along Y

UPPER_CORNER_EAR_1_K1 = 7 / L  # optimizable
UPPER_CORNER_EAR_1_K2 = 1 / H  # optimizable

UPPER_CORNER_EAR_1_WIDTH = UPPER_CORNER_EAR_1_K1 * L
UPPER_CORNER_EAR_1_HEIGHT = UPPER_CORNER_EAR_1_K2 * H
UPPER_CORNER_EAR_1_ATTRIBUTE = "positive"  # small tab added on top of the patch, not cut out

Upper_Corner_Ear_1 = box(
    X1, Y1,
    X1 + UPPER_CORNER_EAR_1_WIDTH, Y1 + UPPER_CORNER_EAR_1_HEIGHT,
)

# Mirror the right-side corner features across the Y axis (x -> -x) onto the left side.
Upper_Corner_Notch_2 = scale(Upper_Corner_Notch_1, xfact=-1, yfact=1, origin=(0, 0))
Upper_Corner_Ear_2 = scale(Upper_Corner_Ear_1, xfact=-1, yfact=1, origin=(0, 0))

Upper_Substrate = (
    Substrate_Top
    .difference(Upper_Corner_Notch_1)
    .difference(Upper_Corner_Notch_2)
    .union(Upper_Corner_Ear_1)
    .union(Upper_Corner_Ear_2)
)

all_shapes_lower = (Patch_Brick_3, Patch_Brick_4)
lower_sub_min_x = min(s.bounds[0] for s in all_shapes_lower)
lower_sub_min_y = min(s.bounds[1] for s in all_shapes_lower)
lower_sub_max_x = max(s.bounds[2] for s in all_shapes_lower)
lower_sub_max_y = max(s.bounds[3] for s in all_shapes_lower)
Substrate_Bottom = box(lower_sub_min_x, lower_sub_min_y, lower_sub_max_x, lower_sub_max_y)

LOWER_CORNER_NOTCH_1_K1 = 21.3 / ((substrate_width - PATCH_BRICK_2_WIDTH) / 2)  # optimizable
LOWER_CORNER_NOTCH_1_K2 = 12 / (PATCH_BRICK_4_MARGIN + PATCH_BRICK_4_FIXED)  # optimizable

LOWER_CORNER_NOTCH_1_WIDTH = -(substrate_width - PATCH_BRICK_2_WIDTH) / 2 * LOWER_CORNER_NOTCH_1_K1
LOWER_CORNER_NOTCH_1_HEIGHT = (PATCH_BRICK_4_MARGIN + PATCH_BRICK_4_FIXED) * LOWER_CORNER_NOTCH_1_K2

# Cut from the bottom-right corner of Substrate_Bottom, extending left and up.
Lower_Corner_Notch_1 = box(
    lower_sub_max_x + LOWER_CORNER_NOTCH_1_WIDTH, lower_sub_min_y,
    lower_sub_max_x, lower_sub_min_y + LOWER_CORNER_NOTCH_1_HEIGHT,
)

# Lower_Corner_Notch_1 side lengths, kept around for the ear below.
LOWER_CORNER_NOTCH_1_X_LEN = -LOWER_CORNER_NOTCH_1_WIDTH
LOWER_CORNER_NOTCH_1_Y_LEN = LOWER_CORNER_NOTCH_1_HEIGHT

LOWER_CORNER_EAR_1_X1 = 5
LOWER_CORNER_EAR_1_Y1 = 4

LOWER_CORNER_EAR_1_K1 = LOWER_CORNER_EAR_1_X1 / LOWER_CORNER_NOTCH_1_X_LEN  # optimizable
LOWER_CORNER_EAR_1_K2 = LOWER_CORNER_EAR_1_Y1 / (LOWER_CORNER_NOTCH_1_Y_LEN / 2)  # optimizable

LOWER_CORNER_EAR_1_WIDTH = LOWER_CORNER_EAR_1_K1 * LOWER_CORNER_NOTCH_1_X_LEN
LOWER_CORNER_EAR_1_HEIGHT = LOWER_CORNER_EAR_1_K2 * (LOWER_CORNER_NOTCH_1_Y_LEN / 2)
LOWER_CORNER_EAR_1_ATTRIBUTE = "positive"

# Top-left corner of Lower_Corner_Notch_1, ear extends right and down from here.
X2, Y2 = Lower_Corner_Notch_1.bounds[0], Lower_Corner_Notch_1.bounds[3]
Lower_Corner_Ear_1 = box(
    X2, Y2 - LOWER_CORNER_EAR_1_HEIGHT,
    X2 + LOWER_CORNER_EAR_1_WIDTH, Y2,
)

# Ear2 starts at Ear1's top-right corner (X1, Y1); Notch1's bottom-right corner is (X2, Y2).
EAR2_X1, EAR2_Y1 = Lower_Corner_Ear_1.bounds[2], Lower_Corner_Ear_1.bounds[3]
EAR2_X2, EAR2_Y2 = Lower_Corner_Notch_1.bounds[2], Lower_Corner_Notch_1.bounds[1]

LOWER_CORNER_EAR_2_K1 = 4 / (EAR2_X2 - EAR2_X1)  # optimizable
LOWER_CORNER_EAR_2_K2 = 1.5 / ((EAR2_Y1 - EAR2_Y2) / 2)  # optimizable

LOWER_CORNER_EAR_2_WIDTH = LOWER_CORNER_EAR_2_K1 * (EAR2_X2 - EAR2_X1)
LOWER_CORNER_EAR_2_HEIGHT = LOWER_CORNER_EAR_2_K2 * (EAR2_Y1 - EAR2_Y2) / 2
LOWER_CORNER_EAR_2_ATTRIBUTE = "positive"

Lower_Corner_Ear_2 = box(
    EAR2_X1, EAR2_Y1 - LOWER_CORNER_EAR_2_HEIGHT,
    EAR2_X1 + LOWER_CORNER_EAR_2_WIDTH, EAR2_Y1,
)

# Y3: midline between Lower_Corner_Notch_1's top and bottom edges
# (its bottom edge is also Substrate_Bottom's bottom edge).
Y3 = (Lower_Corner_Notch_1.bounds[1] + Lower_Corner_Notch_1.bounds[3]) / 2

Lower_Corner_Ear_3 = scale(Lower_Corner_Ear_1, xfact=1, yfact=-1, origin=(0, Y3))
Lower_Corner_Ear_4 = scale(Lower_Corner_Ear_2, xfact=1, yfact=-1, origin=(0, Y3))

# Mirror the notch and all four ears across the Y axis (x -> -x) onto the left side.
Lower_Corner_Notch_2 = scale(Lower_Corner_Notch_1, xfact=-1, yfact=1, origin=(0, 0))
Lower_Corner_Ear_5 = scale(Lower_Corner_Ear_1, xfact=-1, yfact=1, origin=(0, 0))
Lower_Corner_Ear_6 = scale(Lower_Corner_Ear_2, xfact=-1, yfact=1, origin=(0, 0))
Lower_Corner_Ear_7 = scale(Lower_Corner_Ear_3, xfact=-1, yfact=1, origin=(0, 0))
Lower_Corner_Ear_8 = scale(Lower_Corner_Ear_4, xfact=-1, yfact=1, origin=(0, 0))

Lower_Substrate = (
    Substrate_Bottom
    .difference(Lower_Corner_Notch_1)
    .difference(Lower_Corner_Notch_2)
    .union(Lower_Corner_Ear_1)
    .union(Lower_Corner_Ear_2)
    .union(Lower_Corner_Ear_3)
    .union(Lower_Corner_Ear_4)
    .union(Lower_Corner_Ear_5)
    .union(Lower_Corner_Ear_6)
    .union(Lower_Corner_Ear_7)
    .union(Lower_Corner_Ear_8)
)

print("Rectangle:", slot_main)
print("Area:", slot_main.area)
print("Bounds:", slot_main.bounds)

# print("Patch_Brick_1:", Patch_Brick_1)
# print("Patch_Brick_1 Bounds:", Patch_Brick_1.bounds)

# print("Patch_Brick_2:", Patch_Brick_2)
# print("Patch_Brick_2 Bounds:", Patch_Brick_2.bounds)

print("Patch_Brick_3:", Patch_Brick_3)
print("Patch_Brick_3 Bounds:", Patch_Brick_3.bounds)

print("Patch_Brick_4:", Patch_Brick_4)
print("Patch_Brick_4 Bounds:", Patch_Brick_4.bounds)

print("Substrate_Top:", Substrate_Top)
print("Substrate_Top Bounds:", Substrate_Top.bounds)

print("Upper_Corner_Notch_1:", Upper_Corner_Notch_1)
print("Upper_Corner_Notch_1 Bounds:", Upper_Corner_Notch_1.bounds)

print("Upper_Corner_Ear_1:", Upper_Corner_Ear_1)
print("Upper_Corner_Ear_1 Bounds:", Upper_Corner_Ear_1.bounds)
print("Upper_Corner_Ear_1 Attribute:", UPPER_CORNER_EAR_1_ATTRIBUTE)

print("Upper_Corner_Notch_2:", Upper_Corner_Notch_2)
print("Upper_Corner_Notch_2 Bounds:", Upper_Corner_Notch_2.bounds)

print("Upper_Corner_Ear_2:", Upper_Corner_Ear_2)
print("Upper_Corner_Ear_2 Bounds:", Upper_Corner_Ear_2.bounds)

print("Upper_Substrate:", Upper_Substrate)
print("Upper_Substrate Bounds:", Upper_Substrate.bounds)

print("Substrate_Bottom:", Substrate_Bottom)
print("Substrate_Bottom Bounds:", Substrate_Bottom.bounds)

print("Lower_Corner_Notch_1:", Lower_Corner_Notch_1)
print("Lower_Corner_Notch_1 Bounds:", Lower_Corner_Notch_1.bounds)

print("Lower_Corner_Ear_1:", Lower_Corner_Ear_1)
print("Lower_Corner_Ear_1 Bounds:", Lower_Corner_Ear_1.bounds)
print("Lower_Corner_Ear_1 Attribute:", LOWER_CORNER_EAR_1_ATTRIBUTE)

print("Lower_Corner_Ear_2:", Lower_Corner_Ear_2)
print("Lower_Corner_Ear_2 Bounds:", Lower_Corner_Ear_2.bounds)
print("Lower_Corner_Ear_2 Attribute:", LOWER_CORNER_EAR_2_ATTRIBUTE)

print("Lower_Corner_Ear_3:", Lower_Corner_Ear_3)
print("Lower_Corner_Ear_3 Bounds:", Lower_Corner_Ear_3.bounds)

print("Lower_Corner_Ear_4:", Lower_Corner_Ear_4)
print("Lower_Corner_Ear_4 Bounds:", Lower_Corner_Ear_4.bounds)

print("Lower_Corner_Notch_2:", Lower_Corner_Notch_2)
print("Lower_Corner_Notch_2 Bounds:", Lower_Corner_Notch_2.bounds)

print("Lower_Corner_Ear_5:", Lower_Corner_Ear_5)
print("Lower_Corner_Ear_5 Bounds:", Lower_Corner_Ear_5.bounds)

print("Lower_Corner_Ear_6:", Lower_Corner_Ear_6)
print("Lower_Corner_Ear_6 Bounds:", Lower_Corner_Ear_6.bounds)

print("Lower_Corner_Ear_7:", Lower_Corner_Ear_7)
print("Lower_Corner_Ear_7 Bounds:", Lower_Corner_Ear_7.bounds)

print("Lower_Corner_Ear_8:", Lower_Corner_Ear_8)
print("Lower_Corner_Ear_8 Bounds:", Lower_Corner_Ear_8.bounds)

print("Lower_Substrate:", Lower_Substrate)
print("Lower_Substrate Bounds:", Lower_Substrate.bounds)

# SMA connector GND solder-pad markers. Visual annotation only -- does not cut Substrate_Bottom.
SMA_GND_PAD_X_LOW = 3.49
SMA_GND_PAD_X_HIGH = 4.76
SMA_GND_PAD_HEIGHT = 4.5

SMA_GND_Pad_1 = box(
    SMA_GND_PAD_X_LOW, lower_sub_min_y,
    SMA_GND_PAD_X_HIGH, lower_sub_min_y + SMA_GND_PAD_HEIGHT,
)
SMA_GND_Pad_2 = scale(SMA_GND_Pad_1, xfact=-1, yfact=1, origin=(0, 0))

print("SMA_GND_Pad_1:", SMA_GND_Pad_1)
print("SMA_GND_Pad_1 Bounds:", SMA_GND_Pad_1.bounds)
print("SMA_GND_Pad_2:", SMA_GND_Pad_2)
print("SMA_GND_Pad_2 Bounds:", SMA_GND_Pad_2.bounds)

CPW_FEED_SLOT_WIDE_WIDTH = 2.4  # defensive: no optimizable use yet
CPW_FEED_SLOT_NARROW_WIDTH = 1.7  # defensive: no optimizable use yet
CPW_FEED_SLOT_ATTRIBUTE = "Negative/Slot"

# (0, 0) = midpoint of Substrate_Bottom's bottom edge -- ordinary y's are relative to that
# (shift by lower_sub_min_y), except the two points pinned to slot_min_y, which is already
# the absolute Y where the feed line reaches up to meet main_slot's bottom edge.
CPW_Feed_Slot_1 = Polygon([
    (0, lower_sub_min_y),
    (CPW_FEED_SLOT_WIDE_WIDTH, lower_sub_min_y),
    (CPW_FEED_SLOT_WIDE_WIDTH, lower_sub_min_y + 11),
    (CPW_FEED_SLOT_NARROW_WIDTH, lower_sub_min_y + 12),
    (CPW_FEED_SLOT_NARROW_WIDTH, slot_min_y),
    (0, slot_min_y),
])

# Mirror across the Y axis (x -> -x) onto the left side.
CPW_Feed_Slot_2 = scale(CPW_Feed_Slot_1, xfact=-1, yfact=1, origin=(0, 0))

print("CPW_Feed_Slot_1:", CPW_Feed_Slot_1)
print("CPW_Feed_Slot_1 Bounds:", CPW_Feed_Slot_1.bounds)
print("CPW_Feed_Slot_2:", CPW_Feed_Slot_2)
print("CPW_Feed_Slot_2 Bounds:", CPW_Feed_Slot_2.bounds)
print("CPW_Feed_Slot Attribute:", CPW_FEED_SLOT_ATTRIBUTE)

CPW_KEEPOUT_MARGIN = 2  # mm, safety margin kept clear around CPW_Feed_Slot
CPW_Feed_Slot_Keepout_X = CPW_FEED_SLOT_WIDE_WIDTH + CPW_KEEPOUT_MARGIN

MATCHING_STUB_ATTRIBUTE = "Negative/Slot"

# Corners given relative to y=0 at Substrate_Bottom's lowest point -- shift by lower_sub_min_y.
Matching_Stub1 = box(-3.5, 5.5 + lower_sub_min_y, 3.5, 6.5 + lower_sub_min_y)
Matching_Stub2 = box(-3, 8 + lower_sub_min_y, 3, 8.9 + lower_sub_min_y)

print("Matching_Stub1:", Matching_Stub1)
print("Matching_Stub1 Bounds:", Matching_Stub1.bounds)
print("Matching_Stub2:", Matching_Stub2)
print("Matching_Stub2 Bounds:", Matching_Stub2.bounds)
print("Matching_Stub Attribute:", MATCHING_STUB_ATTRIBUTE)

CPW_FEED_PIN_BASE_WIDTH = 0.5  # defensive: no optimizable use yet
CPW_FEED_PIN_WIDE_WIDTH = 1.375  # defensive: no optimizable use yet
CPW_FEED_PIN_CHAMFER_HEIGHT = 0.3  # defensive: no optimizable use yet
CPW_FEED_PIN_ATTRIBUTE = "Positive"

# Same (0, 0)-origin convention as CPW_Feed_Slot_1, but this pin reaches all the way up to
# slot_max_y (main_slot's top edge), crossing over the main slot.
CPW_Feed_Pin_1 = Polygon([
    (0, lower_sub_min_y),
    (CPW_FEED_PIN_BASE_WIDTH, lower_sub_min_y),
    (CPW_FEED_PIN_WIDE_WIDTH, lower_sub_min_y + CPW_FEED_PIN_CHAMFER_HEIGHT),
    (CPW_FEED_PIN_WIDE_WIDTH, lower_sub_min_y + 11),
    (CPW_FEED_PIN_BASE_WIDTH, lower_sub_min_y + 12),
    (CPW_FEED_PIN_BASE_WIDTH, slot_max_y),
    (0, slot_max_y),
])

# Mirror across the Y axis (x -> -x) onto the left side.
CPW_Feed_Pin_2 = scale(CPW_Feed_Pin_1, xfact=-1, yfact=1, origin=(0, 0))

print("CPW_Feed_Pin_1:", CPW_Feed_Pin_1)
print("CPW_Feed_Pin_1 Bounds:", CPW_Feed_Pin_1.bounds)
print("CPW_Feed_Pin_2:", CPW_Feed_Pin_2)
print("CPW_Feed_Pin_2 Bounds:", CPW_Feed_Pin_2.bounds)
print("CPW_Feed_Pin Attribute:", CPW_FEED_PIN_ATTRIBUTE)


def _boundary_hit_ys(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom.y]
    if geom.geom_type == "LineString":
        return [c[1] for c in geom.coords]
    if hasattr(geom, "geoms"):
        ys = []
        for g in geom.geoms:
            ys.extend(_boundary_hit_ys(g))
        return ys
    return []


def ray_distance_to_upper_substrate(x, base_y=None, shape=None):
    """Shoot a ray straight up from (x, base_y) and return the distance to
    the first point where it hits shape's boundary, or None if it misses."""
    if base_y is None:
        base_y = slot_max_y
    if shape is None:
        shape = Upper_Substrate
    ray_top = shape.bounds[3] + 10
    ray = LineString([(x, base_y), (x, ray_top)])
    hit_ys = [y for y in _boundary_hit_ys(ray.intersection(shape.boundary)) if y > base_y + 1e-9]
    if not hit_ys:
        return None
    return min(hit_ys) - base_y


def ray_distance_to_lower_substrate(x, base_y=None, shape=None):
    """Shoot a ray straight down from (x, base_y) and return the distance to
    the first point where it hits shape's boundary, or None if it misses."""
    if base_y is None:
        base_y = slot_min_y
    if shape is None:
        shape = Lower_Substrate
    ray_bottom = shape.bounds[1] - 10
    ray = LineString([(x, base_y), (x, ray_bottom)])
    hit_ys = [y for y in _boundary_hit_ys(ray.intersection(shape.boundary)) if y < base_y - 1e-9]
    if not hit_ys:
        return None
    return base_y - max(hit_ys)


def build_branch_up(k1, k2, k3, x_low=2, x_high=None, near_x=1, upper_substrate=None):
    """Build one upward branch rooted on slot_main's top edge, plus its mirror across x=0.

    k1: position of the branch's midpoint within [x_low, x_high]     -- optimizable, [0,1]
    k2: half-width of the branch as a fraction of the max available width -- optimizable, [0,1]
    k3: length of the branch as a fraction of the max reachable length    -- optimizable, [0,1]

    Returns a dict with the midpoint, endpoints, max_width, max_length, length,
    and the two boxes (branch, and its mirror across x=0).
    """
    if x_high is None:
        x_high = slot_max_x - FIXED_OFFSET
    if upper_substrate is None:
        upper_substrate = Upper_Substrate

    midpoint_x = k1 * (x_high - x_low) + x_low
    midpoint = Point(midpoint_x, slot_max_y)

    # Closer of the two distances (to slot_max_x, and to near_x) caps how far the
    # branch's endpoints below can reach.
    max_width = min(slot_max_x - midpoint_x, midpoint_x - near_x)

    left_endpoint = Point(midpoint_x - k2 * max_width, slot_max_y)
    right_endpoint = Point(midpoint_x + k2 * max_width, slot_max_y)

    left_reach = ray_distance_to_upper_substrate(left_endpoint.x, shape=upper_substrate)
    right_reach = ray_distance_to_upper_substrate(right_endpoint.x, shape=upper_substrate)
    assert left_reach is not None and right_reach is not None
    max_length = min(left_reach, right_reach)

    length = k3 * max_length

    branch = box(left_endpoint.x, slot_max_y, right_endpoint.x, slot_max_y + length)
    branch_mirror = scale(branch, xfact=-1, yfact=1, origin=(0, 0))

    return {
        "midpoint": midpoint,
        "left_endpoint": left_endpoint,
        "right_endpoint": right_endpoint,
        "max_width": max_width,
        "max_length": max_length,
        "length": length,
        "branch": branch,
        "branch_mirror": branch_mirror,
    }


def build_branch_down(k1, k2, k3, x_low=None, x_high=None, near_x=None, lower_substrate=None):
    """Build one downward branch rooted on slot_main's bottom edge, plus its mirror across x=0.

    Same idea as build_branch_up, mirrored to the bottom edge. x_low/near_x default to
    CPW_Feed_Slot_Keepout_X so branches can never be positioned on top of, or reach back
    into, the CPW feed slot's safety margin.
    """
    if x_low is None:
        x_low = CPW_Feed_Slot_Keepout_X
    if x_high is None:
        x_high = slot_max_x - FIXED_OFFSET
    if near_x is None:
        near_x = CPW_Feed_Slot_Keepout_X
    if lower_substrate is None:
        lower_substrate = Lower_Substrate

    midpoint_x = k1 * (x_high - x_low) + x_low
    midpoint = Point(midpoint_x, slot_min_y)

    max_width = min(slot_max_x - midpoint_x, midpoint_x - near_x)

    left_endpoint = Point(midpoint_x - k2 * max_width, slot_min_y)
    right_endpoint = Point(midpoint_x + k2 * max_width, slot_min_y)

    left_reach = ray_distance_to_lower_substrate(left_endpoint.x, shape=lower_substrate)
    right_reach = ray_distance_to_lower_substrate(right_endpoint.x, shape=lower_substrate)
    assert left_reach is not None and right_reach is not None
    max_length = min(left_reach, right_reach)

    length = k3 * max_length

    branch = box(left_endpoint.x, slot_min_y - length, right_endpoint.x, slot_min_y)
    branch_mirror = scale(branch, xfact=-1, yfact=1, origin=(0, 0))

    return {
        "midpoint": midpoint,
        "left_endpoint": left_endpoint,
        "right_endpoint": right_endpoint,
        "max_width": max_width,
        "max_length": max_length,
        "length": length,
        "branch": branch,
        "branch_mirror": branch_mirror,
    }


BRANCH_UP_1_K = 0.5  # optimizable
BRANCH_UP_1_K2 = 0.5  # optimizable
BRANCH_UP_1_K3 = 0.5  # optimizable

branch_up_1 = build_branch_up(BRANCH_UP_1_K, BRANCH_UP_1_K2, BRANCH_UP_1_K3)
Branch_Up_1_Midpoint = branch_up_1["midpoint"]
Branch_Up_1_Left_Endpoint = branch_up_1["left_endpoint"]
Branch_Up_1_Right_Endpoint = branch_up_1["right_endpoint"]
Branch_Up_1_Max_Width = branch_up_1["max_width"]
Branch_Up_1_Max_Length = branch_up_1["max_length"]
Branch_Up_1 = branch_up_1["branch"]
Branch_Up_2 = branch_up_1["branch_mirror"]

print("Branch_Up_1_Midpoint:", Branch_Up_1_Midpoint)
print("Branch_Up_1_Max_Width:", Branch_Up_1_Max_Width)
print("Branch_Up_1_Left_Endpoint:", Branch_Up_1_Left_Endpoint)
print("Branch_Up_1_Right_Endpoint:", Branch_Up_1_Right_Endpoint)
print("Branch_Up_1_Max_Length:", Branch_Up_1_Max_Length)
print("Branch_Up_1:", Branch_Up_1)
print("Branch_Up_1 Bounds:", Branch_Up_1.bounds)
print("Branch_Up_2:", Branch_Up_2)
print("Branch_Up_2 Bounds:", Branch_Up_2.bounds)

BRANCH_DOWN_1_K = 0.5  # optimizable
BRANCH_DOWN_1_K2 = 0.5  # optimizable
BRANCH_DOWN_1_K3 = 0  # optimizable

branch_down_1 = build_branch_down(BRANCH_DOWN_1_K, BRANCH_DOWN_1_K2, BRANCH_DOWN_1_K3)
Branch_Down_1_Midpoint = branch_down_1["midpoint"]
Branch_Down_1_Left_Endpoint = branch_down_1["left_endpoint"]
Branch_Down_1_Right_Endpoint = branch_down_1["right_endpoint"]
Branch_Down_1_Max_Width = branch_down_1["max_width"]
Branch_Down_1_Max_Length = branch_down_1["max_length"]
Branch_Down_1 = branch_down_1["branch"]
Branch_Down_2 = branch_down_1["branch_mirror"]

print("Branch_Down_1_Midpoint:", Branch_Down_1_Midpoint)
print("Branch_Down_1_Max_Width:", Branch_Down_1_Max_Width)
print("Branch_Down_1_Left_Endpoint:", Branch_Down_1_Left_Endpoint)
print("Branch_Down_1_Right_Endpoint:", Branch_Down_1_Right_Endpoint)
print("Branch_Down_1_Max_Length:", Branch_Down_1_Max_Length)
print("Branch_Down_1:", Branch_Down_1)
print("Branch_Down_1 Bounds:", Branch_Down_1.bounds)
print("Branch_Down_2:", Branch_Down_2)
print("Branch_Down_2 Bounds:", Branch_Down_2.bounds)


def merge_shapes(shapes):
    """Union any number of shapes into one combined polygon. Add more shapes to the
    list to fold in future pieces (branches, patches, ...)."""
    return unary_union(list(shapes))


Slot = merge_shapes([
    slot_main,
    Branch_Up_1, Branch_Up_2,
    Branch_Down_1, Branch_Down_2,
    CPW_Feed_Slot_1, CPW_Feed_Slot_2,
    Matching_Stub1, Matching_Stub2,
])

print("Slot:", Slot)
print("Slot Bounds:", Slot.bounds)

Patch = merge_shapes([Upper_Substrate, Lower_Substrate])

print("Patch:", Patch)
print("Patch Bounds:", Patch.bounds)

CPW_Feed_Pin = merge_shapes([CPW_Feed_Pin_1, CPW_Feed_Pin_2])

print("CPW_Feed_Pin:", CPW_Feed_Pin)
print("CPW_Feed_Pin Bounds:", CPW_Feed_Pin.bounds)

QUANTIZE_STEP = 0.01

FINAL_POLYGONS = {
    "Slot": Slot,
    "Patch": Patch,
    "CPW_Feed_Pin": CPW_Feed_Pin,
}

# 1) CCW vertices, 2) quantize to 0.01
quantized_vertices = {}
for name, poly in FINAL_POLYGONS.items():
    ccw_poly = orient(poly, sign=1.0)
    coords = list(ccw_poly.exterior.coords)[:-1]  # drop the closing duplicate point
    quantized_vertices[name] = [
        (round(x / QUANTIZE_STEP) * QUANTIZE_STEP, round(y / QUANTIZE_STEP) * QUANTIZE_STEP)
        for x, y in coords
    ]

# 3) global min Y across all three, 4) shift everything up so min Y = 0
global_min_y = min(y for pts in quantized_vertices.values() for _, y in pts)
shifted_vertices = {
    name: [(x, round(y - global_min_y, 2)) for x, y in pts]
    for name, pts in quantized_vertices.items()
}

# 5) non-self-intersection check
self_intersection_check = {}
for name, pts in shifted_vertices.items():
    ring = LinearRing(pts + [pts[0]])
    check_poly = Polygon(pts)
    self_intersection_check[name] = {
        "ring_is_simple": ring.is_simple,
        "polygon_is_valid": check_poly.is_valid,
    }
    print(f"{name} self-intersection check:", self_intersection_check[name])

# 6) export
EXPORT_PATH = Path(__file__).resolve().parent / "antenna_polygon_vertices.json"
EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(EXPORT_PATH, "w") as f:
    json.dump({
        "meta": {
            "quantize_step": QUANTIZE_STEP,
            "global_min_y_before_shift": global_min_y,
            "self_intersection_check": self_intersection_check,
        },
        "vertices": shifted_vertices,
    }, f, indent=2)

print("Exported vertices to:", EXPORT_PATH)


def plot_geom(ax, geom, style):
    style = dict(style)
    fill = style.pop("fill", True)
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        x, y = poly.exterior.xy
        ax.plot(x, y, color=style.get("color"))
        if fill:
            ax.fill(x, y, **style)


Substrate_Full = box(
    min(Substrate_Top.bounds[0], Substrate_Bottom.bounds[0]),
    min(Substrate_Top.bounds[1], Substrate_Bottom.bounds[1]),
    max(Substrate_Top.bounds[2], Substrate_Bottom.bounds[2]),
    max(Substrate_Top.bounds[3], Substrate_Bottom.bounds[3]),
)

fig, ax = plt.subplots()
shapes = (
    (Patch, dict(alpha=0.3, color="tab:green")),
    (Substrate_Full, dict(fill=False, color="black")),
    (Slot, dict(alpha=0.6, color="tab:blue")),
    (CPW_Feed_Pin, dict(alpha=0.5, color="tab:cyan")),
)
for shape, style in shapes:
    plot_geom(ax, shape, style)

# Three virtual 0-width parts -- drawn as annotations only, no polygon is actually cut/joined:
# 1) a slit at Pin's two highest points, separating Pin from Patch
# 2) a connection across Slot's two lowest points, sealing Patch into a proper annulus
# 3) a slit at Pin's two lowest points, working with (1) to also seal Slot into a proper annulus
VIRTUAL_SLIT_STYLE = dict(color="magenta", linewidth=3, linestyle=(0, (2, 2)))
VIRTUAL_CONNECTION_STYLE = dict(color="orange", linewidth=3)

ax.plot(
    [-CPW_FEED_PIN_BASE_WIDTH, CPW_FEED_PIN_BASE_WIDTH], [slot_max_y, slot_max_y],
    **VIRTUAL_SLIT_STYLE, label="virtual slit (Pin ↔ Patch)",
)
ax.plot(
    [-CPW_FEED_SLOT_WIDE_WIDTH, CPW_FEED_SLOT_WIDE_WIDTH], [lower_sub_min_y, lower_sub_min_y],
    **VIRTUAL_CONNECTION_STYLE, label="virtual connection (seals Patch)",
)
ax.plot(
    [-CPW_FEED_PIN_BASE_WIDTH, CPW_FEED_PIN_BASE_WIDTH], [lower_sub_min_y, lower_sub_min_y],
    **VIRTUAL_SLIT_STYLE, label="virtual slit (seals Slot)",
)

ax.legend(loc="upper right")
ax.set_aspect("equal")
ax.set_title("MSA-BP quasi-topological decomposition (annotated)")

plt.show()
