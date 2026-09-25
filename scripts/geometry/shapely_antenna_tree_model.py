"""Headless arbitrary-tree MSA-BP geometry extracted from the interactive demo.

Only Shapely is required. Importing or building performs no plotting, GUI setup,
printing, file writes, CST calls, or solver work. Geometry formulas and branch
semantics derive from the G5 implementation at b503ff1, with optional K2 saturation.

Use default_params(), default_tree(), then build(params, tree). The tree is an
insertion-ordered dictionary with parents before children; each node records
parent, side (U/D/L/R), and three K values. Mirroring is (x,y)->(-x,y).
The model retains the demo's assumptions rather than adding a new validator.
"""

from __future__ import annotations

from math import isfinite

from shapely.affinity import scale
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Fixed (non-optimizable) constants
# ---------------------------------------------------------------------------

FIXED_OFFSET = 1  # clearance that always remains even if the optimizable margins go to 0
PATCH_BRICK_2_WIDTH = 12
PATCH_BRICK_4_FIXED = 13

SMA_GND_PAD_X_LOW = 3.49
SMA_GND_PAD_X_HIGH = 4.76
SMA_GND_PAD_HEIGHT = 4.5

CPW_FEED_SLOT_WIDE_WIDTH = 2.4
CPW_FEED_SLOT_NARROW_WIDTH = 1.7
CPW_KEEPOUT_MARGIN = 2

CPW_FEED_PIN_BASE_WIDTH = 0.5
CPW_FEED_PIN_WIDE_WIDTH = 1.375
CPW_FEED_PIN_CHAMFER_HEIGHT = 0.3

# ---------------------------------------------------------------------------
# Parameter table: group -> [(name, kind, default), ...]
# kind "abs" = absolute mm value, kind "k" = ratio in [0, 1].
# The K defaults are the ratios that reproduce the original hand-tuned mm
# dimensions when every absolute parameter sits at its own default.
# ---------------------------------------------------------------------------

K_RANGE = (0.05, 1.0)
ABS_RANGE_FACTORS = (0.6, 1.4)
BRANCH_K2_SNAP_FRACTION = 0.10


def relax_upper_k(k, snap_fraction=BRANCH_K2_SNAP_FRACTION):
    """Identity below 1-2s, slope two up to 1-s, then exactly one."""
    if not isfinite(k) or not 0.0 <= k <= 1.0:
        raise ValueError("K must be finite and in [0, 1]")
    if not isfinite(snap_fraction) or not 0.0 <= snap_fraction <= 0.5:
        raise ValueError("snap_fraction must be finite and in [0, 0.5]")
    if snap_fraction == 0.0 or k <= 1.0 - 2.0 * snap_fraction:
        return float(k)
    if k >= 1.0 - snap_fraction:
        return 1.0
    return 2.0 * k - (1.0 - 2.0 * snap_fraction)

PARAM_GROUPS = {
    "1. Main Slot": [
        ("SLOT_MAIN_LENGTH", "abs", 53),
        ("SLOT_MAIN_HEIGHT", "abs", 2),
    ],
    "2. Patch Bricks": [
        ("PATCH_BRICK_1_SIDE_MARGIN", "abs", 6),
        ("PATCH_BRICK_1_TOP_MARGIN", "abs", 2.6),
        ("PATCH_BRICK_2_HEIGHT_MARGIN", "abs", 15),
        ("PATCH_BRICK_3_BOTTOM_MARGIN", "abs", 2),
        ("PATCH_BRICK_4_MARGIN", "abs", 4),
    ],
    "3. Upper Corner": [
        ("UPPER_CORNER_NOTCH_1_K1", "k", 17 / 27.5),
        ("UPPER_CORNER_NOTCH_1_K2", "k", 14 / 15),
        ("UPPER_CORNER_EAR_1_K1", "k", 7 / 17),
        ("UPPER_CORNER_EAR_1_K2", "k", 1 / 14),
    ],
    "4. Lower Corner": [
        ("LOWER_CORNER_NOTCH_1_K1", "k", 21.3 / 27.5),
        ("LOWER_CORNER_NOTCH_1_K2", "k", 12 / 17),
        ("LOWER_CORNER_EAR_1_K1", "k", 5 / 21.3),
        ("LOWER_CORNER_EAR_1_K2", "k", 4 / 6),
        ("LOWER_CORNER_EAR_2_K1", "k", 4 / 16.3),
        ("LOWER_CORNER_EAR_2_K2", "k", 1.5 / 6),
    ],
}

PARAM_SPECS = {
    name: (kind, default)
    for entries in PARAM_GROUPS.values()
    for name, kind, default in entries
}


def param_range(name):
    kind, default = PARAM_SPECS[name]
    if kind == "k":
        return K_RANGE
    return (default * ABS_RANGE_FACTORS[0], default * ABS_RANGE_FACTORS[1])


def default_params():
    return {name: default for name, (_, default) in PARAM_SPECS.items()}


# ---------------------------------------------------------------------------
# Branch tree
#
# The main slot is the root. Every branch hangs off one face of its parent and
# grows perpendicular to it: the slot and horizontal branches offer U/D faces,
# vertical branches offer L/R faces. Each branch owns [K1, K2, K3]. Node ids are
# paths of "<side><n>" steps, e.g. "U1/R2/U1", so a subtree is an id prefix.
# The tree lives in a dict ordered parents-before-children:
#   {node_id: {"parent": parent_id, "side": "U" | "D" | "L" | "R", "k": [k1, k2, k3]}}
# ---------------------------------------------------------------------------

ROOT = "SLOT"
SIDE_DIRECTION = {"U": "up", "D": "down", "L": "left", "R": "right"}
NEW_BRANCH_K = (0.5, 0.5, 0.5)


def default_tree():
    """The two branches of the original hand-tuned design."""
    return {
        "U1": {"parent": ROOT, "side": "U", "k": [0.5, 0.5, 0.5]},
        "D1": {"parent": ROOT, "side": "D", "k": [0.5, 0.5, 0.05]},
    }


def child_sides(tree, node_id):
    if node_id == ROOT or tree[node_id]["side"] in ("L", "R"):
        return ("U", "D")
    return ("L", "R")


def add_branch(tree, parent, side, k=NEW_BRANCH_K):
    prefix = "" if parent == ROOT else f"{parent}/"
    n = 1
    while f"{prefix}{side}{n}" in tree:
        n += 1
    node_id = f"{prefix}{side}{n}"
    tree[node_id] = {"parent": parent, "side": side, "k": list(k)}
    return node_id


def delete_subtree(tree, node_id):
    for key in [k for k in tree if k == node_id or k.startswith(f"{node_id}/")]:
        del tree[key]


def k_range(tree, node_id, index):
    # Only branches on the main slot may shrink to zero width; their children
    # then hang off the zero-width centre line.
    if index == 1 and tree[node_id]["parent"] == ROOT:
        return (0.0, K_RANGE[1])
    return K_RANGE


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _polygons_only(geom):
    """At extreme slider settings a notch can eat a whole corner, leaving the union with
    stray touching lines/points; those make .boundary None. Keep the area parts only."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    return unary_union([g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")])


DIRECTIONS = {"up": (0, 1), "down": (0, -1), "right": (1, 0), "left": (-1, 0)}


def strip_reach(lo, hi, base, shape, direction):
    """How far a strip spanning [lo, hi] across its growth axis can grow from `base`
    along `direction` before any part of it leaves shape. Checking the whole strip, not
    just rays at its two edges, catches boundary dips that lie strictly between the
    edges (e.g. a gap between lower ears). A zero-width strip degrades to one ray."""
    dx, dy = DIRECTIONS[direction]
    min_x, min_y, max_x, max_y = shape.bounds
    far = {"up": max_y, "down": min_y, "right": max_x, "left": min_x}[direction] + 10 * (dx + dy)
    a, b = min(base, far), max(base, far)
    vertical = dx == 0

    if hi - lo > 1e-12:
        strip = box(lo, a, hi, b) if vertical else box(a, lo, b, hi)
        size_attribute = "area"
    else:
        strip = LineString([(lo, a), (lo, b)] if vertical else [(a, lo), (b, lo)])
        size_attribute = "length"
    outside = strip.difference(shape)
    parts = [g for g in getattr(outside, "geoms", [outside]) if not g.is_empty]
    parts = [g for g in parts if getattr(g, size_attribute) > 1e-9]  # drop float slivers along shared edges
    if not parts:
        return abs(far - base)

    axis = 1 if vertical else 0  # index into bounds: (min_x, min_y, max_x, max_y)
    if dx + dy > 0:
        return min(g.bounds[axis] for g in parts) - base
    return base - max(g.bounds[axis + 2] for g in parts)


def _branch_info(k, direction, k1_span, midpoint, max_width, half_width, max_length, length,
                 *, width_bounds=None, effective_k2=None):
    """Shared shaper bookkeeping for level-1 and level-2 branches, in (x, y) tuples.

    K1: midpoint slides along k1_span. Half-width = effective K2 * max_width, measured across
    the growth axis. K3: length = K3 * max_length along `direction`, where max_length is
    how far the whole strip can grow before leaving the metal patch."""
    dx, dy = DIRECTIONS[direction]
    px, py = abs(dy), abs(dx)  # unit vector across the growth axis
    mx, my = midpoint
    ends = ((mx - px * half_width, my - py * half_width), (mx + px * half_width, my + py * half_width))
    if width_bounds is not None:
        low, high = width_bounds
        ends = ((low, my), (high, my)) if px else ((mx, low), (mx, high))
    tip = (mx + dx * length, my + dy * length)
    xs = [ends[0][0], ends[1][0], ends[0][0] + dx * length, ends[1][0] + dx * length]
    ys = [ends[0][1], ends[1][1], ends[0][1] + dy * length, ends[1][1] + dy * length]
    branch = box(min(xs), min(ys), max(xs), max(ys))
    return {
        "k": k,
        "effective_k2": k[1] if effective_k2 is None else effective_k2,
        "direction": direction,
        "k1_span": k1_span,
        "midpoint": midpoint,
        "max_width": max_width,
        "endpoints": ends,
        "max_length": max_length,
        "length": length,
        "tip": tip,
        "branch": branch,
        "branch_mirror": scale(branch, xfact=-1, yfact=1, origin=(0, 0)),
    }


def _build_branch(k1, k2, k3, face, direction, patch,
                  *, snap_fraction=BRANCH_K2_SNAP_FRACTION):
    """Grow one branch off a parent face.

    face = {"base": coordinate of the face along the growth axis,
            "span": (a, b) the K1 midpoint slides from a to b across the growth axis,
            "cap":  (a, b) face ends the branch width may not cross}.
    Every "left" branch is also capped at the Y axis: past it a branch only
    overlaps its own mirror image, so the geometry would stop changing while K3
    kept moving. That keeps the whole tree in x >= 0 before mirroring."""
    (a, b), (c0, c1), base = face["span"], face["cap"], face["base"]
    mid = a + k1 * (b - a)
    max_width = max(0.0, min(abs(mid - c0), abs(c1 - mid)))
    effective_k2 = relax_upper_k(k2, snap_fraction)
    half_width = effective_k2 * max_width
    low, high = mid - half_width, mid + half_width
    if effective_k2 == 1.0:
        cap_low, cap_high = sorted((c0, c1))
        if abs(mid - cap_low) <= abs(cap_high - mid):
            low = cap_low
        if abs(cap_high - mid) <= abs(mid - cap_low):
            high = cap_high
        # Reversed spans can round their midpoint toward one end. At the
        # exact halfway parameter both caps are selected, not just the nearer.
        if k1 == 0.5 and sorted((a, b)) == [cap_low, cap_high]:
            low, high = cap_low, cap_high

    max_length = max(0.0, strip_reach(low, high, base, patch, direction))
    if direction == "left":
        max_length = min(max_length, max(0.0, base))

    vertical = direction in ("up", "down")
    midpoint = (mid, base) if vertical else (base, mid)
    k1_span = ((a, base), (b, base)) if vertical else ((base, a), (base, b))
    return _branch_info(
        (k1, k2, k3), direction, k1_span, midpoint, max_width, half_width, max_length, k3 * max_length,
        width_bounds=(low, high), effective_k2=effective_k2,
    )


def _branch_faces(info):
    """Faces a built branch offers its children. They run from the branch base to its
    tip, so a child's K1 = 0 sits at the parent's base and K1 = 1 at its tip."""
    min_x, min_y, max_x, max_y = info["branch"].bounds
    (mx, my), (tx, ty) = info["midpoint"], info["tip"]
    if info["direction"] in ("up", "down"):
        run = (my, ty)
        return {"L": {"base": min_x, "span": run, "cap": run},
                "R": {"base": max_x, "span": run, "cap": run}}
    run = (mx, tx)
    return {"U": {"base": max_y, "span": run, "cap": run},
            "D": {"base": min_y, "span": run, "cap": run}}


def build(params, tree, *, snap_fraction=BRANCH_K2_SNAP_FRACTION):
    """Build every polygon of the antenna from the shape parameters and the branch tree."""
    relax_upper_k(0.0, snap_fraction)  # Validate even for an empty tree.
    p = params

    slot_len = p["SLOT_MAIN_LENGTH"]
    slot_h = p["SLOT_MAIN_HEIGHT"]
    slot_main = box(-slot_len / 2, -slot_h / 2, slot_len / 2, slot_h / 2)
    slot_min_x, slot_min_y, slot_max_x, slot_max_y = slot_main.bounds

    ext_side = FIXED_OFFSET + p["PATCH_BRICK_1_SIDE_MARGIN"]
    ext_up = FIXED_OFFSET + p["PATCH_BRICK_1_TOP_MARGIN"]
    ext_down_3 = FIXED_OFFSET + p["PATCH_BRICK_3_BOTTOM_MARGIN"]

    Patch_Brick_1 = box(slot_min_x - ext_side, slot_min_y, slot_max_x + ext_side, slot_max_y + ext_up)
    Patch_Brick_3 = box(slot_min_x - ext_side, slot_min_y - ext_down_3, slot_max_x + ext_side, slot_max_y)

    brick_2_height = ext_up + p["PATCH_BRICK_2_HEIGHT_MARGIN"]
    Patch_Brick_2 = box(
        -PATCH_BRICK_2_WIDTH / 2, slot_max_y,
        PATCH_BRICK_2_WIDTH / 2, slot_max_y + brick_2_height,
    )

    brick_4_height = ext_down_3 + p["PATCH_BRICK_4_MARGIN"] + PATCH_BRICK_4_FIXED
    Patch_Brick_4 = box(
        -PATCH_BRICK_2_WIDTH / 2, slot_min_y - brick_4_height,
        PATCH_BRICK_2_WIDTH / 2, slot_min_y,
    )

    # --- upper half -------------------------------------------------------
    upper = (slot_main, Patch_Brick_1, Patch_Brick_2)
    sub_min_x = min(s.bounds[0] for s in upper)
    sub_min_y = min(s.bounds[1] for s in upper)
    sub_max_x = max(s.bounds[2] for s in upper)
    sub_max_y = max(s.bounds[3] for s in upper)
    Substrate_Top = box(sub_min_x, sub_min_y, sub_max_x, sub_max_y)

    substrate_width = sub_max_x - sub_min_x
    corner_span = (substrate_width - PATCH_BRICK_2_WIDTH) / 2

    notch_u_w = -corner_span * p["UPPER_CORNER_NOTCH_1_K1"]
    notch_u_h = -p["PATCH_BRICK_2_HEIGHT_MARGIN"] * p["UPPER_CORNER_NOTCH_1_K2"]
    Upper_Corner_Notch_1 = box(sub_max_x + notch_u_w, sub_max_y + notch_u_h, sub_max_x, sub_max_y)

    x1, y1 = Upper_Corner_Notch_1.bounds[0], Upper_Corner_Notch_1.bounds[1]
    Upper_Corner_Ear_1 = box(
        x1, y1,
        x1 + p["UPPER_CORNER_EAR_1_K1"] * -notch_u_w,
        y1 + p["UPPER_CORNER_EAR_1_K2"] * -notch_u_h,
    )

    Upper_Corner_Notch_2 = scale(Upper_Corner_Notch_1, xfact=-1, yfact=1, origin=(0, 0))
    Upper_Corner_Ear_2 = scale(Upper_Corner_Ear_1, xfact=-1, yfact=1, origin=(0, 0))

    Upper_Substrate = (
        Substrate_Top
        .difference(Upper_Corner_Notch_1)
        .difference(Upper_Corner_Notch_2)
        .union(Upper_Corner_Ear_1)
        .union(Upper_Corner_Ear_2)
    )
    Upper_Substrate = _polygons_only(Upper_Substrate)

    # --- lower half -------------------------------------------------------
    lower = (Patch_Brick_3, Patch_Brick_4)
    low_min_x = min(s.bounds[0] for s in lower)
    low_min_y = min(s.bounds[1] for s in lower)
    low_max_x = max(s.bounds[2] for s in lower)
    low_max_y = max(s.bounds[3] for s in lower)
    Substrate_Bottom = box(low_min_x, low_min_y, low_max_x, low_max_y)

    brick_4_span = p["PATCH_BRICK_4_MARGIN"] + PATCH_BRICK_4_FIXED
    notch_l_w = -corner_span * p["LOWER_CORNER_NOTCH_1_K1"]
    notch_l_h = brick_4_span * p["LOWER_CORNER_NOTCH_1_K2"]
    Lower_Corner_Notch_1 = box(low_max_x + notch_l_w, low_min_y, low_max_x, low_min_y + notch_l_h)

    ear1_w = p["LOWER_CORNER_EAR_1_K1"] * -notch_l_w
    ear1_h = p["LOWER_CORNER_EAR_1_K2"] * (notch_l_h / 2)
    x2, y2 = Lower_Corner_Notch_1.bounds[0], Lower_Corner_Notch_1.bounds[3]
    Lower_Corner_Ear_1 = box(x2, y2 - ear1_h, x2 + ear1_w, y2)

    ear2_x1, ear2_y1 = Lower_Corner_Ear_1.bounds[2], Lower_Corner_Ear_1.bounds[3]
    ear2_x2, ear2_y2 = Lower_Corner_Notch_1.bounds[2], Lower_Corner_Notch_1.bounds[1]
    ear2_w = p["LOWER_CORNER_EAR_2_K1"] * (ear2_x2 - ear2_x1)
    ear2_h = p["LOWER_CORNER_EAR_2_K2"] * (ear2_y1 - ear2_y2) / 2
    Lower_Corner_Ear_2 = box(ear2_x1, ear2_y1 - ear2_h, ear2_x1 + ear2_w, ear2_y1)

    y3 = (Lower_Corner_Notch_1.bounds[1] + Lower_Corner_Notch_1.bounds[3]) / 2
    Lower_Corner_Ear_3 = scale(Lower_Corner_Ear_1, xfact=1, yfact=-1, origin=(0, y3))
    Lower_Corner_Ear_4 = scale(Lower_Corner_Ear_2, xfact=1, yfact=-1, origin=(0, y3))

    right_side = (
        Lower_Corner_Ear_1, Lower_Corner_Ear_2, Lower_Corner_Ear_3, Lower_Corner_Ear_4,
    )
    Lower_Corner_Notch_2 = scale(Lower_Corner_Notch_1, xfact=-1, yfact=1, origin=(0, 0))
    mirrored_ears = [scale(e, xfact=-1, yfact=1, origin=(0, 0)) for e in right_side]

    Lower_Substrate = (
        Substrate_Bottom
        .difference(Lower_Corner_Notch_1)
        .difference(Lower_Corner_Notch_2)
    )
    for ear in (*right_side, *mirrored_ears):
        Lower_Substrate = Lower_Substrate.union(ear)
    Lower_Substrate = _polygons_only(Lower_Substrate)

    # --- feed network -----------------------------------------------------
    SMA_GND_Pad_1 = box(
        SMA_GND_PAD_X_LOW, low_min_y,
        SMA_GND_PAD_X_HIGH, low_min_y + SMA_GND_PAD_HEIGHT,
    )
    SMA_GND_Pad_2 = scale(SMA_GND_Pad_1, xfact=-1, yfact=1, origin=(0, 0))

    CPW_Feed_Slot_1 = Polygon([
        (0, low_min_y),
        (CPW_FEED_SLOT_WIDE_WIDTH, low_min_y),
        (CPW_FEED_SLOT_WIDE_WIDTH, low_min_y + 11),
        (CPW_FEED_SLOT_NARROW_WIDTH, low_min_y + 12),
        (CPW_FEED_SLOT_NARROW_WIDTH, slot_min_y),
        (0, slot_min_y),
    ])
    CPW_Feed_Slot_2 = scale(CPW_Feed_Slot_1, xfact=-1, yfact=1, origin=(0, 0))

    CPW_Feed_Pin_1 = Polygon([
        (0, low_min_y),
        (CPW_FEED_PIN_BASE_WIDTH, low_min_y),
        (CPW_FEED_PIN_WIDE_WIDTH, low_min_y + CPW_FEED_PIN_CHAMFER_HEIGHT),
        (CPW_FEED_PIN_WIDE_WIDTH, low_min_y + 11),
        (CPW_FEED_PIN_BASE_WIDTH, low_min_y + 12),
        (CPW_FEED_PIN_BASE_WIDTH, slot_max_y),
        (0, slot_max_y),
    ])
    CPW_Feed_Pin_2 = scale(CPW_Feed_Pin_1, xfact=-1, yfact=1, origin=(0, 0))

    Matching_Stub1 = box(-3.5, 5.5 + low_min_y, 3.5, 6.5 + low_min_y)
    Matching_Stub2 = box(-3, 8 + low_min_y, 3, 8.9 + low_min_y)

    # --- branches ---------------------------------------------------------
    # Every shaper uses only the metal patch as its boundary: branches may overlap
    # each other or other slots, and may cut off copper islands.
    Patch = unary_union([Upper_Substrate, Lower_Substrate])
    # Main-slot faces: the upper one keeps 1 mm from the Y axis (2 mm gap to the
    # mirror branch); the lower one keeps clear of the CPW feed slot.
    keepout_x = CPW_FEED_SLOT_WIDE_WIDTH + CPW_KEEPOUT_MARGIN
    x_high = slot_max_x - FIXED_OFFSET
    faces = {ROOT: {
        "U": {"base": slot_max_y, "span": (2, x_high), "cap": (1, slot_max_x)},
        "D": {"base": slot_min_y, "span": (keepout_x, x_high), "cap": (keepout_x, slot_max_x)},
    }}

    branches = {}
    for node_id, node in tree.items():  # parents come before children
        face = faces[node["parent"]][node["side"]]
        branches[node_id] = _build_branch(
            *node["k"], face, SIDE_DIRECTION[node["side"]], Patch,
            snap_fraction=snap_fraction,
        )
        faces[node_id] = _branch_faces(branches[node_id])

    branch_polys = [
        g for info in branches.values()
        for g in (info["branch"], info["branch_mirror"])
        if g.area > 0  # zero-width branches contribute nothing
    ]
    Slot = unary_union([
        slot_main,
        *branch_polys,
        CPW_Feed_Slot_1, CPW_Feed_Slot_2,
        Matching_Stub1, Matching_Stub2,
    ])
    CPW_Feed_Pin = unary_union([CPW_Feed_Pin_1, CPW_Feed_Pin_2])

    Substrate_Full = box(
        min(sub_min_x, low_min_x), min(sub_min_y, low_min_y),
        max(sub_max_x, low_max_x), max(sub_max_y, low_max_y),
    )

    return {
        "Substrate_Full": Substrate_Full,
        "Patch": Patch,
        "Slot": Slot,
        "CPW_Feed_Pin": CPW_Feed_Pin,
        "SMA_Pads": (SMA_GND_Pad_1, SMA_GND_Pad_2),
        "slot_main": slot_main,
        "Upper_Substrate": Upper_Substrate,
        "Lower_Substrate": Lower_Substrate,
        "branches": branches,
        "metal_area": Patch.difference(Slot).union(CPW_Feed_Pin).area,
    }
