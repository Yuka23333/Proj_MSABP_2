"""Interactive demo: MSA-BP planar slot antenna with a free-form branch tree.

Variant of shapely_antenna_demo.py whose branches are not pre-embedded: the slot
branches form a tree of any shape and depth, edited live in the GUI.

Shape: group dropdown -> variable dropdown -> slider, covering the non-branch
``# optimizable`` variables of shapely_rectangle_test.py
  - K variables (relative, ratio type) slide over [0.05, 1]
  - absolute variables (mm) slide over [60%, 140%] of their default

Branch tree: the main slot is the root; every branch grows perpendicular off one
face of its parent (vertical branches off horizontal ones and vice versa), shaped
by K1 position / K2 width / K3 length. "Add" creates a child with all K = 0.5,
"Delete" removes a whole subtree. Branches on the main slot may have zero width.
Shapers only treat the metal patch as a boundary (branches growing toward -x also
stop at the Y axis), so branches may overlap each other and may cut off copper
islands. Everything is mirrored about the Y axis.
"""

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
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
SIDE_LABEL = {"U": "+ Up", "D": "+ Down", "L": "+ Left (inward)", "R": "+ Right (outward)"}
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
        size = lambda g: g.area
    else:
        strip = LineString([(lo, a), (lo, b)] if vertical else [(a, lo), (b, lo)])
        size = lambda g: g.length
    outside = strip.difference(shape)
    parts = [g for g in getattr(outside, "geoms", [outside]) if not g.is_empty]
    parts = [g for g in parts if size(g) > 1e-9]  # drop float slivers along shared edges
    if not parts:
        return abs(far - base)

    axis = 1 if vertical else 0  # index into bounds: (min_x, min_y, max_x, max_y)
    if dx + dy > 0:
        return min(g.bounds[axis] for g in parts) - base
    return base - max(g.bounds[axis + 2] for g in parts)


def _branch_info(k, direction, k1_span, midpoint, max_width, half_width, max_length, length):
    """Shared shaper bookkeeping for level-1 and level-2 branches, in (x, y) tuples.

    K1: midpoint slides along k1_span. K2: half-width = K2 * max_width, measured across
    the growth axis. K3: length = K3 * max_length along `direction`, where max_length is
    how far the whole strip can grow before leaving the metal patch."""
    dx, dy = DIRECTIONS[direction]
    px, py = abs(dy), abs(dx)  # unit vector across the growth axis
    mx, my = midpoint
    ends = ((mx - px * half_width, my - py * half_width), (mx + px * half_width, my + py * half_width))
    tip = (mx + dx * length, my + dy * length)
    xs = [ends[0][0], ends[1][0], ends[0][0] + dx * length, ends[1][0] + dx * length]
    ys = [ends[0][1], ends[1][1], ends[0][1] + dy * length, ends[1][1] + dy * length]
    branch = box(min(xs), min(ys), max(xs), max(ys))
    return {
        "k": k,
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


def _build_branch(k1, k2, k3, face, direction, patch):
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
    half_width = k2 * max_width

    max_length = max(0.0, strip_reach(mid - half_width, mid + half_width, base, patch, direction))
    if direction == "left":
        max_length = min(max_length, max(0.0, base))

    vertical = direction in ("up", "down")
    midpoint = (mid, base) if vertical else (base, mid)
    k1_span = ((a, base), (b, base)) if vertical else ((base, a), (base, b))
    return _branch_info(
        (k1, k2, k3), direction, k1_span, midpoint, max_width, half_width, max_length, k3 * max_length,
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


def build(params, tree):
    """Build every polygon of the antenna from the shape parameters and the branch tree."""
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
        branches[node_id] = _build_branch(*node["k"], face, SIDE_DIRECTION[node["side"]], Patch)
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


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


def _poly_patch(poly, **kwargs):
    vertices, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        coords = list(ring.coords)
        vertices.extend(coords)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(vertices, codes), **kwargs)


def draw_geom(ax, geom, **kwargs):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        if not poly.is_empty:
            ax.add_patch(_poly_patch(poly, **kwargs))


# Opaque gold with a dark edge: the feed pin sits inside the blue feed slot, and a
# translucent blue-ish fill there blends into it.
FEED_PIN_STYLE = dict(facecolor="gold", edgecolor="darkgoldenrod", lw=1.2, alpha=0.95, zorder=4)


# One colour per shaper stage: K1 = position, K2 = width, K3 = length.
K_STYLE = {
    "K1": ("o", "tab:red", "K1 position (dashed: allowed span)"),
    "K2": ("s", "tab:orange", "K2 width (dotted: max width)"),
    "K3": ("^", "tab:purple", "K3 length (dotted: max length)"),
}


def annotate_branch(ax, info, fontsize=8):
    """Mark one branch's K1 midpoint, K2 endpoints and K3 tip, each with its value
    and a guide showing the cap that stage was scaled against."""
    k1, k2, k3 = info["k"]
    dx, dy = DIRECTIONS[info["direction"]]
    px, py = abs(dy), abs(dx)  # across the growth axis
    (mx, my), tip = info["midpoint"], info["tip"]
    text_kw = dict(
        fontsize=fontsize, textcoords="offset points", va="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"), zorder=7,
    )

    m1, c1, _ = K_STYLE["K1"]
    (sx0, sy0), (sx1, sy1) = info["k1_span"]
    ax.plot([sx0, sx1], [sy0, sy1], ls="--", lw=0.9, color=c1, alpha=0.8)
    ax.plot(mx, my, m1, color=c1, ms=5, mec="white", zorder=6)
    ax.annotate(f"K1={k1:.2f}", (mx, my), xytext=(-16 * px + 8 * dx, -16 * py + 8 * dy),
                ha="right" if px else "center", color=c1, **text_kw)

    m2, c2, _ = K_STYLE["K2"]
    w, off = info["max_width"], 0.35
    ax.plot([mx - px * w + dx * off, mx + px * w + dx * off],
            [my - py * w + dy * off, my + py * w + dy * off], ls=":", lw=0.9, color=c2, alpha=0.9)
    for ex, ey in info["endpoints"]:
        ax.plot(ex, ey, m2, color=c2, ms=4.5, mec="white", zorder=6)
    ax.annotate(f"K2={k2:.2f}", info["endpoints"][1], xytext=(16 * px + 8 * dx, 16 * py + 8 * dy),
                ha="left" if px else "center", color=c2, **text_kw)

    m3, c3, _ = K_STYLE["K3"]
    cap = (mx + dx * info["max_length"], my + dy * info["max_length"])
    ax.plot([tip[0], cap[0]], [tip[1], cap[1]], ls=":", lw=0.9, color=c3, alpha=0.9)
    ax.plot(*cap, "|" if px == 0 else "_", color=c3, ms=8, mew=1.2)
    ax.plot(*tip, m3, color=c3, ms=5, mec="white", zorder=6)
    ax.annotate(f"K3={k3:.2f}", tip, xytext=(18 * dx, 18 * dy),
                ha="center" if px else ("left" if dx > 0 else "right"), color=c3, **text_kw)


def k_legend_handles():
    from matplotlib.lines import Line2D
    return [
        Line2D([], [], marker=m, color=c, ls="none", ms=5, label=label)
        for m, c, label in K_STYLE.values()
    ]


# Fixed character width for labels whose text changes, so the left column never
# resizes (and drags the plot with it) while a slider moves.
LABEL_WIDTH = 40
K_LABEL_WIDTH = 29  # fits "K1 position = 0.500 [0.05, 1]" beside its slider


class AntennaDemo:
    def __init__(self, root):
        self.root = root
        self.params = default_params()
        self.tree_nodes = default_tree()
        self.view_limits = self._compute_view_limits()

        root.title("MSA-BP antenna branch-tree explorer")

        controls = ttk.Frame(root, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        # --- shape parameters: group -> variable -> slider -------------------
        ttk.Label(controls, text="Variable group").pack(anchor="w")
        self.group_var = tk.StringVar(value=next(iter(PARAM_GROUPS)))
        self.group_box = ttk.Combobox(
            controls, textvariable=self.group_var, state="readonly",
            values=list(PARAM_GROUPS), width=28,
        )
        self.group_box.pack(anchor="w", pady=(0, 8))
        self.group_box.bind("<<ComboboxSelected>>", self.on_group_change)

        ttk.Label(controls, text="Variable").pack(anchor="w")
        self.name_var = tk.StringVar()
        self.name_box = ttk.Combobox(controls, textvariable=self.name_var, state="readonly", width=28)
        self.name_box.pack(anchor="w", pady=(0, 8))
        self.name_box.bind("<<ComboboxSelected>>", self.on_name_change)

        self.value_label = ttk.Label(controls, text="", font=("TkDefaultFont", 10, "bold"), width=LABEL_WIDTH)
        self.value_label.pack(anchor="w")

        self.slider_var = tk.DoubleVar()
        self.slider = ttk.Scale(
            controls, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.slider_var, command=self.on_slide, length=240,
        )
        self.slider.pack(anchor="w")

        self.range_label = ttk.Label(controls, text="", width=LABEL_WIDTH)
        self.range_label.pack(anchor="w", pady=(0, 8))

        ttk.Button(controls, text="Reset this variable", command=self.reset_current).pack(
            anchor="w", fill=tk.X
        )
        ttk.Button(controls, text="Reset all (shape + tree)", command=self.reset_all).pack(
            anchor="w", fill=tk.X, pady=(4, 8)
        )

        # --- branch tree ----------------------------------------------------
        ttk.Label(controls, text="Branch tree").pack(anchor="w")
        tree_frame = ttk.Frame(controls)
        tree_frame.pack(anchor="w", fill=tk.X)
        self.tree = ttk.Treeview(tree_frame, columns=("grows",), height=8, selectmode="browse")
        self.tree.heading("#0", text="Branch")
        self.tree.heading("grows", text="Grows")
        self.tree.column("#0", width=160)
        self.tree.column("grows", width=60)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Delete>", lambda _event: self.delete_selected())

        tree_buttons = ttk.Frame(controls)
        tree_buttons.pack(anchor="w", fill=tk.X, pady=(4, 0))
        self.add_buttons = [
            ttk.Button(tree_buttons, command=lambda i=i: self.add_child(i)) for i in range(2)
        ]
        for i, button in enumerate(self.add_buttons):
            button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4 if i else 0, 0))
        self.delete_button = ttk.Button(controls, text="Delete subtree", command=self.delete_selected)
        self.delete_button.pack(anchor="w", fill=tk.X, pady=(4, 8))

        # --- K sliders of the selected branch -------------------------------
        self.k_frame = ttk.LabelFrame(controls, text="", padding=4)
        self.k_frame.pack(anchor="w", fill=tk.X, pady=(0, 8))
        self.k_vars, self.k_scales, self.k_labels = [], [], []
        for i in range(3):
            label = ttk.Label(self.k_frame, text="", width=K_LABEL_WIDTH)
            label.grid(row=i, column=0, sticky="w")
            var = tk.DoubleVar()
            scale_widget = ttk.Scale(
                self.k_frame, orient=tk.HORIZONTAL, variable=var, length=130,
                command=lambda _value, i=i: self.on_k_slide(i),
            )
            scale_widget.grid(row=i, column=1, sticky="ew", pady=1)
            self.k_vars.append(var)
            self.k_scales.append(scale_widget)
            self.k_labels.append(label)

        self.status_label = ttk.Label(controls, text="", justify="left", width=LABEL_WIDTH)
        self.status_label.pack(anchor="w")

        # The plot side gets its own frame with the toolbar pinned to its bottom.
        # Packed straight onto root, the toolbar used to claim leftover space between
        # the two columns, so every change in the cursor-coordinate text re-split them.
        plot_side = ttk.Frame(root)
        plot_side.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(7.5, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_side)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_side, pack_toolbar=False)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        message_label = getattr(toolbar, "_message_label", None)  # private in matplotlib
        if message_label is not None:
            message_label.configure(width=32, anchor="e")
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.refresh_tree(select=ROOT)
        self.on_group_change()

    def _compute_view_limits(self):
        widest = default_params()
        for name, (kind, default) in PARAM_SPECS.items():
            if kind == "abs":
                widest[name] = default * ABS_RANGE_FACTORS[1]
        min_x, min_y, max_x, max_y = build(widest, {})["Substrate_Full"].bounds
        pad = 0.04 * max(max_x - min_x, max_y - min_y)
        return (min_x - pad, max_x + pad, min_y - pad, max_y + pad)

    # --- shape parameters -------------------------------------------------

    @property
    def current_name(self):
        return self.name_var.get()

    def on_group_change(self, _event=None):
        names = [name for name, _, _ in PARAM_GROUPS[self.group_var.get()]]
        self.name_box["values"] = names
        self.name_var.set(names[0])
        self.on_name_change()

    def on_name_change(self, _event=None):
        low, high = param_range(self.current_name)
        self.slider.configure(from_=low, to=high)
        self.slider_var.set(self.params[self.current_name])
        kind, default = PARAM_SPECS[self.current_name]
        unit = "" if kind == "k" else " mm"
        self.range_label.configure(
            text=f"range [{low:.3g}, {high:.3g}]{unit}   default {default:.4g}"
        )
        self.redraw()

    def on_slide(self, _value=None):
        self.params[self.current_name] = self.slider_var.get()
        self.redraw()

    def reset_current(self):
        self.params[self.current_name] = PARAM_SPECS[self.current_name][1]
        self.slider_var.set(self.params[self.current_name])
        self.redraw()

    def reset_all(self):
        self.params = default_params()
        self.tree_nodes = default_tree()
        self.slider_var.set(self.params[self.current_name])
        self.refresh_tree(select=ROOT)
        self.redraw()

    # --- branch tree ------------------------------------------------------

    @property
    def selected(self):
        selection = self.tree.selection()
        return selection[0] if selection else ROOT

    def refresh_tree(self, select):
        """Rebuild the Treeview from self.tree_nodes and select one node."""
        self.tree.delete(*self.tree.get_children(""))
        self.tree.insert("", "end", iid=ROOT, text="SLOT (main slot)", open=True)
        for node_id, node in self.tree_nodes.items():  # parents come before children
            self.tree.insert(
                node["parent"], "end", iid=node_id, text=node_id.rsplit("/", 1)[-1],
                values=(SIDE_DIRECTION[node["side"]],), open=True,
            )
        self.tree.selection_set(select)
        self.tree.see(select)
        self.sync_branch_panel()

    def on_tree_select(self, _event=None):
        self.sync_branch_panel()
        self.redraw()

    def sync_branch_panel(self):
        """Point the add/delete buttons and the K sliders at the selected node."""
        node_id = self.selected
        for button, side in zip(self.add_buttons, child_sides(self.tree_nodes, node_id)):
            button.configure(text=SIDE_LABEL[side])
        is_branch = node_id != ROOT
        self.delete_button.state(["!disabled"] if is_branch else ["disabled"])
        self.k_frame.configure(text=f"Selected branch: {node_id if is_branch else '-'}")
        for i, (var, scale_widget) in enumerate(zip(self.k_vars, self.k_scales)):
            if is_branch:
                low, high = k_range(self.tree_nodes, node_id, i)
                scale_widget.configure(from_=low, to=high)
                scale_widget.state(["!disabled"])
                var.set(self.tree_nodes[node_id]["k"][i])
            else:
                scale_widget.state(["disabled"])
            self._update_k_label(i)

    def _update_k_label(self, i):
        name = ("K1 position", "K2 width", "K3 length")[i]
        node_id = self.selected
        if node_id == ROOT:
            self.k_labels[i].configure(text=name)
            return
        low, high = k_range(self.tree_nodes, node_id, i)
        value = self.tree_nodes[node_id]["k"][i]
        self.k_labels[i].configure(text=f"{name} = {value:.3f} [{low:.2g}, {high:.2g}]")

    def on_k_slide(self, i):
        node_id = self.selected
        if node_id == ROOT:
            return
        self.tree_nodes[node_id]["k"][i] = self.k_vars[i].get()
        self._update_k_label(i)
        self.redraw()

    def add_child(self, index):
        parent = self.selected
        side = child_sides(self.tree_nodes, parent)[index]
        self.refresh_tree(select=add_branch(self.tree_nodes, parent, side))
        self.redraw()

    def delete_selected(self):
        node_id = self.selected
        if node_id == ROOT:
            return
        parent = self.tree_nodes[node_id]["parent"]
        delete_subtree(self.tree_nodes, node_id)
        self.refresh_tree(select=parent)
        self.redraw()

    # --- drawing ----------------------------------------------------------

    def redraw(self):
        value = self.params[self.current_name]
        kind = PARAM_SPECS[self.current_name][0]
        self.value_label.configure(
            text=f"{self.current_name} = {value:.4g}" + ("" if kind == "k" else " mm")
        )

        shapes = build(self.params, self.tree_nodes)
        min_x, min_y, max_x, max_y = shapes["Substrate_Full"].bounds
        self.status_label.configure(
            text=(
                f"Branches   {len(self.tree_nodes)} (x2 mirrored)\n"
                f"Substrate  {max_x - min_x:.2f} x {max_y - min_y:.2f} mm\n"
                f"Footprint  {(max_x - min_x) * (max_y - min_y):.1f} mm^2\n"
                f"Metal      {shapes['metal_area']:.1f} mm^2"
            )
        )

        self.ax.clear()
        draw_geom(self.ax, shapes["Patch"], facecolor="tab:green", alpha=0.3, edgecolor="tab:green")
        draw_geom(self.ax, shapes["Substrate_Full"], facecolor="none", edgecolor="black")
        draw_geom(self.ax, shapes["Slot"], facecolor="tab:blue", alpha=0.6, edgecolor="tab:blue")
        draw_geom(self.ax, shapes["CPW_Feed_Pin"], **FEED_PIN_STYLE)
        for pad in shapes["SMA_Pads"]:
            draw_geom(self.ax, pad, facecolor="none", edgecolor="black", hatch="//")

        # Only the selected branch is outlined and annotated; the rest stay plain.
        info = shapes["branches"].get(self.selected)
        if info is not None:
            for geom in (info["branch"], info["branch_mirror"]):
                if geom.area > 0:
                    draw_geom(self.ax, geom, facecolor="none", edgecolor="tab:orange", lw=2, zorder=5)
            annotate_branch(self.ax, info)
            self.ax.legend(handles=k_legend_handles(), loc="upper left", fontsize=7, framealpha=0.9)

        self.ax.set_xlim(self.view_limits[0], self.view_limits[1])
        self.ax.set_ylim(self.view_limits[2], self.view_limits[3])
        self.ax.set_aspect("equal")
        self.ax.set_title("MSA-BP: patch / slot / feed pin")
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    AntennaDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
