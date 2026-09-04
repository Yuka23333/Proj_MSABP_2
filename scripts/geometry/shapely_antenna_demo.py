"""Interactive demo: MSA-BP planar slot antenna with a slider GUI.

Group dropdown -> variable dropdown -> slider. Every ``# optimizable`` variable of
shapely_rectangle_test.py is exposed:
  - K variables (relative, ratio type) slide over [0.05, 1]
  - absolute variables (mm) slide over [60%, 140%] of their default
"""

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.affinity import scale
from shapely.geometry import LineString, Point, Polygon, box
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
    "5. Branches": [
        ("BRANCH_UP_1_K", "k", 0.5),
        ("BRANCH_UP_1_K2", "k", 0.5),
        ("BRANCH_UP_1_K3", "k", 0.5),
        ("BRANCH_DOWN_1_K", "k", 0.5),
        ("BRANCH_DOWN_1_K2", "k", 0.5),
        ("BRANCH_DOWN_1_K3", "k", 0.05),
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
# Geometry
# ---------------------------------------------------------------------------


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


def _polygons_only(geom):
    """At extreme slider settings a notch can eat a whole corner, leaving the union with
    stray touching lines/points; those make .boundary None. Keep the area parts only."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    return unary_union([g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")])


def ray_distance_up(x, base_y, shape):
    """Shoot a ray straight up from (x, base_y); distance to shape's boundary, else None."""
    ray = LineString([(x, base_y), (x, shape.bounds[3] + 10)])
    hit_ys = [y for y in _boundary_hit_ys(ray.intersection(shape.boundary)) if y > base_y + 1e-9]
    return min(hit_ys) - base_y if hit_ys else None


def ray_distance_down(x, base_y, shape):
    """Shoot a ray straight down from (x, base_y); distance to shape's boundary, else None."""
    ray = LineString([(x, base_y), (x, shape.bounds[1] - 10)])
    hit_ys = [y for y in _boundary_hit_ys(ray.intersection(shape.boundary)) if y < base_y - 1e-9]
    return base_y - max(hit_ys) if hit_ys else None


def _build_branch(k1, k2, k3, base_y, x_low, x_high, near_x, slot_max_x, substrate, upward):
    midpoint_x = k1 * (x_high - x_low) + x_low
    max_width = max(0.0, min(slot_max_x - midpoint_x, midpoint_x - near_x))

    left_x = midpoint_x - k2 * max_width
    right_x = midpoint_x + k2 * max_width

    probe = ray_distance_up if upward else ray_distance_down
    reaches = [probe(x, base_y, substrate) for x in (left_x, right_x)]
    max_length = min((r for r in reaches if r is not None), default=0.0)
    length = k3 * max_length

    tip_y = base_y + length if upward else base_y - length
    branch = box(left_x, min(base_y, tip_y), right_x, max(base_y, tip_y))
    return {
        "midpoint": Point(midpoint_x, base_y),
        "left_endpoint": Point(left_x, base_y),
        "right_endpoint": Point(right_x, base_y),
        "branch": branch,
        "branch_mirror": scale(branch, xfact=-1, yfact=1, origin=(0, 0)),
    }


def build(params):
    """Build every polygon of the antenna from one dict of optimizable parameters."""
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
    keepout_x = CPW_FEED_SLOT_WIDE_WIDTH + CPW_KEEPOUT_MARGIN

    branch_up = _build_branch(
        p["BRANCH_UP_1_K"], p["BRANCH_UP_1_K2"], p["BRANCH_UP_1_K3"],
        base_y=slot_max_y, x_low=2, x_high=slot_max_x - FIXED_OFFSET, near_x=1,
        slot_max_x=slot_max_x, substrate=Upper_Substrate, upward=True,
    )
    branch_down = _build_branch(
        p["BRANCH_DOWN_1_K"], p["BRANCH_DOWN_1_K2"], p["BRANCH_DOWN_1_K3"],
        base_y=slot_min_y, x_low=keepout_x, x_high=slot_max_x - FIXED_OFFSET, near_x=keepout_x,
        slot_max_x=slot_max_x, substrate=Lower_Substrate, upward=False,
    )

    Slot = unary_union([
        slot_main,
        branch_up["branch"], branch_up["branch_mirror"],
        branch_down["branch"], branch_down["branch_mirror"],
        CPW_Feed_Slot_1, CPW_Feed_Slot_2,
        Matching_Stub1, Matching_Stub2,
    ])
    Patch = unary_union([Upper_Substrate, Lower_Substrate])
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
        "branch_points": (
            branch_up["midpoint"], branch_up["left_endpoint"], branch_up["right_endpoint"],
        ),
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


class AntennaDemo:
    def __init__(self, root):
        self.root = root
        self.params = default_params()
        self.view_limits = self._compute_view_limits()

        root.title("MSA-BP antenna parameter explorer")

        controls = ttk.Frame(root, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(controls, text="Variable group").pack(anchor="w")
        self.group_var = tk.StringVar(value=next(iter(PARAM_GROUPS)))
        self.group_box = ttk.Combobox(
            controls, textvariable=self.group_var, state="readonly",
            values=list(PARAM_GROUPS), width=28,
        )
        self.group_box.pack(anchor="w", pady=(0, 12))
        self.group_box.bind("<<ComboboxSelected>>", self.on_group_change)

        ttk.Label(controls, text="Variable").pack(anchor="w")
        self.name_var = tk.StringVar()
        self.name_box = ttk.Combobox(controls, textvariable=self.name_var, state="readonly", width=28)
        self.name_box.pack(anchor="w", pady=(0, 12))
        self.name_box.bind("<<ComboboxSelected>>", self.on_name_change)

        self.value_label = ttk.Label(controls, text="", font=("TkDefaultFont", 10, "bold"))
        self.value_label.pack(anchor="w")

        self.slider_var = tk.DoubleVar()
        self.slider = ttk.Scale(
            controls, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.slider_var, command=self.on_slide, length=240,
        )
        self.slider.pack(anchor="w")

        self.range_label = ttk.Label(controls, text="")
        self.range_label.pack(anchor="w", pady=(0, 12))

        ttk.Button(controls, text="Reset this variable", command=self.reset_current).pack(
            anchor="w", fill=tk.X
        )
        ttk.Button(controls, text="Reset all", command=self.reset_all).pack(
            anchor="w", fill=tk.X, pady=(4, 12)
        )

        self.status_label = ttk.Label(controls, text="", justify="left")
        self.status_label.pack(anchor="w")

        self.figure = Figure(figsize=(7.5, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False).pack(side=tk.BOTTOM, fill=tk.X)

        self.on_group_change()

    def _compute_view_limits(self):
        widest = default_params()
        for name, (kind, default) in PARAM_SPECS.items():
            if kind == "abs":
                widest[name] = default * ABS_RANGE_FACTORS[1]
        min_x, min_y, max_x, max_y = build(widest)["Substrate_Full"].bounds
        pad = 0.04 * max(max_x - min_x, max_y - min_y)
        return (min_x - pad, max_x + pad, min_y - pad, max_y + pad)

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
        self.slider_var.set(self.params[self.current_name])
        self.redraw()

    def redraw(self):
        value = self.params[self.current_name]
        kind = PARAM_SPECS[self.current_name][0]
        self.value_label.configure(
            text=f"{self.current_name} = {value:.4g}" + ("" if kind == "k" else " mm")
        )

        shapes = build(self.params)
        min_x, min_y, max_x, max_y = shapes["Substrate_Full"].bounds
        self.status_label.configure(
            text=(
                f"Substrate  {max_x - min_x:.2f} x {max_y - min_y:.2f} mm\n"
                f"Footprint  {(max_x - min_x) * (max_y - min_y):.1f} mm^2\n"
                f"Metal      {shapes['metal_area']:.1f} mm^2"
            )
        )

        self.ax.clear()
        draw_geom(self.ax, shapes["Patch"], facecolor="tab:green", alpha=0.3, edgecolor="tab:green")
        draw_geom(self.ax, shapes["Substrate_Full"], facecolor="none", edgecolor="black")
        draw_geom(self.ax, shapes["Slot"], facecolor="tab:blue", alpha=0.6, edgecolor="tab:blue")
        draw_geom(self.ax, shapes["CPW_Feed_Pin"], facecolor="tab:cyan", alpha=0.5, edgecolor="tab:cyan")
        for pad in shapes["SMA_Pads"]:
            draw_geom(self.ax, pad, facecolor="none", edgecolor="black", hatch="//")

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
