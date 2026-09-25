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

# Geometry and tree operations are shared with headless workflow callers.
if __package__:
    from .shapely_antenna_tree_model import (
        ABS_RANGE_FACTORS,
        DIRECTIONS,
        K_RANGE as K_RANGE,
        NEW_BRANCH_K as NEW_BRANCH_K,
        PARAM_GROUPS,
        PARAM_SPECS,
        ROOT,
        SIDE_DIRECTION,
        add_branch,
        build,
        child_sides,
        default_params,
        default_tree,
        delete_subtree,
        k_range,
        param_range,
        relax_upper_k,
    )
else:
    from shapely_antenna_tree_model import (
        ABS_RANGE_FACTORS,
        DIRECTIONS,
        K_RANGE as K_RANGE,
        NEW_BRANCH_K as NEW_BRANCH_K,
        PARAM_GROUPS,
        PARAM_SPECS,
        ROOT,
        SIDE_DIRECTION,
        add_branch,
        build,
        child_sides,
        default_params,
        default_tree,
        delete_subtree,
        k_range,
        param_range,
        relax_upper_k,
    )

SIDE_LABEL = {"U": "+ Up", "D": "+ Down", "L": "+ Left (inward)", "R": "+ Right (outward)"}


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
    ax.annotate(f"K2={k2:.2f} -> {info['effective_k2']:.2f}", info["endpoints"][1], xytext=(16 * px + 8 * dx, 16 * py + 8 * dy),
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
        effective = f" -> effective {relax_upper_k(value):.3f}" if i == 1 else ""
        self.k_labels[i].configure(text=f"{name} = {value:.3f}{effective} [{low:.2g}, {high:.2g}]")

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
