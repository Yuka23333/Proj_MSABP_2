"""Three static figures explaining the Branch up 1 shaper (K1 -> K2 -> K3).

The shaper never has to reject a branch after the fact: each K is a fraction of
whatever room is *left* at that stage, so every (K1, K2, K3) in [0, 1]^3 yields a
branch that stays inside the patch and clear of the centre line / slot end.

  step 1  K1  midpoint_x = x_low + K1 * (x_high - x_low)
  step 2  K2  half_width = K2 * min(slot_max_x - midpoint_x, midpoint_x - near_x)
  step 3  K3  length     = K3 * min(ray_up(left_x), ray_up(right_x))

Geometry comes from ``shapely_antenna_demo.build`` whose ``_build_branch`` mirrors
``shapely_antenna_model._build_branch_pair`` (the production shaper).

Usage:
    python scripts/geometry/branch_shaper_figures.py
Writes results/figures/branch_shaper_step{1,2,3}_k{1,2,3}.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # set before the demo module pulls in the Tk backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from shapely_antenna_demo import FIXED_OFFSET, build, default_params, draw_geom

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "figures"

PATCH_KW = dict(facecolor="tab:green", alpha=0.3, edgecolor="tab:green")
SLOT_KW = dict(facecolor="tab:blue", alpha=0.6, edgecolor="tab:blue")
K_COLOUR = {"K1": "tab:red", "K2": "tab:orange", "K3": "tab:purple"}
SWEEP = [0.05, 0.25, 0.5, 0.75, 1.0]
SWEEP_COLOURS = plt.cm.viridis(np.linspace(0.05, 0.85, len(SWEEP)))
LABEL_BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)


def branch_up(k1, k2, k3):
    params = default_params()
    params.update({"BRANCH_UP_1_K": k1, "BRANCH_UP_1_K2": k2, "BRANCH_UP_1_K3": k3})
    shapes = build(params)
    return shapes, shapes["branches"]["up"]


def setup_axes(ax, shapes, title):
    """Zoom on the right half of the upper patch, where Branch up 1 lives."""
    draw_geom(ax, shapes["Upper_Substrate"], **PATCH_KW)
    draw_geom(ax, shapes["Lower_Substrate"], **PATCH_KW)
    draw_geom(ax, shapes["slot_main"], **SLOT_KW)
    y_min = shapes["slot_main"].bounds[1] - 3.0
    ax.axvline(0, color="0.5", lw=0.7, ls=":")
    ax.text(0.3, y_min + 0.3, "x = 0 (mirror axis)", fontsize=7, color="0.4", ha="left", va="bottom")
    ax.set_xlim(-3.5, shapes["Upper_Substrate"].bounds[2] + 1.5)
    ax.set_ylim(y_min, shapes["Upper_Substrate"].bounds[3] + 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)


def vline_label(ax, x, y, text, colour):
    ax.axvline(x, color=colour, lw=0.8, ls=":")
    ax.text(x, y, text, fontsize=7, color=colour, ha="center", va="bottom",
            rotation=90, bbox=LABEL_BOX)


def sweep_legend(ax, name, values, extra=()):
    handles = [Line2D([], [], color=c, lw=1.6, label=f"{name} = {v:g}")
               for v, c in zip(values, SWEEP_COLOURS)]
    ax.legend(handles=[*handles, *extra], loc="upper left", fontsize=7, framealpha=0.92)


# ---------------------------------------------------------------------------
# Step 1: K1 -> midpoint position
# ---------------------------------------------------------------------------


def figure_k1():
    shapes, info = branch_up(0.5, 0.5, 0.5)
    base_y, x_low, x_high = info["base_y"], info["x_low"], info["x_high"]
    slot_max_x = info["slot_max_x"]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    setup_axes(ax, shapes, "Step 1 / K1: the midpoint slides along the allowed span of the slot top edge")

    # Allowed span vs. the two excluded stubs.
    ax.plot([x_low, x_high], [base_y, base_y], color=K_COLOUR["K1"], lw=4,
            solid_capstyle="butt", zorder=4)
    ax.plot([0, x_low], [base_y, base_y], color="0.2", lw=4, solid_capstyle="butt", zorder=4)
    ax.plot([x_high, slot_max_x], [base_y, base_y], color="0.2", lw=4, solid_capstyle="butt", zorder=4)
    vline_label(ax, x_low, base_y + 10, f"x_low = {x_low:g}", "0.2")
    vline_label(ax, x_high, base_y + 8, f"x_high = slot_max_x - {FIXED_OFFSET:g} = {x_high:g}", "0.2")
    vline_label(ax, slot_max_x, base_y + 8, f"slot_max_x = {slot_max_x:g}", "0.2")

    # Midpoints for a K1 sweep, plus the branch each one produces at K2 = K3 = 0.5
    # (its width shrinks automatically near either end: that is step 2's job).
    for k1, colour in zip(SWEEP, SWEEP_COLOURS):
        _, inf = branch_up(k1, 0.5, 0.5)
        draw_geom(ax, inf["branch"], facecolor="none", edgecolor=colour, lw=1.4, zorder=3)
        ax.plot(inf["midpoint"].x, base_y, "o", color=colour, ms=8, mec="white", mew=1.2, zorder=6)
        ax.annotate(f"K1={k1:g}\nx={inf['midpoint'].x:.1f}", (inf["midpoint"].x, base_y),
                    xytext=(0, -16), textcoords="offset points", ha="center", va="top",
                    fontsize=7.5, color=colour, bbox=LABEL_BOX, zorder=7)

    ax.text(0.99, 0.97,
            "midpoint_x = x_low + K1 * (x_high - x_low)\n"
            "K1 in [0, 1]  ->  midpoint always on the red span:\n"
            f"never inside the centre stub (x < {x_low:g}) nor within\n"
            f"FIXED_OFFSET = {FIXED_OFFSET:g} mm of the slot end",
            transform=ax.transAxes, fontsize=8, ha="right", va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"))
    sweep_legend(ax, "K1", SWEEP, extra=[
        Line2D([], [], color=K_COLOUR["K1"], lw=4, label="allowed span for the midpoint"),
        Line2D([], [], color="0.2", lw=4, label="excluded"),
        Line2D([], [], color="0.4", lw=1.4, label="resulting branch (K2 = K3 = 0.5)"),
    ])
    return fig


# ---------------------------------------------------------------------------
# Step 2: K2 -> width, capped by the nearer of slot end / centre keep-out
# ---------------------------------------------------------------------------


def _panel_k2(ax, shapes, k1, k3):
    _, info = branch_up(k1, 0.5, k3)
    base_y, mid_x = info["base_y"], info["midpoint"].x
    near_x, slot_max_x, max_w = info["near_x"], info["slot_max_x"], info["max_width"]
    to_centre, to_end = mid_x - near_x, slot_max_x - mid_x
    centre_binds = to_centre <= to_end

    setup_axes(ax, shapes, f"Step 2 / K2 at K1 = {k1:g}: half-width = K2 * min({to_end:.2f}, {to_centre:.2f}) = K2 * {max_w:.2f}")
    vline_label(ax, near_x, base_y + 9, f"near_x = {near_x:g}", "0.2")
    vline_label(ax, slot_max_x, base_y + 9, f"slot_max_x = {slot_max_x:g}", "0.2")

    # The two candidate distances, drawn as arrows from the midpoint.
    y_arrow = base_y - 1.9
    for x_to, dist, binds, label in (
        (near_x, to_centre, centre_binds, f"midpoint - near_x = {to_centre:.2f}"),
        (slot_max_x, to_end, not centre_binds, f"slot_max_x - midpoint = {to_end:.2f}"),
    ):
        colour = K_COLOUR["K2"] if binds else "0.45"
        ax.annotate("", (x_to, y_arrow), (mid_x, y_arrow),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=2 if binds else 1.2,
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.text((x_to + mid_x) / 2, y_arrow - 0.35,
                label + ("  <- binds" if binds else ""), fontsize=7.5, color=colour,
                ha="center", va="top", fontweight="bold" if binds else "normal", bbox=LABEL_BOX)
    ax.plot(mid_x, base_y, "o", color=K_COLOUR["K1"], ms=8, mec="white", mew=1.2, zorder=6)

    # Nested branches for a K2 sweep; K2 = 1 just touches the binding limit,
    # and its mirror shows the 2 * near_x gap kept at the centre.
    top_y = base_y
    for k2, colour in zip(SWEEP, SWEEP_COLOURS):
        _, inf = branch_up(k1, k2, k3)
        top_y = max(top_y, inf["tip"].y)
        draw_geom(ax, inf["branch"], facecolor="none", edgecolor=colour, lw=1.5, zorder=3)
        if k2 == 1.0:
            draw_geom(ax, inf["branch_mirror"], facecolor="none", edgecolor=colour, lw=1.0,
                      ls="--", zorder=3)
        for pt in (inf["left_endpoint"], inf["right_endpoint"]):
            ax.plot(pt.x, pt.y, "s", color=colour, ms=5, mec="white", mew=0.8, zorder=6)
    ax.annotate(f"midpoint (K1={k1:g})", (mid_x, base_y), (mid_x, top_y + 0.5),
                ha="center", va="bottom", fontsize=7.5, color=K_COLOUR["K1"], bbox=LABEL_BOX,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=K_COLOUR["K1"]), zorder=7)
    sweep_legend(ax, "K2", SWEEP, extra=[
        Line2D([], [], color=SWEEP_COLOURS[-1], lw=1.0, ls="--", label="mirror of K2 = 1 branch"),
    ])


def figure_k2():
    shapes, _ = branch_up(0.5, 0.5, 0.5)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 11.5))
    _panel_k2(axes[0], shapes, k1=0.25, k3=0.3)   # centre keep-out is the nearer limit
    _panel_k2(axes[1], shapes, k1=0.8, k3=0.3)    # slot end is the nearer limit
    fig.text(0.5, 0.005,
             "half_width = K2 * min(slot_max_x - midpoint_x, midpoint_x - near_x)\n"
             "K2 in [0, 1] -> the branch can at most touch the slot end or the centre keep-out, never cross either.",
             ha="center", va="bottom", fontsize=8.5, family="monospace")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Step 3: K3 -> length, capped by the shorter ray to the patch boundary
# ---------------------------------------------------------------------------


def _panel_k3(ax, shapes, k1, k2):
    _, info = branch_up(k1, k2, 0.5)
    base_y, mid_x = info["base_y"], info["midpoint"].x
    left, right = info["left_endpoint"], info["right_endpoint"]
    reaches, max_len = info["reaches"], info["max_length"]
    binding = int(np.argmin(reaches))

    setup_axes(ax, shapes, f"Step 3 / K3 at K1 = {k1:g}, K2 = {k2:g}: length = K3 * min({reaches[0]:.2f}, {reaches[1]:.2f}) = K3 * {max_len:.2f}")

    # Rays cast upward from both endpoints until they hit the patch boundary.
    for i, (pt, reach) in enumerate(zip((left, right), reaches)):
        binds = i == binding
        colour = K_COLOUR["K3"] if binds else "0.45"
        ax.annotate("", (pt.x, base_y + reach), (pt.x, base_y),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=2 if binds else 1.2,
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.plot(pt.x, base_y + reach, "x", color=colour, ms=8, mew=2, zorder=6)
        ax.plot(pt.x, pt.y, "s", color=K_COLOUR["K2"], ms=6, mec="white", mew=0.8, zorder=6)
        side = "left" if i == 0 else "right"
        ax.text(pt.x + (-0.5 if i == 0 else 0.5), base_y + reach / 2,
                f"ray from {side} edge\nreach = {reach:.2f}" + ("\n<- binds" if binds else ""),
                fontsize=7.5, color=colour, ha="right" if i == 0 else "left", va="center",
                fontweight="bold" if binds else "normal", bbox=LABEL_BOX, zorder=7)
    ax.plot([left.x - 1.5, right.x + 1.5], [base_y + max_len] * 2, color=K_COLOUR["K3"],
            lw=1.0, ls=":", zorder=4)
    ax.text(right.x + 1.7, base_y + max_len, f"max_length = {max_len:.2f}", fontsize=7.5,
            color=K_COLOUR["K3"], ha="left", va="center", bbox=LABEL_BOX, zorder=7)

    # Nested branches for a K3 sweep; K3 = 1 just touches the boundary.
    for k3, colour in zip(SWEEP, SWEEP_COLOURS):
        _, inf = branch_up(k1, k2, k3)
        draw_geom(ax, inf["branch"], facecolor="none", edgecolor=colour, lw=1.5, zorder=3)
        ax.plot(mid_x, inf["tip"].y, "^", color=colour, ms=6, mec="white", mew=0.8, zorder=6)
    sweep_legend(ax, "K3", SWEEP)


def figure_k3():
    shapes, _ = branch_up(0.5, 0.5, 0.5)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 11.5))
    _panel_k3(axes[0], shapes, k1=0.5, k2=0.5)   # right edge sits under the ear
    _panel_k3(axes[1], shapes, k1=0.9, k2=0.3)   # both edges under the notch, different steps
    fig.text(0.5, 0.005,
             "length = K3 * min(ray_up(left_x), ray_up(right_x))\n"
             "K3 in [0, 1] -> the tip can at most touch the patch boundary above the branch, never leave it.",
             ha="center", va="bottom", fontsize=8.5, family="monospace")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, make in (
        ("branch_shaper_step1_k1.png", figure_k1),
        ("branch_shaper_step2_k2.png", figure_k2),
        ("branch_shaper_step3_k3.png", figure_k3),
    ):
        fig = make()
        path = OUT_DIR / name
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
