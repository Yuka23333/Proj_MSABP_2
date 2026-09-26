"""Prepare native-Shapely hole/island fixtures and figures; never connect to CST."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

if __package__:
    from . import shapely_antenna_tree_model as model
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import shapely_antenna_tree_model as model


def parts(geometry):
    if isinstance(geometry, Polygon):
        return [geometry]
    return [p for g in geometry.geoms for p in parts(g)]


def paint(ax, polygon, color):
    polygon = orient(polygon, sign=1)
    points, codes = [], []
    for ring in [polygon.exterior, *polygon.interiors]:
        xy = list(ring.coords)
        points.extend(xy)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MplPath(points, codes), facecolor=color, edgecolor="#273342", lw=.8))


def main():
    output = Path(__file__).resolve().parents[2] / "results/processed/tree_topology_cases_20260926"
    output.mkdir(parents=True, exist_ok=True)
    params = model.default_params()
    joined = model.default_tree()
    joined["U1"]["k"] = [.15, .3, .8]
    model.add_branch(joined, "U1", "L", (.7, .3, 1))
    hole = deepcopy(joined)
    hole["U1"]["k"][1] = 0
    floating = deepcopy(joined)
    model.add_branch(floating, "U1", "L", (.3, .2, 1))
    cases = [("joined", joined), ("zero_width_hole", hole), ("floating_island", floating)]
    titles = ["A. Joined inward branches", "B. Parent K2 = 0", "C. Two bridges: floating island"]
    colors = {"outer_copper": "#daa846", "feed_connected": "#438fc4", "floating_island": "#da675a"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    summaries = []
    for col, (name, tree) in enumerate(cases):
        shapes = model.build(params, tree)
        copper = shapes["Patch"].difference(shapes["Slot"]).union(shapes["CPW_Feed_Pin"])
        assert copper.is_valid and shapes["Slot"].is_valid
        components = sorted(parts(copper), key=lambda p: -p.area)
        records = []
        for index, polygon in enumerate(components):
            feed = polygon.intersection(shapes["CPW_Feed_Pin"]).area > 1e-9
            outer = polygon.boundary.intersection(shapes["Patch"].boundary).length > 1e-8
            role = "feed_connected" if feed else "outer_copper" if outer else "floating_island"
            records.append({"index": index, "role": role, "area_mm2": polygon.area,
                            "exterior": list(polygon.exterior.coords),
                            "holes": [list(r.coords) for r in polygon.interiors]})
            for ax in axes[:, col]:
                paint(ax, polygon, colors[role])
        holes = sum(len(p.interiors) for p in components)
        islands = sum(r["role"] == "floating_island" for r in records)
        summary = {"case": name, "copper_components": len(components), "copper_holes": holes,
                   "floating_islands": islands, "slot_components": len(parts(shapes["Slot"])),
                   "slot_holes": sum(len(p.interiors) for p in parts(shapes["Slot"]))}
        payload = {"params": params, "tree": tree, "build_options": {"snap_fraction": .1, "tip_clearance": 1.0},
                   "summary": summary, "copper_components": records}
        (output / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summaries.append(summary)
        for row, ax in enumerate(axes[:, col]):
            x, y = shapes["Substrate_Full"].exterior.xy
            ax.plot(x, y, "--", color="#89979f", lw=1)
            ax.axvline(0, color="#99a6ad", lw=.5, ls=":")
            ax.set_aspect("equal")
            ax.set_facecolor("#edf3f4")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            if row == 0:
                ax.set_xlim(-36, 36)
                ax.set_ylim(-23, 22)
                ax.set_title(titles[col] + f"\n{len(components)} copper components / {holes} holes")
            else:
                ax.set_xlim(-8, 8)
                ax.set_ylim(-2, 16)
                ax.set_title("Upper-slot detail")
                for info in shapes["branches"].values():
                    if info["branch"].area == 0 and info["direction"] == "up":
                        (x0, y0), (x1, y1) = info["midpoint"], info["tip"]
                        for sign in (-1, 1):
                            ax.plot([sign*x0, sign*x1], [y0, y1], ":", color="#9b59b6", lw=1)
    assert summaries[0]["copper_components"] == 2
    assert summaries[1]["copper_components"] == 1 and summaries[1]["copper_holes"] == 1
    assert summaries[2]["floating_islands"] >= 1
    fig.suptitle("Native Shapely topology fixtures — default 1 mm tip protection retained\n"
                 "Gold: outer copper | Blue: feed-connected copper | Red: floating island | Pale: no copper\n"
                 "Purple dotted: zero-width construction line (not a slot)", fontsize=13)
    fig.savefig(output / "topology_comparison.png", dpi=180)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(output)


if __name__ == "__main__":
    main()
