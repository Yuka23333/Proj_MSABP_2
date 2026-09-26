"""Plot and export a deliberately outer-boundary-cutting copper stress case.

This is not an admissible design or a second-order branch implementation. It
exercises the same disconnected-copper export/build path needed by future
enclosed islands. No CST process or solver is started by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from shapely.affinity import scale
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.geometry import shapely_antenna_model as model  # noqa: E402
from scripts.geometry.conductor_components import final_conductor  # noqa: E402

DEFAULT_JSON = ROOT / "data/geometry/copper_island_stress.json"
DEFAULT_FIGURE = ROOT / "results/figures/copper_island_stress.png"


def stress_parameters():
    return replace(model.DEFAULT_PARAMETERS, BRANCH_UP_1_K3=1.0, BRANCH_DOWN_1_K3=1.0)


def stress_payload():
    payload = model.polygon_export_payload(
        stress_parameters(),
        include_conductor_components=True,
    )
    copper = final_conductor(payload["vertices"])
    symmetry_error = copper.symmetric_difference(
        scale(copper, xfact=-1, origin=(0, 0))
    ).area
    if symmetry_error > 1e-8:
        raise ValueError(
            "Stress case no longer respects the existing left/right mirror"
        )
    pads = box(3.49, 0, 4.76, 4.5).union(box(-4.76, 0, -3.49, 4.5))
    for component in payload["conductor_components"]:
        polygon = Polygon(component["exterior"], component["holes"])
        if component["role"] != "feed_connected":
            component["role"] = (
                "ground_pad_contact" if polygon.intersects(pads) else "floating_island"
            )
    payload["meta"]["case_note"] = (
        "Deliberate outer-boundary-cutting stress case, NOT an allowed design. "
        "Desired future islands are enclosed by intersecting second-order slots. "
        "Existing left/right mirror (x,y)->(-x,y) is preserved. Roles refer only "
        "to planar feed/SMA-pad contact, not a full 3D electrical validation."
    )
    return payload


def write_case(output=DEFAULT_JSON, figure=DEFAULT_FIGURE):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon as PolygonPatch, Rectangle

    payload = stress_payload()
    output, figure = Path(output), Path(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    colors = {
        "feed_connected": "#276e9b",
        "ground_pad_contact": "#5c927b",
        "floating_island": "#d87124",
    }
    labels = {
        "feed_connected": "Feed-connected copper",
        "ground_pad_contact": "SMA-ground-connected pieces",
        "floating_island": "Floating islands",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    baseline = final_conductor(model.polygon_export_payload()["vertices"])
    datasets = [
        [
            {
                "exterior": baseline.exterior.coords,
                "holes": [r.coords for r in baseline.interiors],
                "role": "feed_connected",
            }
        ],
        payload["conductor_components"],
    ]
    xmin, ymin, xmax, ymax = Polygon(payload["vertices"]["Patch"]).bounds
    for ax, components, title in zip(
        axes,
        datasets,
        [
            "Default geometry: one connected copper body",
            "Stress case: 5 copper regions, including 2 islands",
        ],
    ):
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor="#f5f1e6",
                edgecolor="#928b7b",
                linewidth=1.5,
            )
        )
        for index, component in enumerate(components):
            ax.add_patch(
                PolygonPatch(
                    component["exterior"],
                    facecolor=colors[component["role"]],
                    edgecolor="#25323b",
                    linewidth=0.7,
                )
            )
            for ring in component["holes"]:
                ax.add_patch(
                    PolygonPatch(
                        ring, facecolor="#f5f1e6", edgecolor="#25323b", linewidth=0.7
                    )
                )
            if len(components) > 1:
                point = Polygon(
                    component["exterior"], component["holes"]
                ).representative_point()
                ax.text(
                    point.x,
                    point.y,
                    str(index),
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                )
        ax.set(
            xlim=(xmin - 3, xmax + 3),
            ylim=(ymin - 3, ymax + 3),
            xlabel="X [mm]",
            ylabel="Y [mm]",
            title=title,
        )
        ax.set_aspect("equal")
        ax.axvline(0, color="#858585", linewidth=0.7, linestyle="--")
    fig.legend(
        handles=[Patch(color=colors[k], label=labels[k]) for k in colors],
        loc="outside lower center",
        ncol=3,
    )
    fig.suptitle(
        "Copper-island export test | upper K3 = lower K3 = 1\nForced outer-boundary cut; not a valid design-space example",
        fontsize=13,
    )
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"Curves: {output}\nFigure: {figure}")
    for index, component in enumerate(payload["conductor_components"]):
        print(
            f"{index}: {component['role']}, area={component['area_mm2']:.6f} mm^2, holes={len(component['holes'])}"
        )
    return output, figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    write_case(args.output, args.figure)
