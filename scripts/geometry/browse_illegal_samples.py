"""Browse the 18 invalid curves and 3 Slot/Patch crossings interactively."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.widgets import Button
from shapely.geometry import LineString, LinearRing, Polygon
from shapely.validation import explain_validity


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import antenna_sampler  # noqa: E402
from scripts.geometry import shapely_antenna_model  # noqa: E402


DEFAULT_INPUT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "samples"
    / "antenna_samples_1024_curve_intersections.csv"
)
CURVE_NAMES = ("Patch", "Slot", "CPW_Feed_Pin")
CURVE_STYLES = {
    "Patch": {"color": "#2e8b57", "alpha": 0.20},
    "Slot": {"color": "#2563eb", "alpha": 0.28},
    "CPW_Feed_Pin": {"color": "#f59e0b", "alpha": 0.38},
}
SLOT_PATCH_PAIR = "Slot__Patch"
VALIDITY_POINT_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\]"
)


def _boolean_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().eq("true")


def load_illegal_samples(path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load the 18 invalid polygons plus true non-bottom Slot/Patch crossings."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"intersection sample CSV not found: {source}; run "
            "scripts/automation/check_sampled_curve_intersections.py first"
        )
    frame = pd.read_csv(source)
    required = {
        "sample_id",
        "geometry_valid",
        "geometry_error",
        "non_bottom_crossing_pairs",
        *antenna_sampler.PARAMETER_REGISTRY,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sample CSV is missing columns: {sorted(missing)}")
    geometry_valid = _boolean_series(frame["geometry_valid"])
    crossing_pairs = frame["non_bottom_crossing_pairs"].fillna("").astype(str)
    slot_patch_crossing = crossing_pairs.str.split(";").map(
        lambda pairs: SLOT_PATCH_PAIR in pairs
    )
    selected = frame.loc[(~geometry_valid) | slot_patch_crossing].copy()
    selected["geometry_valid"] = geometry_valid.loc[selected.index].to_numpy()
    selected.sort_values("sample_id", inplace=True)
    selected.reset_index(drop=True, inplace=True)
    if selected.empty:
        raise ValueError("the sample CSV contains no selected illegal samples")
    return selected


def _closed_points(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    converted = [(float(x_value), float(y_value)) for x_value, y_value in points]
    return [*converted, converted[0]]


def _plot_geometry(
    axes: Axes,
    geometry: Any,
    *,
    color: str,
    label: str | None = None,
) -> None:
    if geometry.is_empty:
        return
    geometry_type = geometry.geom_type
    if geometry_type == "Point":
        axes.scatter(
            [geometry.x],
            [geometry.y],
            marker="x",
            s=90,
            linewidths=2.4,
            color=color,
            zorder=8,
            label=label,
        )
        return
    if geometry_type == "LineString" or geometry_type == "LinearRing":
        x_values, y_values = geometry.xy
        axes.plot(
            x_values,
            y_values,
            color=color,
            linewidth=4.0,
            zorder=7,
            label=label,
        )
        return
    if hasattr(geometry, "geoms"):
        for index, part in enumerate(geometry.geoms):
            _plot_geometry(
                axes,
                part,
                color=color,
                label=label if index == 0 else None,
            )


def _validity_marker(validity_text: str) -> tuple[float, float] | None:
    match = VALIDITY_POINT_PATTERN.search(validity_text)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


class IllegalSampleBrowser:
    """Matplotlib two-button browser for the selected illegal samples."""

    def __init__(self, samples: pd.DataFrame) -> None:
        self.samples = samples.reset_index(drop=True)
        self.index = 0
        self.figure = plt.figure(figsize=(14.5, 8.2))
        self.axes = self.figure.add_axes((0.055, 0.15, 0.66, 0.75))
        self.info_axes = self.figure.add_axes((0.74, 0.10, 0.25, 0.86))
        self.previous_button = Button(
            self.figure.add_axes((0.36, 0.035, 0.12, 0.055)),
            "Previous",
        )
        self.next_button = Button(
            self.figure.add_axes((0.52, 0.035, 0.12, 0.055)),
            "Next",
        )
        self.previous_button.on_clicked(self._previous)
        self.next_button.on_clicked(self._next)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.draw_current()

    def _previous(self, _event: Any = None) -> None:
        self.index = (self.index - 1) % len(self.samples)
        self.draw_current()

    def _next(self, _event: Any = None) -> None:
        self.index = (self.index + 1) % len(self.samples)
        self.draw_current()

    def _on_key(self, event: Any) -> None:
        if event.key == "left":
            self._previous()
        elif event.key == "right":
            self._next()

    def draw_current(self) -> None:
        row = self.samples.iloc[self.index]
        parameters = antenna_sampler.parameters_from_csv_row(row.to_dict())
        payload = shapely_antenna_model.polygon_export_payload(parameters)
        vertices: Mapping[str, Sequence[Sequence[float]]] = payload["vertices"]

        self.axes.clear()
        self.info_axes.clear()
        self.info_axes.axis("off")

        rings: dict[str, LinearRing] = {}
        validity_lines: list[str] = []
        all_points: list[tuple[float, float]] = []
        for name in CURVE_NAMES:
            points = _closed_points(vertices[name])
            all_points.extend(points[:-1])
            style = CURVE_STYLES[name]
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            self.axes.fill(
                x_values,
                y_values,
                color=style["color"],
                alpha=style["alpha"],
                zorder=1,
            )
            self.axes.plot(
                x_values,
                y_values,
                color=style["color"],
                linewidth=1.6,
                zorder=3,
                label=name,
            )
            rings[name] = LinearRing(points)
            polygon = Polygon(points)
            validity = explain_validity(polygon)
            if validity != "Valid Geometry":
                validity_lines.append(f"{name}: {validity}")
                marker = _validity_marker(validity)
                if marker is not None:
                    self.axes.scatter(
                        [marker[0]],
                        [marker[1]],
                        marker="x",
                        s=130,
                        linewidths=3.0,
                        color="#dc2626",
                        zorder=9,
                        label="invalid/self-intersection",
                    )

        bottom_y = min(point[1] for point in all_points)
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        bottom_edge = LineString(((min_x - 1.0, bottom_y), (max_x + 1.0, bottom_y)))
        self.axes.axhline(
            bottom_y,
            color="#64748b",
            linestyle="--",
            linewidth=1.0,
            label="ignored common bottom edge",
            zorder=2,
        )

        slot_patch = rings["Slot"].intersection(rings["Patch"]).difference(
            bottom_edge
        )
        if not slot_patch.is_empty:
            _plot_geometry(
                self.axes,
                slot_patch,
                color="#dc2626",
                label="Slot/Patch crossing",
            )

        slot_pin = rings["Slot"].intersection(
            rings["CPW_Feed_Pin"]
        ).difference(bottom_edge)
        if not slot_pin.is_empty:
            _plot_geometry(
                self.axes,
                slot_pin,
                color="#9333ea",
                label="fixed Slot/Pin overlap",
            )

        x_span = max_x - min_x
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        y_span = max_y - min_y
        self.axes.set_xlim(min_x - max(1.0, 0.05 * x_span), max_x + max(1.0, 0.05 * x_span))
        self.axes.set_ylim(min_y - max(1.0, 0.05 * y_span), max_y + max(1.0, 0.05 * y_span))
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
        self.axes.set_xlabel("X (mm)")
        self.axes.set_ylabel("Y (mm)")
        self.axes.legend(loc="upper left", ncol=2, fontsize=8)

        sample_id = int(row["sample_id"])
        reasons: list[str] = []
        if not bool(row["geometry_valid"]):
            reasons.append(str(row["geometry_error"]))
        crossing_pairs = str(row.get("non_bottom_crossing_pairs", ""))
        if SLOT_PATCH_PAIR in crossing_pairs:
            reasons.append("Slot/Patch boundaries cross outside bottom edge")
        self.axes.set_title(
            f"Illegal sample {self.index + 1}/{len(self.samples)}  |  sample_id={sample_id}\n"
            + " | ".join(reasons),
            fontsize=9.5,
            pad=9,
        )

        parameter_lines = [
            f"{name:<29} {float(row[name]): .6g}"
            for name in antenna_sampler.PARAMETER_REGISTRY
        ]
        information = [
            f"sample_id: {sample_id}",
            f"geometry_valid: {bool(row['geometry_valid'])}",
            "",
            "Detected issue:",
            *(validity_lines or ["No individual polygon validity error"]),
        ]
        if SLOT_PATCH_PAIR in crossing_pairs:
            information.append("Slot/Patch: true boundary crossing")
        information.extend(
            [
                "",
                "Purple line is the fixed Slot/Pin",
                "top overlap; it is not the reason",
                "this sample was selected.",
                "",
                "Parameters:",
                *parameter_lines,
            ]
        )
        self.info_axes.text(
            0.0,
            1.0,
            "\n".join(information),
            ha="left",
            va="top",
            family="monospace",
            fontsize=7.4,
            color="#0f172a",
        )
        self.figure.canvas.draw_idle()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--start-sample-id", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = load_illegal_samples(args.input)
    browser = IllegalSampleBrowser(samples)
    if args.start_sample_id is not None:
        matches = samples.index[samples["sample_id"] == args.start_sample_id]
        if len(matches) == 0:
            raise ValueError(f"sample_id {args.start_sample_id} is not selected")
        browser.index = int(matches[0])
        browser.draw_current()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
