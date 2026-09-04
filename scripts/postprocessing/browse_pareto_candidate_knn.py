"""Inspect geometric k-nearest neighbours of shortlisted Pareto candidates."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon, box


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.geometry import shapely_antenna_model  # noqa: E402
from scripts.postprocessing.browse_pareto_candidates import (  # noqa: E402
    DEFAULT_INPUT_PATH,
    DEFAULT_RETURN_LOSS_THRESHOLD_DB,
    load_candidate_shortlist,
)


DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "pareto_candidate_knn_k5.csv"
)
DEFAULT_K = 5


def _polygon(points: Sequence[Sequence[float]]) -> Polygon:
    return Polygon([(float(x_value), float(y_value)) for x_value, y_value in points])


def normalized_conductor(row: dict[str, Any]) -> tuple[Any, tuple[float, ...]]:
    """Return the final conductor normalized to its own substrate rectangle."""

    parameters = shapely_antenna_model.parameters_from_mapping(row)
    vertices = shapely_antenna_model.polygon_export_payload(parameters)["vertices"]
    patch = _polygon(vertices["Patch"])
    slot = _polygon(vertices["Slot"])
    feed_pin = _polygon(vertices["CPW_Feed_Pin"])
    conductor = patch.difference(slot).union(feed_pin)

    all_points = [
        (float(x_value), float(y_value))
        for points in vertices.values()
        for x_value, y_value in points
    ]
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0.0 or height <= 0.0:
        raise ValueError("candidate substrate has a non-positive extent")

    centered = affinity.translate(
        conductor,
        xoff=-(min_x + max_x) / 2.0,
        yoff=-min_y,
    )
    normalized = affinity.scale(
        centered,
        xfact=1.0 / width,
        yfact=1.0 / height,
        origin=(0.0, 0.0),
    )
    return normalized, (-0.5, 0.0, 0.5, 1.0)


def geometric_jaccard_distance(first: Any, second: Any) -> float:
    """Area Jaccard distance, zero for identical and one for disjoint shapes."""

    union_area = float(first.union(second).area)
    if union_area <= 0.0:
        raise ValueError("cannot compare empty conductor geometries")
    intersection_area = float(first.intersection(second).area)
    distance = 1.0 - intersection_area / union_area
    return min(1.0, max(0.0, distance))


def pairwise_geometry_distances(candidates: pd.DataFrame) -> np.ndarray:
    if candidates.empty:
        raise ValueError("candidate shortlist is empty")
    geometries = [
        normalized_conductor(row)[0]
        for row in candidates.to_dict(orient="records")
    ]
    distances = np.zeros((len(geometries), len(geometries)), dtype=float)
    for first_index, first in enumerate(geometries):
        for second_index in range(first_index + 1, len(geometries)):
            distance = geometric_jaccard_distance(first, geometries[second_index])
            distances[first_index, second_index] = distance
            distances[second_index, first_index] = distance
    return distances


def nearest_neighbor_indices(distances: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(distances, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.isfinite(values).all():
        raise ValueError("distance matrix contains non-finite values")
    if not 1 <= int(k) < len(values):
        raise ValueError(f"k must be in [1, {len(values) - 1}]")

    adjusted = values.copy()
    np.fill_diagonal(adjusted, np.inf)
    return np.argsort(adjusted, axis=1, kind="stable")[:, : int(k)]


def build_neighbor_table(
    candidates: pd.DataFrame,
    distances: np.ndarray,
    neighbors: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_index, neighbor_indices in enumerate(neighbors):
        query = candidates.iloc[query_index]
        for order, neighbor_index in enumerate(neighbor_indices, start=1):
            neighbor = candidates.iloc[int(neighbor_index)]
            rows.append(
                {
                    "query_rank": int(query["candidate_rank"]),
                    "query_source": str(query["source"]),
                    "query_case_id": str(query["case_id"]),
                    "neighbor_order": order,
                    "neighbor_rank": int(neighbor["candidate_rank"]),
                    "neighbor_source": str(neighbor["source"]),
                    "neighbor_case_id": str(neighbor["case_id"]),
                    "geometry_jaccard_distance": float(
                        distances[query_index, int(neighbor_index)]
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_neighbor_table(table: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _fill_geometry(axes: Any, geometry: Any, *, color: str) -> None:
    if geometry.geom_type == "Polygon":
        x_values, y_values = geometry.exterior.xy
        axes.fill(
            x_values,
            y_values,
            color=color,
            alpha=0.82,
            linewidth=1.2,
            edgecolor="#3f3f46",
        )
        return
    for part in geometry.geoms:
        _fill_geometry(axes, part, color=color)


def _short_source(source: str) -> str:
    for prefix in ("msabp-krvea-11var-", "doe-11var-branch-up-lhs-"):
        if source.startswith(prefix):
            return source[len(prefix) :]
    return source


class ParetoKnnBrowser:
    """Show one query candidate and its k closest normalized silhouettes."""

    def __init__(
        self,
        candidates: pd.DataFrame,
        distances: np.ndarray,
        neighbors: np.ndarray,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._plt = plt
        self.candidates = candidates.reset_index(drop=True)
        self.distances = distances
        self.neighbors = neighbors
        self.index = 0
        self.geometries = [
            normalized_conductor(row)[0]
            for row in self.candidates.to_dict(orient="records")
        ]

        panel_count = neighbors.shape[1] + 1
        self.columns = min(3, panel_count)
        self.rows = math.ceil(panel_count / self.columns)
        self.figure, axes = plt.subplots(
            self.rows,
            self.columns,
            figsize=(4.6 * self.columns, 3.75 * self.rows + 1.15),
            squeeze=False,
        )
        self.axes = list(axes.flat)
        self.figure.subplots_adjust(
            left=0.045,
            right=0.985,
            bottom=0.13,
            top=0.86,
            wspace=0.16,
            hspace=0.30,
        )
        self.previous_button = Button(
            self.figure.add_axes((0.35, 0.035, 0.13, 0.055)),
            "Previous  [Left]",
        )
        self.next_button = Button(
            self.figure.add_axes((0.52, 0.035, 0.13, 0.055)),
            "Next  [Right]",
        )
        self.previous_button.on_clicked(self.previous)
        self.next_button.on_clicked(self.next)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.draw_current()

    def previous(self, _event: Any = None) -> None:
        self.index = (self.index - 1) % len(self.candidates)
        self.draw_current()

    def next(self, _event: Any = None) -> None:
        self.index = (self.index + 1) % len(self.candidates)
        self.draw_current()

    def _on_key(self, event: Any) -> None:
        if event.key in {"left", "pageup"}:
            self.previous()
        elif event.key in {"right", "pagedown"}:
            self.next()
        elif event.key == "home":
            self.index = 0
            self.draw_current()
        elif event.key == "end":
            self.index = len(self.candidates) - 1
            self.draw_current()

    def _draw_candidate(
        self,
        axes: Any,
        candidate_index: int,
        *,
        order: int | None,
        distance: float | None,
    ) -> None:
        row = self.candidates.iloc[candidate_index]
        color = "#d4a72c" if order is None else "#2a9d8f"
        _fill_geometry(axes, self.geometries[candidate_index], color=color)
        substrate = box(-0.5, 0.0, 0.5, 1.0)
        x_values, y_values = substrate.exterior.xy
        axes.plot(x_values, y_values, color="#64748b", linewidth=0.9)
        axes.axvline(0.0, color="#94a3b8", linewidth=0.65, linestyle=":")
        axes.set_xlim(-0.54, 0.54)
        axes.set_ylim(-0.04, 1.04)
        axes.set_aspect("equal", adjustable="box")
        axes.axis("off")

        rank = int(row["candidate_rank"])
        identifier = f"#{rank}  {_short_source(str(row['source']))} / {row['case_id']}"
        if order is None:
            title = f"QUERY\n{identifier}"
        else:
            title = f"NN {order}   distance={distance:.5f}\n{identifier}"
        axes.set_title(title, fontsize=9.0, color="#7c2d12" if order is None else "#0f172a")

    def draw_current(self) -> None:
        for axes in self.axes:
            axes.clear()
            axes.axis("off")

        self._draw_candidate(
            self.axes[0],
            self.index,
            order=None,
            distance=None,
        )
        for order, neighbor_index in enumerate(self.neighbors[self.index], start=1):
            neighbor_index = int(neighbor_index)
            self._draw_candidate(
                self.axes[order],
                neighbor_index,
                order=order,
                distance=float(self.distances[self.index, neighbor_index]),
            )

        row = self.candidates.iloc[self.index]
        self.figure.suptitle(
            "Geometry kNN on substrate-normalized conductor silhouettes\n"
            f"Candidate {int(row['candidate_rank'])}/{len(self.candidates)} | "
            f"RL={float(row['return_loss_db']):.3f} dB | "
            f"Tot_Eff={float(row['mean_total_efficiency_linear']):.4f} | "
            f"Area={float(row['normalized_substrate_area']):.4f} | "
            f"Cap={float(row['cap_realized_gain_dbi']):.3f} dBi",
            fontsize=11.0,
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        self._plt.show()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--return-loss-threshold-db",
        type=float,
        default=DEFAULT_RETURN_LOSS_THRESHOLD_DB,
    )
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--no-gui", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidates, front_size = load_candidate_shortlist(
        args.input,
        return_loss_threshold_db=args.return_loss_threshold_db,
    )
    distances = pairwise_geometry_distances(candidates)
    neighbors = nearest_neighbor_indices(distances, args.k)
    table = build_neighbor_table(candidates, distances, neighbors)
    output_path = write_neighbor_table(table, args.output)
    print(
        f"Pareto front={front_size}; candidates={len(candidates)}; "
        f"k={args.k}; neighbor rows={len(table)}"
    )
    print(f"Neighbor table: {output_path}")
    if args.no_gui:
        return 0
    if not 1 <= args.start_rank <= len(candidates):
        raise ValueError(
            f"start rank must be in [1, {len(candidates)}], got {args.start_rank}"
        )
    browser = ParetoKnnBrowser(candidates, distances, neighbors)
    browser.index = args.start_rank - 1
    browser.draw_current()
    browser.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
