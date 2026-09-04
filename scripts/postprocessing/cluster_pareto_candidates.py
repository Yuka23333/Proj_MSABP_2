"""Reduce the RL-filtered Pareto shortlist with deterministic k-medoids."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from shapely.geometry import box


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing.browse_pareto_candidate_knn import (  # noqa: E402
    normalized_conductor,
    pairwise_geometry_distances,
)
from scripts.postprocessing.browse_pareto_candidates import (  # noqa: E402
    DEFAULT_INPUT_PATH,
    DEFAULT_RETURN_LOSS_THRESHOLD_DB,
    load_candidate_shortlist,
)


DEFAULT_CLUSTER_COUNT = 12
DEFAULT_ASSIGNMENTS_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "pareto_candidate_clusters_k12.csv"
)
DEFAULT_SELECTED_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "pareto_candidates_selected_k12.csv"
)
DEFAULT_FIGURE_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "figures"
    / "pareto_candidate_cluster_medoids_k12.png"
)


def _validate_distance_matrix(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("distance matrix must be square")
    if len(values) == 0:
        raise ValueError("distance matrix must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("distance matrix contains non-finite values")
    if np.any(values < -1e-12):
        raise ValueError("distance matrix contains negative distances")
    if not np.allclose(values, values.T, atol=1e-12, rtol=0.0):
        raise ValueError("distance matrix must be symmetric")
    if not np.allclose(np.diag(values), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("distance matrix diagonal must be zero")
    return values


def _assignment_cost(distances: np.ndarray, medoids: Sequence[int]) -> float:
    return float(np.min(distances[:, np.asarray(medoids, dtype=int)], axis=1).sum())


def _pam_build_initialization(distances: np.ndarray, k: int) -> list[int]:
    """PAM BUILD initialization with stable index tie-breaking."""

    first = int(np.argmin(distances.sum(axis=1)))
    medoids = [first]
    current_nearest = distances[:, first].copy()
    while len(medoids) < k:
        best_candidate = -1
        best_reduction = -math.inf
        for candidate in range(len(distances)):
            if candidate in medoids:
                continue
            reduction = float(
                current_nearest.sum()
                - np.minimum(current_nearest, distances[:, candidate]).sum()
            )
            if reduction > best_reduction + 1e-15:
                best_candidate = candidate
                best_reduction = reduction
        medoids.append(best_candidate)
        current_nearest = np.minimum(current_nearest, distances[:, best_candidate])
    return medoids


def pam_kmedoids(
    distances: np.ndarray,
    k: int,
    *,
    maximum_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run deterministic PAM and return medoids, labels, cost, iterations."""

    values = _validate_distance_matrix(distances)
    cluster_count = int(k)
    if not 1 <= cluster_count <= len(values):
        raise ValueError(f"k must be in [1, {len(values)}]")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")

    medoids = _pam_build_initialization(values, cluster_count)
    cost = _assignment_cost(values, medoids)
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        medoid_set = set(medoids)
        non_medoids = [index for index in range(len(values)) if index not in medoid_set]
        best_cost = cost
        best_swap: tuple[int, int] | None = None
        for position in range(len(medoids)):
            for candidate in non_medoids:
                trial = medoids.copy()
                trial[position] = candidate
                trial_cost = _assignment_cost(values, trial)
                if trial_cost < best_cost - 1e-15:
                    best_cost = trial_cost
                    best_swap = position, candidate
        if best_swap is None:
            break
        medoids[best_swap[0]] = best_swap[1]
        cost = best_cost

    # Stable cluster IDs follow candidate rank/index, not PAM insertion order.
    medoid_array = np.asarray(sorted(medoids), dtype=int)
    labels = np.argmin(values[:, medoid_array], axis=1)
    return medoid_array, labels, cost, iterations


def silhouette_score_precomputed(distances: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette from a precomputed distance matrix."""

    values = _validate_distance_matrix(distances)
    groups = np.asarray(labels, dtype=int)
    if groups.shape != (len(values),):
        raise ValueError("labels must contain one cluster ID per sample")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0.0

    silhouettes = np.zeros(len(values), dtype=float)
    for index in range(len(values)):
        own_members = np.flatnonzero(groups == groups[index])
        own_others = own_members[own_members != index]
        if len(own_others) == 0:
            silhouettes[index] = 0.0
            continue
        within = float(values[index, own_others].mean())
        nearest_other = min(
            float(values[index, groups == group].mean())
            for group in unique_groups
            if group != groups[index]
        )
        denominator = max(within, nearest_other)
        silhouettes[index] = (
            0.0 if denominator == 0.0 else (nearest_other - within) / denominator
        )
    return float(silhouettes.mean())


def build_cluster_tables(
    candidates: pd.DataFrame,
    distances: np.ndarray,
    medoids: np.ndarray,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the annotated 36-row table and the 12-row medoid table."""

    assignments = candidates.copy()
    assignments.insert(1, "cluster_id", labels + 1)
    assignments.insert(2, "is_selected_medoid", False)
    assignments.insert(3, "cluster_medoid_rank", 0)
    assignments.insert(4, "distance_to_medoid", 0.0)
    assignments.insert(5, "cluster_size", 0)

    for cluster_index, medoid_index in enumerate(medoids):
        members = np.flatnonzero(labels == cluster_index)
        medoid_rank = int(candidates.iloc[int(medoid_index)]["candidate_rank"])
        assignments.loc[members, "cluster_medoid_rank"] = medoid_rank
        assignments.loc[members, "distance_to_medoid"] = distances[
            members, int(medoid_index)
        ]
        assignments.loc[members, "cluster_size"] = len(members)
        assignments.loc[int(medoid_index), "is_selected_medoid"] = True

    selected = assignments.loc[assignments["is_selected_medoid"]].copy()
    selected.sort_values("cluster_id", inplace=True)
    selected.reset_index(drop=True, inplace=True)
    return assignments, selected


def _write_csv_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _fill_geometry(axes: Any, geometry: Any, color: Any) -> None:
    if geometry.geom_type == "Polygon":
        x_values, y_values = geometry.exterior.xy
        axes.fill(
            x_values,
            y_values,
            color=color,
            alpha=0.82,
            edgecolor="#27272a",
            linewidth=0.9,
        )
        return
    for part in geometry.geoms:
        _fill_geometry(axes, part, color)


def plot_selected_medoids(
    selected: pd.DataFrame,
    path: str | Path,
    *,
    show: bool,
) -> Path:
    import matplotlib.pyplot as plt

    columns = 4
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.8 * columns, 3.65 * rows),
        squeeze=False,
    )
    palette = plt.get_cmap("tab20")
    for axes_item in axes.flat:
        axes_item.axis("off")

    for plot_index, (_, row) in enumerate(selected.iterrows()):
        axes_item = axes.flat[plot_index]
        geometry, _ = normalized_conductor(row.to_dict())
        color = palette(plot_index % 20)
        _fill_geometry(axes_item, geometry, color)
        substrate = box(-0.5, 0.0, 0.5, 1.0)
        x_values, y_values = substrate.exterior.xy
        axes_item.plot(x_values, y_values, color="#64748b", linewidth=0.8)
        axes_item.axvline(0.0, color="#94a3b8", linewidth=0.6, linestyle=":")
        axes_item.set_xlim(-0.54, 0.54)
        axes_item.set_ylim(-0.04, 1.04)
        axes_item.set_aspect("equal", adjustable="box")
        axes_item.set_title(
            f"Cluster {int(row['cluster_id'])}  n={int(row['cluster_size'])}\n"
            f"Candidate #{int(row['candidate_rank'])}  {row['case_id']}\n"
            f"RL {float(row['return_loss_db']):.2f} dB | "
            f"Eff {float(row['mean_total_efficiency_linear']):.3f} | "
            f"Area {float(row['normalized_substrate_area']):.3f}",
            fontsize=8.5,
        )

    figure.suptitle(
        "12 geometry k-medoids retained from 36 Pareto candidates",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.96))
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument(
        "--return-loss-threshold-db",
        type=float,
        default=DEFAULT_RETURN_LOSS_THRESHOLD_DB,
    )
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS_PATH)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidates, front_size = load_candidate_shortlist(
        args.input,
        return_loss_threshold_db=args.return_loss_threshold_db,
    )
    distances = pairwise_geometry_distances(candidates)
    medoids, labels, cost, iterations = pam_kmedoids(distances, args.clusters)
    score = silhouette_score_precomputed(distances, labels)
    assignments, selected = build_cluster_tables(
        candidates,
        distances,
        medoids,
        labels,
    )
    assignments_path = _write_csv_atomic(assignments, args.assignments)
    selected_path = _write_csv_atomic(selected, args.selected)
    figure_path = plot_selected_medoids(selected, args.figure, show=args.show)

    print(
        f"Pareto front={front_size}; candidates={len(candidates)}; "
        f"clusters={args.clusters}; selected={len(selected)}"
    )
    print(
        f"PAM cost={cost:.9f}; iterations={iterations}; "
        f"silhouette={score:.6f}"
    )
    print(f"Assignments: {assignments_path}")
    print(f"Selected: {selected_path}")
    print(f"Figure: {figure_path}")
    print("Selected candidate ranks: " + ", ".join(map(str, selected["candidate_rank"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
