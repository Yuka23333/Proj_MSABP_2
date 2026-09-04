"""Browse four-objective Pareto candidates with return loss above a threshold.

The default input is the final Stage-3 K-RVEA history snapshot.  Running this
file directly refreshes the shortlist CSV and opens a two-button Matplotlib
browser.  Left/right arrow keys provide the same circular navigation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.geometry import shapely_antenna_model  # noqa: E402


DEFAULT_INPUT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "raw"
    / "msabp-krvea-11var-stage3-feasible-32-007"
    / "_krvea"
    / "history_observations.csv"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "pareto_candidates_return_loss_gt_7db.csv"
)
DEFAULT_RETURN_LOSS_THRESHOLD_DB = 7.0

MINIMIZATION_COLUMNS = (
    "worst_s11_linear_amplitude",
    "one_minus_mean_total_efficiency_linear",
    "normalized_substrate_area",
    "cap_realized_gain_linear",
)
METRIC_COLUMNS = (
    "return_loss_db",
    "mean_total_efficiency_linear",
    "normalized_substrate_area",
    "cap_realized_gain_dbi",
)
IDENTITY_COLUMNS = ("source", "case_id", "case_directory")


def _boolean_series(values: pd.Series) -> pd.Series:
    """Interpret bool-like CSV values without treating ``"False"`` as true."""

    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.casefold().isin(
        {"true", "1", "yes", "y"}
    )


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Return the strict Pareto mask for an all-minimization objective array."""

    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("objectives must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("objectives contain non-finite values")

    selected = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        weakly_better = np.all(values <= point, axis=1)
        strictly_better = np.any(values < point, axis=1)
        if np.any(weakly_better & strictly_better):
            selected[index] = False
    return selected


def build_candidate_shortlist(
    observations: pd.DataFrame,
    *,
    return_loss_threshold_db: float = DEFAULT_RETURN_LOSS_THRESHOLD_DB,
) -> tuple[pd.DataFrame, int]:
    """Build the sorted RL-filtered shortlist from completed observations.

    Returns ``(shortlist, pareto_front_size)``.  Four-objective dominance is
    evaluated as min(worst |S11|), max(mean Tot_Eff), min(normalized area), and
    min(cap-average realized gain).
    """

    threshold = float(return_loss_threshold_db)
    if not math.isfinite(threshold):
        raise ValueError("return-loss threshold must be finite")

    required = {
        "status",
        "is_penalty",
        *IDENTITY_COLUMNS,
        *MINIMIZATION_COLUMNS,
        "mean_total_efficiency_linear",
        "cap_realized_gain_dbi",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observation table is missing columns: {sorted(missing)}")

    completed = observations["status"].astype(str).str.casefold().eq("completed")
    penalty = _boolean_series(observations["is_penalty"])
    usable = observations.loc[completed & ~penalty].copy()
    if usable.empty:
        raise ValueError("observation table contains no completed non-penalty rows")

    # Optimization histories contain only active variables. Complete the
    # geometry mapping with explicit fixed defaults; present columns win.
    default_parameters = asdict(shapely_antenna_model.DEFAULT_PARAMETERS)
    for name in shapely_antenna_model.PARAMETER_NAMES:
        if name not in usable.columns:
            usable[name] = default_parameters[name]

    numeric_columns = {
        *MINIMIZATION_COLUMNS,
        "mean_total_efficiency_linear",
        "cap_realized_gain_dbi",
        *shapely_antenna_model.PARAMETER_NAMES,
    }
    usable.loc[:, list(numeric_columns)] = usable.loc[
        :, list(numeric_columns)
    ].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(usable.loc[:, list(numeric_columns)].to_numpy(float)).all(
        axis=1
    )
    usable = usable.loc[finite].copy()
    if usable.empty:
        raise ValueError("no observation has finite objectives and geometry parameters")

    front_mask = nondominated_mask(
        usable.loc[:, MINIMIZATION_COLUMNS].to_numpy(float)
    )
    front = usable.loc[front_mask].copy()
    front["return_loss_db"] = -20.0 * np.log10(
        front["worst_s11_linear_amplitude"].to_numpy(float)
    )
    shortlist = front.loc[front["return_loss_db"] > threshold].copy()
    shortlist.sort_values(
        ["return_loss_db", "source", "case_id"],
        ascending=(False, True, True),
        kind="stable",
        inplace=True,
    )
    shortlist.reset_index(drop=True, inplace=True)
    shortlist.insert(0, "candidate_rank", np.arange(1, len(shortlist) + 1))

    # normalized_substrate_area is both a displayed metric and a native
    # minimization objective. Preserve first occurrence order while avoiding
    # duplicate CSV headers (which pandas otherwise reads back as a Series).
    output_columns = tuple(
        dict.fromkeys(
            (
                "candidate_rank",
                *IDENTITY_COLUMNS,
                *METRIC_COLUMNS,
                *MINIMIZATION_COLUMNS,
                *shapely_antenna_model.PARAMETER_NAMES,
            )
        )
    )
    return shortlist.loc[:, output_columns], len(front)


def load_candidate_shortlist(
    path: str | Path = DEFAULT_INPUT_PATH,
    *,
    return_loss_threshold_db: float = DEFAULT_RETURN_LOSS_THRESHOLD_DB,
) -> tuple[pd.DataFrame, int]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"K-RVEA history observation CSV not found: {source}")
    observations = pd.read_csv(
        source,
        dtype={"source": "string", "case_id": "string", "case_directory": "string"},
    )
    return build_candidate_shortlist(
        observations,
        return_loss_threshold_db=return_loss_threshold_db,
    )


def write_shortlist(shortlist: pd.DataFrame, path: str | Path) -> Path:
    """Atomically persist the ordered shortlist."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shortlist.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _closed_polygon(points: Sequence[Sequence[float]]) -> Polygon:
    return Polygon([(float(x_value), float(y_value)) for x_value, y_value in points])


def _fill_geometry(axes: Any, geometry: Any, **style: Any) -> None:
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        x_values, y_values = geometry.exterior.xy
        axes.fill(x_values, y_values, **style)
        return
    for part in geometry.geoms:
        _fill_geometry(axes, part, **style)


class ParetoCandidateBrowser:
    """Two-button and keyboard browser for the ordered candidate shortlist."""

    def __init__(self, candidates: pd.DataFrame, threshold_db: float) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        if candidates.empty:
            raise ValueError("candidate shortlist is empty")
        self._plt = plt
        self.candidates = candidates.reset_index(drop=True)
        self.threshold_db = float(threshold_db)
        self.index = 0

        self.figure = plt.figure(figsize=(14.8, 8.4))
        self.axes = self.figure.add_axes((0.055, 0.15, 0.65, 0.76))
        self.info_axes = self.figure.add_axes((0.73, 0.08, 0.26, 0.86))
        self.previous_button = Button(
            self.figure.add_axes((0.35, 0.035, 0.13, 0.06)),
            "Previous  [Left]",
        )
        self.next_button = Button(
            self.figure.add_axes((0.52, 0.035, 0.13, 0.06)),
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

    def draw_current(self) -> None:
        row = self.candidates.iloc[self.index]
        parameters = shapely_antenna_model.parameters_from_mapping(row.to_dict())
        payload = shapely_antenna_model.polygon_export_payload(parameters)
        vertices = payload["vertices"]

        patch = _closed_polygon(vertices["Patch"])
        slot = _closed_polygon(vertices["Slot"])
        feed_pin = _closed_polygon(vertices["CPW_Feed_Pin"])
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
        substrate = box(min_x, min_y, max_x, max_y)

        self.axes.clear()
        self.info_axes.clear()
        self.info_axes.axis("off")

        _fill_geometry(
            self.axes,
            substrate,
            color="#dbeafe",
            alpha=0.42,
            zorder=0,
            label="substrate extent",
        )
        _fill_geometry(
            self.axes,
            conductor,
            color="#d4a72c",
            alpha=0.78,
            zorder=2,
            label="final conductor",
        )
        for geometry, color, label, linestyle in (
            (patch, "#8a5a00", "Patch source curve", "-"),
            (slot, "#dc2626", "Slot source curve", "--"),
            (feed_pin, "#2563eb", "CPW feed pin", "-"),
        ):
            x_values, y_values = geometry.exterior.xy
            self.axes.plot(
                x_values,
                y_values,
                color=color,
                linewidth=1.35,
                linestyle=linestyle,
                zorder=4,
                label=label,
            )

        x_span = max_x - min_x
        y_span = max_y - min_y
        self.axes.set_xlim(min_x - 0.05 * x_span, max_x + 0.05 * x_span)
        self.axes.set_ylim(min_y - 0.05 * y_span, max_y + 0.05 * y_span)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
        self.axes.axvline(0.0, color="#64748b", linewidth=0.8, linestyle=":")
        self.axes.set_xlabel("X (mm)")
        self.axes.set_ylabel("Y (mm)")
        self.axes.legend(loc="upper center", ncol=4, fontsize=8)

        source = str(row["source"])
        case_id = str(row["case_id"])
        rank = int(row["candidate_rank"])
        self.axes.set_title(
            f"Candidate {rank}/{len(self.candidates)}  |  {source} / {case_id}",
            fontsize=10.5,
            pad=10,
        )

        metrics = (
            f"Return loss       {float(row['return_loss_db']):9.4f} dB   higher is better",
            f"Mean Tot_Eff      {float(row['mean_total_efficiency_linear']):9.6f}      higher is better",
            f"Normalized area   {float(row['normalized_substrate_area']):9.6f}      lower is better",
            f"Cap gain          {float(row['cap_realized_gain_dbi']):9.4f} dBi  lower is better",
        )
        parameter_lines = tuple(
            f"{name:<30} {float(row[name]): .7g}"
            for name in shapely_antenna_model.PARAMETER_NAMES
        )
        information = (
            f"Candidate {rank} of {len(self.candidates)}",
            f"RL threshold: > {self.threshold_db:g} dB",
            "",
            source,
            case_id,
            "",
            "Four objectives:",
            *metrics,
            "",
            "Geometry parameters:",
            *parameter_lines,
            "",
            "Keys: Left/Right, PgUp/PgDn, Home/End",
        )
        self.info_axes.text(
            0.0,
            1.0,
            "\n".join(information),
            ha="left",
            va="top",
            family="monospace",
            fontsize=7.45,
            color="#0f172a",
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        self._plt.show()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--return-loss-threshold-db",
        type=float,
        default=DEFAULT_RETURN_LOSS_THRESHOLD_DB,
    )
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="refresh the shortlist CSV without opening the browser",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    shortlist, front_size = load_candidate_shortlist(
        args.input,
        return_loss_threshold_db=args.return_loss_threshold_db,
    )
    output_path = write_shortlist(shortlist, args.output)
    print(
        f"Pareto front={front_size}; return loss > "
        f"{args.return_loss_threshold_db:g} dB: {len(shortlist)} candidates"
    )
    print(f"Shortlist: {output_path}")
    if args.no_gui:
        return 0
    if shortlist.empty:
        raise ValueError("no Pareto candidate passes the return-loss threshold")
    if not 1 <= args.start_rank <= len(shortlist):
        raise ValueError(
            f"start rank must be in [1, {len(shortlist)}], got {args.start_rank}"
        )
    browser = ParetoCandidateBrowser(shortlist, args.return_loss_threshold_db)
    browser.index = args.start_rank - 1
    browser.draw_current()
    browser.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
