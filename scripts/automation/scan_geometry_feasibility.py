"""Scan feasible-geometry share while widening the five ordinary mm margins.

The main-slot special ranges and all sixteen ratio ranges in
``antenna_sampling.json`` are preserved.  The other five absolute variables are
sampled around nominal, first at 95--105 percent and then five percentage
points wider on each side.

Running this file with IDE F5 displays the saved curve.  Command-line runs save
the CSV and PNG without opening a window unless ``--show`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import antenna_sampler  # noqa: E402
from scripts.automation import sample_geometry_labels  # noqa: E402


SCENARIO = "redesigned_23d"
SAMPLES_PER_WINDOW = 2_048
WORKER_COUNT = 24
CHUNK_SIZE = 64
START_HALF_WIDTH_PERCENT = 5.0
STOP_HALF_WIDTH_PERCENT = 95.0
STEP_HALF_WIDTH_PERCENT = 5.0
DEFAULT_OUTPUT_TABLE_DIRECTORY = REPOSITORY_ROOT / "results" / "processed"
DEFAULT_OUTPUT_FIGURE_DIRECTORY = REPOSITORY_ROOT / "results" / "figures"
PERCENTAGE_SCANNED_PARAMETER_NAMES = (
    "PATCH_BRICK_1_SIDE_MARGIN",
    "PATCH_BRICK_1_TOP_MARGIN",
    "PATCH_BRICK_3_BOTTOM_MARGIN",
    "PATCH_BRICK_2_HEIGHT_MARGIN",
    "PATCH_BRICK_4_MARGIN",
)


def build_half_width_percentages(
    start: float = START_HALF_WIDTH_PERCENT,
    stop: float = STOP_HALF_WIDTH_PERCENT,
    step: float = STEP_HALF_WIDTH_PERCENT,
) -> tuple[float, ...]:
    """Return an inclusive, exactly stepped sequence of percentage half-widths."""

    values = (float(start), float(stop), float(step))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scan limits must be finite")
    if start <= 0.0 or stop < start or step <= 0.0:
        raise ValueError("require 0 < start <= stop and step > 0")
    interval_count = (stop - start) / step
    rounded_count = round(interval_count)
    if not math.isclose(interval_count, rounded_count, abs_tol=1e-12):
        raise ValueError("step must divide the inclusive start-to-stop interval")
    return tuple(float(start + index * step) for index in range(rounded_count + 1))


def configure_scan_window(
    config: Mapping[str, Any],
    scenario: str,
    half_width_percent: float,
) -> dict[str, Any]:
    """Apply one nominal-centred range to the five ordinary mm margins."""

    if not math.isfinite(half_width_percent) or not 0.0 < half_width_percent < 100.0:
        raise ValueError("half_width_percent must be finite and inside (0, 100)")
    configured = sample_geometry_labels.configure_scenario(
        copy.deepcopy(dict(config)),
        scenario,
    )
    relative_half_width = half_width_percent / 100.0
    sampling = configured.setdefault("sampling", {})
    parameters = sampling.setdefault("parameters", {})
    for name in PERCENTAGE_SCANNED_PARAMETER_NAMES:
        parameter = dict(parameters[name])
        parameter["range"] = {
            "mode": "relative",
            "lower": -relative_half_width,
            "upper": relative_half_width,
            "reference": "nominal",
        }
        parameters[name] = parameter
    return configured


def globally_scanned_parameter_names(
    plan: antenna_sampler.SamplingPlan,
) -> tuple[str, ...]:
    """Return effective variables controlled by the widening scan window."""

    return tuple(
        item.spec.name
        for item in plan.resolved_parameters
        if item.effective_sample and item.spec.name in PERCENTAGE_SCANNED_PARAMETER_NAMES
    )


def _atomic_save_table(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, float_format="%.17g")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def plot_feasible_fraction(
    results: pd.DataFrame,
    output_path: str | Path,
    *,
    show: bool = False,
) -> Path:
    """Save the feasible-geometry percentage curve and optionally display it."""

    required = {"half_width_percent", "feasible_percent"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"scan results missing columns: {sorted(missing)}")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    x_values = results["half_width_percent"].to_numpy(dtype=float)
    y_values = results["feasible_percent"].to_numpy(dtype=float)
    axes.plot(
        x_values,
        y_values,
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        markersize=6.0,
    )
    axes.fill_between(x_values, 0.0, y_values, color="#93c5fd", alpha=0.28)
    axes.set_xticks(x_values)
    axes.set_xticklabels(
        [
            f"[{100.0 - half_width:g}, {100.0 + half_width:g}]%"
            for half_width in x_values
        ],
        rotation=35,
        ha="right",
    )
    axes.set_xlabel("Nominal-relative range of not-yet-relative variables")
    axes.set_ylabel("Feasible geometry (%)")
    axes.set_title("Feasible geometry share versus sampling-range width")
    axes.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.75)
    axes.set_axisbelow(True)
    upper_limit = max(1.0, min(100.0, float(np.max(y_values)) * 1.15 + 0.25))
    axes.set_ylim(0.0, upper_limit)
    for x_value, y_value in zip(x_values, y_values, strict=True):
        axes.annotate(
            f"{y_value:.2f}%",
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp{output.suffix}")
    try:
        figure.savefig(temporary, dpi=180)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output


def run_scan(
    *,
    scenario: str = SCENARIO,
    samples_per_window: int = SAMPLES_PER_WINDOW,
    workers: int = WORKER_COUNT,
    chunk_size: int = CHUNK_SIZE,
    start_half_width_percent: float = START_HALF_WIDTH_PERCENT,
    stop_half_width_percent: float = STOP_HALF_WIDTH_PERCENT,
    step_half_width_percent: float = STEP_HALF_WIDTH_PERCENT,
    config_path: str | Path = antenna_sampler.DEFAULT_CONFIG_PATH,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample all scan windows, label them in one worker pool, and summarize."""

    if samples_per_window <= 0:
        raise ValueError("samples_per_window must be positive")
    half_widths = build_half_width_percentages(
        start_half_width_percent,
        stop_half_width_percent,
        step_half_width_percent,
    )
    base_config = antenna_sampler.load_sampling_config(config_path)
    plans: list[antenna_sampler.SamplingPlan] = []
    frames: list[pd.DataFrame] = []
    for half_width in half_widths:
        configured = configure_scan_window(base_config, scenario, half_width)
        resolve_kwargs: dict[str, Any] = {"n_samples": samples_per_window}
        if seed is not None:
            resolve_kwargs["seed"] = seed
        plan = antenna_sampler.resolve_sampling_plan(configured, **resolve_kwargs)
        plans.append(plan)
        frames.append(antenna_sampler.generate_parameter_frame(plan))

    scanned_names = globally_scanned_parameter_names(plans[0])
    active_dimension_count = sum(
        item.effective_sample for item in plans[0].resolved_parameters
    )
    for plan in plans[1:]:
        if globally_scanned_parameter_names(plan) != scanned_names:
            raise RuntimeError("globally scanned parameter set changed between windows")
        if plan.geometry_policy != plans[0].geometry_policy:
            raise RuntimeError("geometry policy changed between scan windows")

    print(
        f"[feasibility-scan] scenario={scenario} windows={len(plans)} "
        f"samples/window={samples_per_window} total={len(plans) * samples_per_window} "
        f"dimensions={active_dimension_count} workers={workers}",
        flush=True,
    )
    print(
        f"[feasibility-scan] widening {len(scanned_names)} parameters: "
        + ", ".join(scanned_names),
        flush=True,
    )

    combined_frame = pd.concat(frames, ignore_index=True)
    combined_labels = sample_geometry_labels.label_parameter_frame_parallel(
        combined_frame,
        plans[0],
        workers=workers,
        chunk_size=chunk_size,
    )
    rows: list[dict[str, Any]] = []
    cursor = 0
    for half_width, plan in zip(half_widths, plans, strict=True):
        window_labels = combined_labels[cursor : cursor + samples_per_window]
        cursor += samples_per_window
        valid_count = int(np.count_nonzero(window_labels))
        feasible_fraction = valid_count / samples_per_window
        rows.append(
            {
                "scenario": scenario,
                "half_width_percent": half_width,
                "range_lower_percent": 100.0 - half_width,
                "range_upper_percent": 100.0 + half_width,
                "sample_count": samples_per_window,
                "valid_count": valid_count,
                "invalid_count": samples_per_window - valid_count,
                "feasible_fraction": feasible_fraction,
                "feasible_percent": 100.0 * feasible_fraction,
                "active_dimension_count": active_dimension_count,
                "globally_scanned_parameter_count": len(scanned_names),
                "method": plan.method,
                "seed": plan.seed,
            }
        )
        print(
            f"[feasibility-scan] range=[{100.0 - half_width:g}%, "
            f"{100.0 + half_width:g}%] valid={valid_count}/{samples_per_window} "
            f"share={100.0 * feasible_fraction:.4f}%",
            flush=True,
        )
    return pd.DataFrame(rows)


def _running_from_ide_f5() -> bool:
    if sys.gettrace() is not None:
        return True
    return any(
        os.environ.get(name)
        for name in ("PYCHARM_HOSTED", "SPYDER_KERNEL_ID", "IDLESTARTUP")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=tuple(sample_geometry_labels.SCENARIO_BRANCH_STATES),
        default=SCENARIO,
    )
    parser.add_argument("--samples-per-window", type=int, default=SAMPLES_PER_WINDOW)
    parser.add_argument("--workers", type=int, default=WORKER_COUNT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--start-half-width-percent",
        type=float,
        default=START_HALF_WIDTH_PERCENT,
    )
    parser.add_argument(
        "--stop-half-width-percent",
        type=float,
        default=STOP_HALF_WIDTH_PERCENT,
    )
    parser.add_argument(
        "--step-half-width-percent",
        type=float,
        default=STEP_HALF_WIDTH_PERCENT,
    )
    parser.add_argument("--config", type=Path, default=antenna_sampler.DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-table", type=Path)
    parser.add_argument("--output-figure", type=Path)
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Display the plot; defaults to yes under IDE F5 and no in a terminal.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    results = run_scan(
        scenario=args.scenario,
        samples_per_window=args.samples_per_window,
        workers=args.workers,
        chunk_size=args.chunk_size,
        start_half_width_percent=args.start_half_width_percent,
        stop_half_width_percent=args.stop_half_width_percent,
        step_half_width_percent=args.step_half_width_percent,
        config_path=args.config,
        seed=args.seed,
    )
    output_table = args.output_table or (
        DEFAULT_OUTPUT_TABLE_DIRECTORY
        / f"geometry_feasibility_scan_{args.scenario}.csv"
    )
    output_figure = args.output_figure or (
        DEFAULT_OUTPUT_FIGURE_DIRECTORY
        / f"geometry_feasibility_scan_{args.scenario}.png"
    )
    saved_table = _atomic_save_table(results, output_table)
    saved_figure = plot_feasible_fraction(
        results,
        output_figure,
        show=_running_from_ide_f5() if args.show is None else args.show,
    )
    print(f"[feasibility-scan] table={saved_table}", flush=True)
    print(f"[feasibility-scan] figure={saved_figure}", flush=True)
    return results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
