"""Sample the 23-D antenna space and inspect pairwise curve intersections.

Only boundary contact on the shared global bottom edge is ignored.  Every
other point or segment shared by Slot, Patch, or CPW_Feed_Pin is reported but
does not change the sampler's ``geometry_valid`` label.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from shapely.errors import ShapelyError
from shapely.geometry import LineString, LinearRing


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import antenna_sampler  # noqa: E402
from scripts.geometry import shapely_antenna_model  # noqa: E402


SAMPLE_COUNT = 1_024
CURVE_NAMES = ("Slot", "Patch", "CPW_Feed_Pin")
CURVE_PAIRS = tuple(combinations(CURVE_NAMES, 2))
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "samples"
    / "antenna_samples_1024_curve_intersections.csv"
)
DEFAULT_SUMMARY_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "antenna_curve_intersections_1024_summary.json"
)


def _pair_key(first: str, second: str) -> str:
    return f"{first}__{second}"


def inspect_curve_boundaries(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect one exported payload after removing its global bottom edge."""

    vertices = payload["vertices"]
    rings = {
        name: LinearRing([*vertices[name], vertices[name][0]])
        for name in CURVE_NAMES
    }
    all_points = [point for name in CURVE_NAMES for point in vertices[name]]
    bottom_y = min(float(point[1]) for point in all_points)
    min_x = min(float(point[0]) for point in all_points)
    max_x = max(float(point[0]) for point in all_points)
    bottom_edge = LineString(((min_x - 1.0, bottom_y), (max_x + 1.0, bottom_y)))

    bottom_contact_pairs: list[str] = []
    non_bottom_pairs: list[str] = []
    crossing_pairs: list[str] = []
    overlapping_pairs: list[str] = []
    touching_pairs: list[str] = []
    details: dict[str, dict[str, str]] = {}
    for first, second in CURVE_PAIRS:
        key = _pair_key(first, second)
        complete = rings[first].intersection(rings[second])
        outside_bottom = complete.difference(bottom_edge)
        if not complete.intersection(bottom_edge).is_empty:
            bottom_contact_pairs.append(key)
        if not outside_bottom.is_empty:
            non_bottom_pairs.append(key)
            first_without_bottom = rings[first].difference(bottom_edge)
            second_without_bottom = rings[second].difference(bottom_edge)
            relations: list[str] = []
            if first_without_bottom.crosses(second_without_bottom):
                crossing_pairs.append(key)
                relations.append("crosses")
            if first_without_bottom.overlaps(second_without_bottom):
                overlapping_pairs.append(key)
                relations.append("overlaps")
            if first_without_bottom.touches(second_without_bottom):
                touching_pairs.append(key)
                relations.append("touches")
            details[key] = {
                "geometry_type": outside_bottom.geom_type,
                "relations": ",".join(relations),
                "wkt": outside_bottom.wkt,
            }

    return {
        "allowed_bottom_y_mm": bottom_y,
        "bottom_contact_pairs": ";".join(bottom_contact_pairs),
        "non_bottom_intersection": bool(non_bottom_pairs),
        "non_bottom_intersection_pairs": ";".join(non_bottom_pairs),
        "non_bottom_crossing_pairs": ";".join(crossing_pairs),
        "non_bottom_overlapping_pairs": ";".join(overlapping_pairs),
        "non_bottom_touching_pairs": ";".join(touching_pairs),
        "non_bottom_intersection_details": (
            json.dumps(details, ensure_ascii=False, sort_keys=True) if details else ""
        ),
    }


def run_check(
    *,
    sample_count: int = SAMPLE_COUNT,
    config_path: str | Path = antenna_sampler.DEFAULT_CONFIG_PATH,
    method: str | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate samples and return the detailed table plus aggregate summary."""

    config = antenna_sampler.load_sampling_config(config_path)
    plan_overrides: dict[str, Any] = {
        "n_samples": sample_count,
        "method": method,
    }
    if seed is not None:
        plan_overrides["seed"] = seed
    plan = antenna_sampler.resolve_sampling_plan(config, **plan_overrides)
    sampled = antenna_sampler.generate_samples(plan).frame
    rows: list[dict[str, Any]] = []
    for row in sampled.to_dict(orient="records"):
        diagnostics: dict[str, Any] = {
            "intersection_check_completed": False,
            "intersection_check_error": "",
            "allowed_bottom_y_mm": None,
            "bottom_contact_pairs": "",
            "non_bottom_intersection": False,
            "non_bottom_intersection_pairs": "",
            "non_bottom_crossing_pairs": "",
            "non_bottom_overlapping_pairs": "",
            "non_bottom_touching_pairs": "",
            "non_bottom_intersection_details": "",
        }
        try:
            parameters = antenna_sampler.parameters_from_csv_row(row)
            payload = shapely_antenna_model.polygon_export_payload(
                parameters,
                quantize_step_mm=plan.geometry_policy.coordinate_quantum_mm,
            )
            diagnostics = inspect_curve_boundaries(payload)
            diagnostics.update(
                intersection_check_completed=True,
                intersection_check_error="",
            )
        except (TypeError, ValueError, ShapelyError) as exc:
            diagnostics["intersection_check_error"] = f"{type(exc).__name__}: {exc}"
        row.update(diagnostics)
        rows.append(row)

    frame = pd.DataFrame(rows)
    inspected = frame.loc[frame["intersection_check_completed"]]
    valid = inspected.loc[inspected["geometry_valid"]]

    def count_pairs(source: pd.DataFrame, column: str) -> dict[str, int]:
        return {
            _pair_key(first, second): int(
                source[column]
                .str.split(";")
                .map(lambda names, key=_pair_key(first, second): key in names)
                .sum()
            )
            for first, second in CURVE_PAIRS
        }

    pair_counts = count_pairs(inspected, "non_bottom_intersection_pairs")
    relation_counts = {
        relation: count_pairs(inspected, column)
        for relation, column in (
            ("crosses", "non_bottom_crossing_pairs"),
            ("overlaps", "non_bottom_overlapping_pairs"),
            ("touches", "non_bottom_touching_pairs"),
        )
    }
    summary = {
        "sample_count": int(len(frame)),
        "geometry_valid_count": int(frame["geometry_valid"].sum()),
        "geometry_invalid_count": int((~frame["geometry_valid"]).sum()),
        "intersection_check_completed_count": int(len(inspected)),
        "intersection_check_failed_count": int(
            (~frame["intersection_check_completed"]).sum()
        ),
        "samples_with_non_bottom_intersection": int(
            inspected["non_bottom_intersection"].sum()
        ),
        "samples_without_non_bottom_intersection": int(
            (~inspected["non_bottom_intersection"]).sum()
        ),
        "valid_samples_with_non_bottom_intersection": int(
            valid["non_bottom_intersection"].sum()
        ),
        "valid_samples_without_non_bottom_intersection": int(
            (~valid["non_bottom_intersection"]).sum()
        ),
        "pair_counts": pair_counts,
        "pair_relation_counts": relation_counts,
        "valid_geometry_pair_counts": count_pairs(
            valid,
            "non_bottom_intersection_pairs",
        ),
        "ignored_contact": "pairwise boundary intersection on global bottom edge",
        "sampling_method": plan.method,
        "seed": plan.seed,
        "coordinate_quantum_mm": plan.geometry_policy.coordinate_quantum_mm,
    }
    return frame, summary


def _atomic_write_outputs(
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
    *,
    output_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_path).expanduser().resolve()
    summary_output = Path(summary_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary_json = summary_output.with_name(
        f".{summary_output.name}.{os.getpid()}.tmp"
    )
    try:
        frame.to_csv(
            temporary_csv,
            index=False,
            float_format=antenna_sampler.CSV_FLOAT_FORMAT,
        )
        temporary_json.write_text(
            json.dumps(dict(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_csv, output)
        os.replace(temporary_json, summary_output)
    finally:
        if temporary_csv.exists():
            temporary_csv.unlink()
        if temporary_json.exists():
            temporary_json.unlink()
    return output, summary_output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--method", choices=("sobol", "latin"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path, default=antenna_sampler.DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame, summary = run_check(
        sample_count=args.n_samples,
        config_path=args.config,
        method=args.method,
        seed=args.seed,
    )
    output, summary_output = _atomic_write_outputs(
        frame,
        summary,
        output_path=args.output,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[curve-intersections] samples={output}")
    print(f"[curve-intersections] summary={summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
