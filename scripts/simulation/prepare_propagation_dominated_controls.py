"""Select eight reproducible dominated controls for the phantom test bench.

The frozen input is the 960-row observation snapshot after the 512-point DoE
and six K-RVEA rounds. Only completed, non-penalty designs are eligible.

Four controls isolate a poor value in one optimization objective, one control
represents poor aggregate performance, and the remaining controls are
k-medoids of the joint bad tail in objective-percentile and normalized-input
space. The result is a compact negative-control set, not merely the eight
numerically worst points.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import krvea_data  # noqa: E402
from msabp_opt.simulation.distributed.propagation_case_runner import (  # noqa: E402
    SIMULATION_MODE,
)
from scripts.automation import antenna_sampler  # noqa: E402
from scripts.geometry import shapely_antenna_model  # noqa: E402
from scripts.postprocessing.browse_pareto_candidates import (  # noqa: E402
    MINIMIZATION_COLUMNS,
    nondominated_mask,
)
from scripts.postprocessing.cluster_pareto_candidates import (  # noqa: E402
    pam_kmedoids,
)


DEFAULT_INPUT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "raw"
    / "msabp-krvea-11var-stage2-learned-64-006"
    / "_krvea"
    / "observations.csv"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "samples"
    / "propagation_dominated_controls_8.csv"
)
DEFAULT_AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "propagation_dominated_controls_8_selection.csv"
)
DEFAULT_CONTROL_COUNT = 8
OBJECTIVE_TAIL_QUANTILE = 0.75
SINGLE_OBJECTIVE_PERCENTILE = 0.90
OBJECTIVE_DISTANCE_WEIGHT = 0.70
INPUT_DISTANCE_WEIGHT = 0.30

OBJECTIVE_ROLES: tuple[tuple[str, str], ...] = (
    ("bad_s11", "worst_s11_linear_amplitude"),
    ("bad_efficiency", "one_minus_mean_total_efficiency_linear"),
    ("bad_area", "normalized_substrate_area"),
    ("bad_cap_gain", "cap_realized_gain_linear"),
)


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.casefold().isin(
        {"true", "1", "yes", "y"}
    )


def prepare_archive(observations: pd.DataFrame) -> pd.DataFrame:
    """Return finite, completed, unique 11-D designs with badness metadata."""

    required = {
        "source",
        "case_id",
        "case_directory",
        "status",
        "is_penalty",
        *MINIMIZATION_COLUMNS,
        "mean_total_efficiency_linear",
        "cap_realized_gain_dbi",
        *krvea_data.ACTIVE_PARAMETER_NAMES,
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observation table is missing columns: {sorted(missing)}")

    completed = observations["status"].astype(str).str.casefold().eq("completed")
    usable = observations.loc[
        completed & ~_bool_series(observations["is_penalty"])
    ].copy()
    if usable.empty:
        raise ValueError("observation table contains no completed non-penalty rows")

    defaults = asdict(shapely_antenna_model.DEFAULT_PARAMETERS)
    for name in shapely_antenna_model.PARAMETER_NAMES:
        if name not in usable.columns:
            usable[name] = defaults[name]

    numeric_columns = tuple(
        dict.fromkeys(
            (
                *MINIMIZATION_COLUMNS,
                "mean_total_efficiency_linear",
                "cap_realized_gain_dbi",
                *shapely_antenna_model.PARAMETER_NAMES,
            )
        )
    )
    usable.loc[:, numeric_columns] = usable.loc[:, numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite = np.isfinite(usable.loc[:, numeric_columns].to_numpy(float)).all(axis=1)
    usable = usable.loc[finite].copy()
    if usable.empty:
        raise ValueError("no completed observation has finite metrics and parameters")

    space = krvea_data.authoritative_input_space()
    raw = usable.loc[:, space.names].to_numpy(dtype=np.float64)
    unit = np.clip(space.normalize(raw), 0.0, 1.0)
    usable["_design_key"] = [tuple(np.round(row, 12)) for row in unit]
    usable.sort_values(["source", "case_id"], kind="stable", inplace=True)
    usable.drop_duplicates("_design_key", keep="first", inplace=True)
    usable.reset_index(drop=True, inplace=True)

    objectives = usable.loc[:, MINIMIZATION_COLUMNS].to_numpy(dtype=np.float64)
    usable["is_dominated"] = ~nondominated_mask(objectives)
    percentile_columns: list[str] = []
    for objective in MINIMIZATION_COLUMNS:
        percentile = f"{objective}_bad_percentile"
        usable[percentile] = usable[objective].rank(method="average", pct=True)
        percentile_columns.append(percentile)
    usable["badness_percentile_mean"] = usable.loc[:, percentile_columns].mean(axis=1)
    usable["archive_badness_rank"] = (
        usable["badness_percentile_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return usable


def _first_unused(rows: pd.DataFrame, used: set[int]) -> int:
    for index in rows.index:
        candidate = int(index)
        if candidate not in used:
            return candidate
    raise ValueError("could not find a unique representative for every requested role")


def _combined_distance_matrix(frame: pd.DataFrame) -> np.ndarray:
    percentile_columns = [
        f"{objective}_bad_percentile" for objective in MINIMIZATION_COLUMNS
    ]
    objective_values = frame.loc[:, percentile_columns].to_numpy(dtype=np.float64)
    space = krvea_data.authoritative_input_space()
    input_values = np.clip(
        space.normalize(frame.loc[:, space.names].to_numpy(dtype=np.float64)),
        0.0,
        1.0,
    )
    objective_difference = (
        objective_values[:, None, :] - objective_values[None, :, :]
    )
    input_difference = input_values[:, None, :] - input_values[None, :, :]
    objective_distance = np.sqrt(np.mean(objective_difference**2, axis=2))
    input_distance = np.sqrt(np.mean(input_difference**2, axis=2))
    return (
        OBJECTIVE_DISTANCE_WEIGHT * objective_distance
        + INPUT_DISTANCE_WEIGHT * input_distance
    )


def _dominator_count(objectives: np.ndarray, point: np.ndarray) -> int:
    weakly_better = np.all(objectives <= point, axis=1)
    strictly_better = np.any(objectives < point, axis=1)
    return int(np.count_nonzero(weakly_better & strictly_better))


def select_representative_controls(
    archive: pd.DataFrame,
    *,
    count: int = DEFAULT_CONTROL_COUNT,
) -> pd.DataFrame:
    """Select interpretable objective failures plus diverse bad-tail medoids."""

    if count < len(OBJECTIVE_ROLES) + 1:
        raise ValueError(f"count must be at least {len(OBJECTIVE_ROLES) + 1}")
    dominated = archive.loc[archive["is_dominated"].astype(bool)].copy()
    if len(dominated) < count:
        raise ValueError(f"only {len(dominated)} dominated completed designs are available")

    used: set[int] = set()
    selections: list[tuple[int, str, int]] = []
    percentile_columns = [
        f"{objective}_bad_percentile" for objective in MINIMIZATION_COLUMNS
    ]
    for role, objective in OBJECTIVE_ROLES:
        target = f"{objective}_bad_percentile"
        others = [column for column in percentile_columns if column != target]
        candidates = dominated.loc[
            dominated[target] >= SINGLE_OBJECTIVE_PERCENTILE
        ].copy()
        candidates["_isolation"] = (
            (candidates.loc[:, others] - 0.5) ** 2
        ).mean(axis=1)
        candidates.sort_values(
            ["_isolation", target, "badness_percentile_mean", "source", "case_id"],
            ascending=(True, False, False, True, True),
            kind="stable",
            inplace=True,
        )
        selected_index = _first_unused(candidates, used)
        used.add(selected_index)
        selections.append((selected_index, role, 1))

    overall = dominated.sort_values(
        ["badness_percentile_mean", "source", "case_id"],
        ascending=(False, True, True),
        kind="stable",
    )
    overall_index = _first_unused(overall, used)
    used.add(overall_index)
    selections.append((overall_index, "bad_overall", 1))

    remaining_count = count - len(selections)
    if remaining_count:
        threshold = float(
            dominated["badness_percentile_mean"].quantile(OBJECTIVE_TAIL_QUANTILE)
        )
        tail = dominated.loc[
            (dominated["badness_percentile_mean"] >= threshold)
            & ~dominated.index.isin(used)
        ].copy()
        if len(tail) < remaining_count:
            raise ValueError("joint bad tail is too small for the requested controls")
        distances = _combined_distance_matrix(tail)
        medoids, labels, _cost, _iterations = pam_kmedoids(
            distances,
            remaining_count,
        )
        for cluster_index, medoid_position in enumerate(medoids):
            selected_index = int(tail.index[int(medoid_position)])
            cluster_size = int(np.count_nonzero(labels == cluster_index))
            selections.append(
                (selected_index, f"bad_tail_cluster_{cluster_index + 1}", cluster_size)
            )

    objectives = archive.loc[:, MINIMIZATION_COLUMNS].to_numpy(dtype=np.float64)
    selected_rows: list[dict[str, Any]] = []
    for selection_rank, (index, role, cluster_size) in enumerate(selections, start=1):
        row = archive.loc[index].to_dict()
        point = archive.loc[index, list(MINIMIZATION_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        row.update(
            {
                "selection_rank": selection_rank,
                "selection_role": role,
                "selection_cluster_size": cluster_size,
                "dominated_by_count": _dominator_count(objectives, point),
                "return_loss_db": -20.0
                * math.log10(float(row["worst_s11_linear_amplitude"])),
            }
        )
        selected_rows.append(row)
    return pd.DataFrame.from_records(selected_rows)


def build_worklist_rows(selected: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, source in selected.sort_values("selection_rank", kind="stable").iterrows():
        sequence = int(source["selection_rank"])
        role = str(source["selection_role"])
        row: dict[str, str] = {
            "sample_id": f"bad_{sequence:02d}_{role}",
            "simulation_mode": SIMULATION_MODE,
            "selection_rank": str(sequence),
            "selection_role": role,
            "archive_badness_rank": str(int(source["archive_badness_rank"])),
            "source": str(source["source"]),
            "source_case_id": str(source["case_id"]),
            "geometry_valid": "True",
            "geometry_error": "",
            "final_conductor_components": "1",
            "dominated_by_count": str(int(source["dominated_by_count"])),
            "return_loss_db": format(float(source["return_loss_db"]), ".17g"),
            "worst_s11_linear_amplitude": format(
                float(source["worst_s11_linear_amplitude"]), ".17g"
            ),
            "mean_total_efficiency_linear": format(
                float(source["mean_total_efficiency_linear"]), ".17g"
            ),
            "normalized_substrate_area": format(
                float(source["normalized_substrate_area"]), ".17g"
            ),
            "cap_realized_gain_dbi": format(
                float(source["cap_realized_gain_dbi"]), ".17g"
            ),
            "badness_percentile_mean": format(
                float(source["badness_percentile_mean"]), ".17g"
            ),
        }
        for name in antenna_sampler.PARAMETER_REGISTRY:
            value = float(source[name])
            if not math.isfinite(value):
                raise ValueError(f"selected parameter {name} is not finite")
            row[name] = format(value, ".17g")
        antenna_sampler.parameters_from_csv_row(row)
        rows.append(row)
    return rows


def _write_csv_atomic(
    rows: Sequence[Mapping[str, Any]],
    destination: str | Path,
) -> Path:
    if not rows:
        raise ValueError("cannot write an empty propagation worklist")
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare_worklist(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    audit_path: str | Path = DEFAULT_AUDIT_PATH,
    *,
    count: int = DEFAULT_CONTROL_COUNT,
) -> tuple[pd.DataFrame, Path, Path]:
    source = Path(input_path).expanduser().resolve()
    observations = pd.read_csv(source)
    archive = prepare_archive(observations)
    selected = select_representative_controls(archive, count=count)
    worklist = _write_csv_atomic(build_worklist_rows(selected), output_path)
    audit = Path(audit_path).expanduser().resolve()
    audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit.with_name(f".{audit.name}.{os.getpid()}.tmp")
    try:
        selected.to_csv(temporary, index=False)
        os.replace(temporary, audit)
    finally:
        temporary.unlink(missing_ok=True)
    return selected, worklist, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--count", type=int, default=DEFAULT_CONTROL_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected, worklist, audit = prepare_worklist(
        args.input,
        args.output,
        args.audit,
        count=args.count,
    )
    columns = [
        "selection_rank",
        "selection_role",
        "archive_badness_rank",
        "source",
        "case_id",
        "return_loss_db",
        "mean_total_efficiency_linear",
        "normalized_substrate_area",
        "cap_realized_gain_dbi",
        "dominated_by_count",
    ]
    print(selected.loc[:, columns].to_string(index=False))
    print(f"Propagation negative-control worklist: {worklist}")
    print(f"Selection audit: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
