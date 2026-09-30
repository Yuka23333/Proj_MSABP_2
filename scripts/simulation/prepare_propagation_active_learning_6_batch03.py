"""Prepare six Phase-2 Pareto representatives for propagation validation.

The source pool is the completed 64-case Phase-2 K-RVEA campaign.  Candidates
must retain at least 7 dB worst-case in-band return loss and be non-dominated
within that campaign.  Three objective endpoints are retained first; the other
three cases are a deterministic maximin cover of normalized objective space.

This script only writes a worklist and selection audit.  It never starts CST.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import phase2_krvea_data as phase2_data  # noqa: E402
from scripts.automation import antenna_sampler  # noqa: E402
from scripts.simulation import (  # noqa: E402
    prepare_propagation_active_learning_12_batch01 as batch01,
)


PHASE2_RUN = (
    REPOSITORY_ROOT
    / "results"
    / "raw"
    / "msabp-phase2-krvea-roi-radgain-64-001"
)
OBSERVATIONS_CSV = PHASE2_RUN / "_krvea" / "observations.csv"
OUTPUT_CSV = (
    REPOSITORY_ROOT
    / "data"
    / "samples"
    / "propagation_active_learning_6_batch03.csv"
)
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "active_learning_batch03"
)
EXPECTED_PHASE2_CASES = 64
CANDIDATE_COUNT = 6
RETURN_LOSS_MIN_DB = 7.0
SIMULATION_MODE = "propagation_s21"

OBJECTIVE_COLUMNS = (
    phase2_data.WORST_S11_COLUMN,
    phase2_data.ROI_GAIN_LOSS_DBI_COLUMN,
    phase2_data.NORMALIZED_AREA_COLUMN,
)
ENDPOINTS = (
    (
        "roi_gain_endpoint",
        phase2_data.ROI_GAIN_LOSS_DBI_COLUMN,
        "maximum frozen-ROI theta radiation gain",
    ),
    (
        "s11_endpoint",
        phase2_data.WORST_S11_COLUMN,
        "minimum worst-case in-band S11 amplitude",
    ),
    (
        "area_endpoint",
        phase2_data.NORMALIZED_AREA_COLUMN,
        "minimum normalized substrate area",
    ),
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")


def _load_phase2_cases() -> pd.DataFrame:
    observations = pd.read_csv(OBSERVATIONS_CSV)
    required = {
        "source",
        "case_id",
        "case_directory",
        "status",
        "is_penalty",
        phase2_data.ROI_GAIN_LINEAR_COLUMN,
        phase2_data.ROI_GAIN_DBI_COLUMN,
        *OBJECTIVE_COLUMNS,
        *phase2_data.ACTIVE_PARAMETER_NAMES,
    }
    _require(observations, required, "Phase-2 observations")
    cases = observations.loc[observations["source"].eq(PHASE2_RUN.name)].copy()
    if len(cases) != EXPECTED_PHASE2_CASES:
        raise ValueError(
            f"expected {EXPECTED_PHASE2_CASES} Phase-2 cases, got {len(cases)}"
        )
    if cases["case_id"].duplicated().any():
        raise ValueError("Phase-2 case IDs are not unique")
    if not cases["status"].eq("completed").all() or cases["is_penalty"].any():
        raise ValueError("Phase-2 source contains incomplete or penalty observations")

    objective_values = cases.loc[:, OBJECTIVE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(objective_values).all():
        raise ValueError("Phase-2 objective table contains non-finite values")
    cases["phase2_nondominated"] = batch01._nondominated_mask(objective_values)
    cases["return_loss_db"] = -20.0 * np.log10(
        cases[phase2_data.WORST_S11_COLUMN].to_numpy(dtype=np.float64)
    )

    ffs_paths: list[str] = []
    ffs_hashes: list[str] = []
    for row in cases.itertuples(index=False):
        case_directory = Path(row.case_directory).resolve()
        expected_directory = (PHASE2_RUN / f"case_{row.case_id}").resolve()
        if case_directory != expected_directory:
            raise ValueError(f"unexpected case directory for {row.case_id}")
        manifest_path = case_directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("status") != "completed":
            raise ValueError(f"case manifest is not completed: {row.case_id}")
        artifact = manifest.get("artifacts", {}).get("farfield_source", {})
        relative_path = artifact.get("path", phase2_data.FARFIELD_FILENAME)
        ffs_path = case_directory / str(relative_path)
        if not ffs_path.is_file():
            raise FileNotFoundError(ffs_path)
        digest = _sha256(ffs_path)
        declared = artifact.get("sha256")
        if declared and str(declared).casefold() != digest.casefold():
            raise ValueError(f"FFS hash mismatch: {row.case_id}")
        ffs_paths.append(str(ffs_path))
        ffs_hashes.append(digest)
    cases["ffs_path"] = ffs_paths
    cases["ffs_sha256"] = ffs_hashes
    return cases.sort_values("case_id", kind="stable").reset_index(drop=True)


def _normalized_objectives(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, OBJECTIVE_COLUMNS].to_numpy(dtype=np.float64)
    lower = values.min(axis=0)
    span = values.max(axis=0) - lower
    span[span == 0.0] = 1.0
    return (values - lower) / span


def select_candidates(cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = cases.loc[
        cases["phase2_nondominated"]
        & (cases["return_loss_db"] >= RETURN_LOSS_MIN_DB)
    ].copy()
    eligible.sort_values("case_id", kind="stable", inplace=True)
    eligible.reset_index(drop=True, inplace=True)
    if len(eligible) < CANDIDATE_COUNT:
        raise ValueError(
            f"only {len(eligible)} eligible Phase-2 Pareto cases; "
            f"cannot select {CANDIDATE_COUNT}"
        )

    normalized = _normalized_objectives(eligible)
    chosen: list[int] = []
    roles: dict[int, tuple[str, str]] = {}
    min_distance_at_selection: dict[int, float] = {}
    for role, column, reason in ENDPOINTS:
        minimum = float(eligible[column].min())
        candidates = eligible.index[
            np.isclose(
                eligible[column].to_numpy(dtype=np.float64),
                minimum,
                atol=1.0e-14,
                rtol=0.0,
            )
        ].tolist()
        index = min(candidates, key=lambda item: str(eligible.loc[item, "case_id"]))
        if index not in chosen:
            chosen.append(index)
            roles[index] = (role, reason)
            min_distance_at_selection[index] = math.nan

    while len(chosen) < CANDIDATE_COUNT:
        available = [index for index in eligible.index if index not in chosen]
        distances = {
            index: min(
                float(np.linalg.norm(normalized[index] - normalized[anchor]))
                for anchor in chosen
            )
            for index in available
        }
        best_distance = max(distances.values())
        tied = [
            index
            for index, distance in distances.items()
            if math.isclose(distance, best_distance, abs_tol=1.0e-14, rel_tol=0.0)
        ]
        index = min(tied, key=lambda item: str(eligible.loc[item, "case_id"]))
        chosen.append(index)
        roles[index] = (
            "pareto_maximin_cover",
            "maximum distance from selected cases in normalized Phase-2 objective space",
        )
        min_distance_at_selection[index] = distances[index]

    selected = eligible.loc[chosen].copy().reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    selected["acquisition_type"] = [roles[index][0] for index in chosen]
    selected["selection_reason"] = [roles[index][1] for index in chosen]
    selected["objective_distance_at_selection"] = [
        min_distance_at_selection[index] for index in chosen
    ]
    diagnostics = {
        "phase2_case_count": int(len(cases)),
        "phase2_nondominated_count": int(cases["phase2_nondominated"].sum()),
        "return_loss_eligible_count": int(
            (cases["return_loss_db"] >= RETURN_LOSS_MIN_DB).sum()
        ),
        "eligible_nondominated_count": int(len(eligible)),
        "selected_count": int(len(selected)),
    }
    return selected, diagnostics


def _parameter_value(row: Any, name: str) -> float:
    if hasattr(row, name):
        return float(getattr(row, name))
    if name in phase2_data.FIXED_PARAMETER_VALUES:
        return float(phase2_data.FIXED_PARAMETER_VALUES[name])
    raise KeyError(f"Phase-2 observation does not define parameter {name}")


def _worklist_rows(selected: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in selected.itertuples(index=False):
        row = {
            "sample_id": f"al03_phase2_{int(source.selection_rank):02d}",
            "simulation_mode": SIMULATION_MODE,
            "selection_rank": str(int(source.selection_rank)),
            "acquisition_type": str(source.acquisition_type),
            "selection_reason": str(source.selection_reason),
            "source": PHASE2_RUN.name,
            "source_case_id": str(source.case_id),
            "source_ffs_sha256": str(source.ffs_sha256),
            "return_loss_db": format(float(source.return_loss_db), ".17g"),
            phase2_data.WORST_S11_COLUMN: format(
                float(getattr(source, phase2_data.WORST_S11_COLUMN)), ".17g"
            ),
            phase2_data.ROI_GAIN_LINEAR_COLUMN: format(
                float(getattr(source, phase2_data.ROI_GAIN_LINEAR_COLUMN)), ".17g"
            ),
            phase2_data.ROI_GAIN_DBI_COLUMN: format(
                float(getattr(source, phase2_data.ROI_GAIN_DBI_COLUMN)), ".17g"
            ),
            phase2_data.NORMALIZED_AREA_COLUMN: format(
                float(getattr(source, phase2_data.NORMALIZED_AREA_COLUMN)), ".17g"
            ),
            "phase2_nondominated": "True",
        }
        for name in antenna_sampler.PARAMETER_REGISTRY:
            value = _parameter_value(source, name)
            if not math.isfinite(value):
                raise ValueError(f"selected parameter {name} is not finite")
            row[name] = format(value, ".17g")
        antenna_sampler.parameters_from_csv_row(row)
        rows.append(row)
    return rows


def _write_worklist(rows: Sequence[Mapping[str, str]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_CSV.with_name(f".{OUTPUT_CSV.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, OUTPUT_CSV)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_batch(*, overwrite: bool = False) -> dict[str, Any]:
    if not OBSERVATIONS_CSV.is_file():
        raise FileNotFoundError(OBSERVATIONS_CSV)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT_DIRECTORY / "selection_audit.csv"
    manifest_path = OUTPUT_DIRECTORY / "selection_manifest.json"
    existing = [path for path in (OUTPUT_CSV, audit_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"batch-03 output exists: {existing[0]}")

    cases = _load_phase2_cases()
    selected, diagnostics = select_candidates(cases)
    rows = _worklist_rows(selected)
    _write_worklist(rows)
    _write_csv_atomic(selected, audit_path)

    payload: dict[str, Any] = {
        "schema": "msabp.propagation_active_learning_selection.v3",
        "batch": 3,
        "simulation_mode": SIMULATION_MODE,
        "source_campaign": PHASE2_RUN.name,
        "selection_policy": {
            "candidate_count": CANDIDATE_COUNT,
            "return_loss_min_db": RETURN_LOSS_MIN_DB,
            "require_phase2_nondominated": True,
            "endpoint_order": [role for role, _, _ in ENDPOINTS],
            "fill": "greedy maximin Euclidean distance in min-max normalized three-objective space",
            "objectives_all_minimize": list(OBJECTIVE_COLUMNS),
            "propagation_labels_used_for_selection": False,
            "purpose": "independent validation of the frozen Phase-2 propagation proxy",
        },
        "population": diagnostics,
        "selected": [
            {
                "sample_id": row["sample_id"],
                "source_case_id": row["source_case_id"],
                "acquisition_type": row["acquisition_type"],
                "return_loss_db": float(row["return_loss_db"]),
                phase2_data.ROI_GAIN_DBI_COLUMN: float(
                    row[phase2_data.ROI_GAIN_DBI_COLUMN]
                ),
                phase2_data.NORMALIZED_AREA_COLUMN: float(
                    row[phase2_data.NORMALIZED_AREA_COLUMN]
                ),
            }
            for row in rows
        ],
        "inputs": {
            "observations": {
                "path": str(OBSERVATIONS_CSV),
                "sha256": _sha256(OBSERVATIONS_CSV),
            }
        },
        "outputs": {
            "worklist": str(OUTPUT_CSV),
            "worklist_sha256": _sha256(OUTPUT_CSV),
            "selection_audit": str(audit_path),
            "selection_audit_sha256": _sha256(audit_path),
        },
    }
    _write_json_atomic(payload, manifest_path)

    display = [
        "selection_rank",
        "acquisition_type",
        "case_id",
        "return_loss_db",
        phase2_data.ROI_GAIN_DBI_COLUMN,
        phase2_data.NORMALIZED_AREA_COLUMN,
        "objective_distance_at_selection",
    ]
    print("[ActiveLearning-03] selected 3 endpoints + 3 Pareto maximin covers")
    print(selected[display].to_string(index=False))
    print(f"[ActiveLearning-03] worklist -> {OUTPUT_CSV}")
    print(f"[ActiveLearning-03] audit -> {audit_path}")
    print(f"[ActiveLearning-03] manifest -> {manifest_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_batch(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
