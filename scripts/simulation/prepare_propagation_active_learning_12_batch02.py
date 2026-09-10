"""Prepare the 12-case propagation batch used to confirm the refitted ROI.

Allocation: 4 old/new ROI disagreements, 4 predicted S21/BER disagreements,
2 uncertain diverse cases, and 2 exploitation cases. No CST solve is started.
"""

from __future__ import annotations

import argparse
import csv
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

from msabp_opt.optimization import krvea_data  # noqa: E402
from scripts.automation import antenna_sampler  # noqa: E402
from scripts.simulation import (  # noqa: E402
    prepare_propagation_active_learning_12_batch01 as batch01,
)


PROXY_ROOT = REPOSITORY_ROOT / "results" / "processed" / "onbody_proxies"
BATCH01_ROOT = PROXY_ROOT / "active_learning_batch01"
MUSHROOM_ROOT = BATCH01_ROOT / "mushroom_analysis_batch01"
FREQUENCY_TABLE = batch01.DEFAULT_FREQUENCY_TABLE
NEW_ROI_RESULT = MUSHROOM_ROOT / "spherical_roi_25_batch01" / "result.json"
NEW_ROI_CASES = NEW_ROI_RESULT.parent / "optimal_roi_case_metrics.csv"
OLD_ROI_CACHE = BATCH01_ROOT / "eligible_roi_features.csv"
OLD_ROI_CASES = PROXY_ROOT / "spherical_roi_13" / "optimal_roi_case_metrics.csv"
OLD_LABELS = (
    PROXY_ROOT
    / "link_correlation_13"
    / "selected_13_proxy_frequency_table.csv"
)
NEW_LABELS = MUSHROOM_ROOT / "frozen_validation_12.csv"
BATCH01_AUDIT = BATCH01_ROOT / "selection_audit.csv"
PREVIOUS_WORKLISTS = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_selected_13.csv",
    REPOSITORY_ROOT
    / "data"
    / "samples"
    / "propagation_active_learning_12_batch01.csv",
)
OUTPUT_CSV = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_active_learning_12_batch02.csv"
)
OUTPUT_DIRECTORY = PROXY_ROOT / "active_learning_batch02"
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
DEFAULT_BOOTSTRAPS = 256
DEFAULT_RANDOM_SEED = 20260909
SIMULATION_MODE = "propagation_s21"
OPTIMISM_BETA = 0.25
COUNTS = {
    "roi_disagreement": 4,
    "link_disagreement": 4,
    "uncertainty": 2,
    "exploitation": 2,
}
ABBREVIATIONS = {
    "roi_disagreement": "roi",
    "link_disagreement": "link",
    "uncertainty": "unc",
    "exploitation": "exp",
}


def _sha256(path: str | Path) -> str:
    return batch01._sha256(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
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


def _load_labels() -> pd.DataFrame:
    old = pd.read_csv(OLD_LABELS).drop_duplicates("link_case_name")
    new = pd.read_csv(NEW_LABELS)
    if "sample_index" not in new.columns:
        audit = pd.read_csv(BATCH01_AUDIT)
        _require(
            audit,
            {"selection_rank", "sample_index"},
            "batch-01 selection audit",
        )
        new = new.merge(
            audit[["selection_rank", "sample_index"]],
            on="selection_rank",
            how="left",
            validate="one_to_one",
        )
    required = {
        "sample_index",
        "s21_power_september",
        "ebn0_ber_v2_at_target_db",
    }
    _require(old, required | {"link_case_name"}, "old labels")
    _require(new, required | {"case_name"}, "new labels")
    old = old[
        [
            "sample_index",
            "link_case_name",
            "s21_power_september",
            "ebn0_ber_v2_at_target_db",
        ]
    ]
    new = new[
        [
            "sample_index",
            "case_name",
            "s21_power_september",
            "ebn0_ber_v2_at_target_db",
        ]
    ].rename(columns={"case_name": "link_case_name"})
    labels = pd.concat([old, new], ignore_index=True)
    if len(labels) != 25 or labels["sample_index"].duplicated().any():
        raise ValueError("expected 25 unique propagation labels")
    return labels


def _old_roi_values(population: pd.DataFrame) -> np.ndarray:
    cached = pd.read_csv(OLD_ROI_CACHE)
    _require(
        cached,
        {"sample_index", "identity", "roi_metric_linear"},
        "old ROI cache",
    )
    values = population[["sample_index", "identity"]].merge(
        cached[["sample_index", "identity", "roi_metric_linear"]],
        on=["sample_index", "identity"],
        how="left",
        validate="one_to_one",
    )
    reference = pd.read_csv(OLD_ROI_CASES)
    reference = reference.loc[
        reference["case_name"].eq(batch01.REFERENCE_CASE_NAME)
    ]
    if len(reference) != 1:
        raise ValueError("old ROI table must contain one reference")
    mask = population["source"].eq(batch01.REFERENCE_CASE_NAME)
    values.loc[mask, "roi_metric_linear"] = float(
        reference.iloc[0]["roi_metric_linear"]
    )
    if values["roi_metric_linear"].isna().any():
        raise ValueError("old ROI cache does not cover the population")
    return values["roi_metric_linear"].to_numpy(dtype=np.float64)


def _attach_predictions(
    population: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    bootstraps: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_names = [
        "new_roi_metric_linear",
        "old_roi_metric_linear",
        *(name for name, _, _ in batch01.PROXY_FEATURES),
    ]
    model = population.copy()
    rank_names = []
    for name in feature_names:
        rank_name = f"{name}_rank"
        model[rank_name] = model[name].rank(method="average", pct=True)
        rank_names.append(rank_name)
    model = model.merge(
        labels[
            [
                "sample_index",
                "s21_power_september",
                "ebn0_ber_v2_at_target_db",
            ]
        ],
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    labeled = model["s21_power_september"].notna()
    if int(labeled.sum()) != 25:
        raise ValueError("not every propagation label maps to the population")
    x_all = model[rank_names].to_numpy(dtype=np.float64)
    x_train_raw = x_all[labeled]
    x_mean = x_train_raw.mean(axis=0)
    x_scale = x_train_raw.std(axis=0, ddof=1)
    x_scale[x_scale == 0.0] = 1.0
    x_all = (x_all - x_mean) / x_scale
    x_train = x_all[labeled]
    targets = {
        "log10_s21_power": np.log10(
            model.loc[labeled, "s21_power_september"].to_numpy(dtype=np.float64)
        ),
        "negative_ebn0_db": -model.loc[
            labeled, "ebn0_ber_v2_at_target_db"
        ].to_numpy(dtype=np.float64),
    }
    diagnostics: dict[str, Any] = {
        "training_case_count": 25,
        "feature_names": feature_names,
        "bootstraps": bootstraps,
        "random_seed": random_seed,
        "targets": {},
    }
    for offset, (target_name, y_train) in enumerate(targets.items()):
        alpha, loo = batch01._select_ridge_alpha(x_train, y_train)
        mean, std = batch01._bootstrap_ridge_predictions(
            x_train,
            y_train,
            x_all,
            alpha=alpha,
            bootstraps=bootstraps,
            random_seed=random_seed + offset,
        )
        model[f"predicted_{target_name}"] = mean
        model[f"uncertainty_{target_name}"] = std
        diagnostics["targets"][target_name] = {
            "selected_alpha": alpha,
            "loo_mse_by_alpha": loo,
            "training_std": float(y_train.std(ddof=1)),
        }
    model["predicted_s21_power"] = 10.0 ** model["predicted_log10_s21_power"]
    model["predicted_ebn0_db"] = -model["predicted_negative_ebn0_db"]
    s21_scale = diagnostics["targets"]["log10_s21_power"]["training_std"]
    ebn0_scale = diagnostics["targets"]["negative_ebn0_db"]["training_std"]
    model["normalized_uncertainty"] = 0.5 * (
        model["uncertainty_log10_s21_power"] / s21_scale
        + model["uncertainty_negative_ebn0_db"] / ebn0_scale
    )
    model["optimistic_s21"] = (
        model["predicted_log10_s21_power"]
        + OPTIMISM_BETA * model["uncertainty_log10_s21_power"]
    )
    model["optimistic_negative_ebn0"] = (
        model["predicted_negative_ebn0_db"]
        + OPTIMISM_BETA * model["uncertainty_negative_ebn0_db"]
    )
    model["predicted_s21_rank"] = model["predicted_log10_s21_power"].rank(
        method="average", pct=True
    )
    model["predicted_negative_ebn0_rank"] = model[
        "predicted_negative_ebn0_db"
    ].rank(method="average", pct=True)
    model["joint_optimistic_score"] = np.minimum(
        model["optimistic_s21"].rank(method="average", pct=True),
        model["optimistic_negative_ebn0"].rank(method="average", pct=True),
    )
    return model, diagnostics


def _known_identities() -> set[str]:
    identities: set[str] = set()
    for path in PREVIOUS_WORKLISTS:
        frame = pd.read_csv(path)
        _require(frame, {"source", "source_case_id"}, path.name)
        identities.update(
            batch01._identity(source, case_id)
            for source, case_id in zip(
                frame["source"], frame["source_case_id"], strict=True
            )
        )
    if len(identities) != 25:
        raise ValueError(f"expected 25 propagated geometries, got {len(identities)}")
    return identities


def _selection_space(frame: pd.DataFrame) -> np.ndarray:
    proxy_names = [
        "new_roi_metric_linear_rank",
        "old_roi_metric_linear_rank",
        *(f"{name}_rank" for name, _, _ in batch01.PROXY_FEATURES),
    ]
    space = krvea_data.authoritative_input_space()
    geometry = space.normalize(frame.loc[:, space.names].to_numpy(dtype=np.float64))
    objectives = np.column_stack(
        [
            frame["worst_s11_linear_amplitude"].rank(pct=True),
            (1.0 - frame["rad_eff_september_linear"]).rank(pct=True),
            frame["normalized_substrate_area"].rank(pct=True),
        ]
    )
    proxy = frame[proxy_names].to_numpy(dtype=np.float64)
    return np.concatenate(
        (
            0.60 * proxy / math.sqrt(proxy.shape[1]),
            0.25 * geometry / math.sqrt(geometry.shape[1]),
            0.15 * objectives / math.sqrt(objectives.shape[1]),
        ),
        axis=1,
    )


def _select(
    population: pd.DataFrame,
    eligible: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    archive = eligible.merge(
        population.drop(
            columns=[
                "source",
                "case_id",
                "case_directory",
                "worst_s11_linear_amplitude",
                "mean_total_efficiency_linear",
                "rad_eff_september_linear",
                "normalized_substrate_area",
                "return_loss_db",
            ],
            errors="ignore",
        ),
        on=["sample_index", "identity"],
        how="left",
        validate="one_to_one",
    ).merge(parameters, on="sample_index", how="left", validate="one_to_one")
    archive.reset_index(drop=True, inplace=True)
    archive["previously_propagated"] = archive["identity"].isin(
        _known_identities()
    )
    archive["roi_rank_disagreement"] = np.abs(
        archive["new_roi_metric_linear_rank"]
        - archive["old_roi_metric_linear_rank"]
    )
    archive["link_rank_disagreement"] = np.abs(
        archive["predicted_s21_rank"]
        - archive["predicted_negative_ebn0_rank"]
    )
    objective_values = np.column_stack(
        [
            archive["worst_s11_linear_amplitude"],
            1.0 - archive["rad_eff_september_linear"],
            archive["normalized_substrate_area"],
        ]
    )
    archive["three_objective_nondominated"] = batch01._nondominated_mask(
        objective_values
    )
    points = _selection_space(archive)
    known = archive.index[archive["previously_propagated"]].tolist()
    available = set(archive.index[~archive["previously_propagated"]].tolist())
    pareto = set(
        archive.index[
            ~archive["previously_propagated"]
            & archive["three_objective_nondominated"]
        ].tolist()
    )
    if len(available) < 12 or len(pareto) < COUNTS["exploitation"]:
        raise ValueError("candidate pool cannot fill batch 02")
    chosen: dict[str, list[int]] = {}
    selected_so_far: list[int] = []

    def choose(role: str, pool: set[int], base: str, diversity: float) -> None:
        indices = batch01._greedy_select(
            sorted(pool),
            COUNTS[role],
            base_values=archive[base].to_numpy(dtype=np.float64),
            selection_space=points,
            anchor_indices=[*known, *selected_so_far],
            diversity_weight=diversity,
        )
        chosen[role] = indices
        selected_so_far.extend(indices)
        available.difference_update(indices)
        pareto.difference_update(indices)

    choose("exploitation", pareto, "joint_optimistic_score", 0.15)
    choose("roi_disagreement", available, "roi_rank_disagreement", 0.25)
    choose("link_disagreement", available, "link_rank_disagreement", 0.25)
    choose("uncertainty", available, "normalized_uncertainty", 0.45)

    archive["distance_to_previous_labels"] = batch01._minimum_distances(
        points,
        points[np.asarray(known, dtype=np.int64)],
    )
    reasons = {
        "roi_disagreement": "maximum frozen-old versus refitted-new ROI rank disagreement",
        "link_disagreement": "maximum predicted September-S21 versus BER-v2 rank disagreement",
        "uncertainty": "bootstrap uncertainty plus distance from prior propagation labels",
        "exploitation": "joint optimistic link prediction on retained S11/RadEff/area Pareto set",
    }
    rows = []
    for role in (
        "roi_disagreement",
        "link_disagreement",
        "uncertainty",
        "exploitation",
    ):
        for role_rank, index in enumerate(chosen[role], start=1):
            row = archive.loc[index].copy()
            row["acquisition_type"] = role
            row["acquisition_rank_within_type"] = role_rank
            row["selection_reason"] = reasons[role]
            rows.append(row)
    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, 13))
    diagnostics = {
        "eligible_return_loss_ge_7db": len(archive),
        "previously_propagated_excluded": int(
            archive["previously_propagated"].sum()
        ),
        "untested_candidate_count": int((~archive["previously_propagated"]).sum()),
        "untested_retained_pareto_count": int(
            (
                ~archive["previously_propagated"]
                & archive["three_objective_nondominated"]
            ).sum()
        ),
        "selection_counts": {
            role: len(indices) for role, indices in chosen.items()
        },
    }
    return selected, diagnostics


def _worklist_rows(selected: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counters = {name: 0 for name in COUNTS}
    for source in selected.itertuples(index=False):
        role = str(source.acquisition_type)
        counters[role] += 1
        row = {
            "sample_id": f"al02_{ABBREVIATIONS[role]}_{counters[role]:02d}",
            "simulation_mode": SIMULATION_MODE,
            "selection_rank": str(int(source.selection_rank)),
            "acquisition_type": role,
            "acquisition_rank_within_type": str(counters[role]),
            "selection_reason": str(source.selection_reason),
            "source": str(source.source),
            "source_case_id": str(source.case_id),
            "source_ffs_sha256": str(source.ffs_sha256),
            "geometry_valid": "True",
            "geometry_error": "",
            "final_conductor_components": "1",
            "return_loss_db": format(float(source.return_loss_db), ".17g"),
            "predicted_s21_power": format(
                float(source.predicted_s21_power), ".17g"
            ),
            "predicted_ebn0_db": format(float(source.predicted_ebn0_db), ".17g"),
            "uncertainty_log10_s21_power": format(
                float(source.uncertainty_log10_s21_power), ".17g"
            ),
            "uncertainty_negative_ebn0_db": format(
                float(source.uncertainty_negative_ebn0_db), ".17g"
            ),
            "old_roi_metric_linear": format(
                float(source.old_roi_metric_linear), ".17g"
            ),
            "new_roi_metric_linear": format(
                float(source.new_roi_metric_linear), ".17g"
            ),
            "roi_rank_disagreement": format(
                float(source.roi_rank_disagreement), ".17g"
            ),
            "link_rank_disagreement": format(
                float(source.link_rank_disagreement), ".17g"
            ),
            "three_objective_nondominated": str(
                bool(source.three_objective_nondominated)
            ),
        }
        for name in antenna_sampler.PARAMETER_REGISTRY:
            value = float(getattr(source, name))
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


def prepare_batch(
    *,
    workers: int = DEFAULT_WORKERS,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    inputs = [
        FREQUENCY_TABLE,
        NEW_ROI_RESULT,
        NEW_ROI_CASES,
        OLD_ROI_CACHE,
        OLD_ROI_CASES,
        OLD_LABELS,
        NEW_LABELS,
        BATCH01_AUDIT,
        *PREVIOUS_WORKLISTS,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    feature_cache = OUTPUT_DIRECTORY / "eligible_new_roi_features.csv"
    predictions_path = OUTPUT_DIRECTORY / "candidate_predictions_frozen.csv"
    audit_path = OUTPUT_DIRECTORY / "selection_audit.csv"
    manifest_path = OUTPUT_DIRECTORY / "selection_manifest.json"
    existing = [
        path
        for path in (OUTPUT_CSV, predictions_path, audit_path, manifest_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"batch-02 output exists: {existing[0]}")

    contract = batch01._load_roi_contract(NEW_ROI_RESULT)
    per_frequency, eligible = batch01._load_candidate_metadata(FREQUENCY_TABLE)
    new_roi = batch01._load_or_create_roi_cache(
        eligible,
        contract,
        feature_cache,
        workers=workers,
        overwrite=overwrite,
    )
    population = batch01._build_model_feature_table(
        per_frequency,
        eligible,
        new_roi,
        NEW_ROI_CASES,
    )
    population.rename(
        columns={
            "roi_metric_linear": "new_roi_metric_linear",
            "roi_metric_linear_rank": "new_roi_metric_linear_rank",
        },
        inplace=True,
    )
    population["old_roi_metric_linear"] = _old_roi_values(population)
    population, model_diagnostics = _attach_predictions(
        population,
        _load_labels(),
        bootstraps=bootstraps,
        random_seed=random_seed,
    )
    selected, selection_diagnostics = _select(
        population,
        eligible,
        batch01._load_parameters(eligible),
    )
    for source in selected.itertuples(index=False):
        if _sha256(source.ffs_path).casefold() != str(source.ffs_sha256).casefold():
            raise ValueError(f"selected FFS hash mismatch: {source.ffs_path}")

    rows = _worklist_rows(selected)
    _write_worklist(rows)
    _write_csv(population, predictions_path)
    _write_csv(selected, audit_path)
    payload: dict[str, Any] = {
        "schema": "msabp.propagation_active_learning_selection.v2",
        "batch": 2,
        "simulation_mode": SIMULATION_MODE,
        "phantom_scope": (
            "open-boundary infinite planar Muscle phantom; not validated for a "
            "finite cylindrical phantom"
        ),
        "selection_policy": {
            "allocation": COUNTS,
            "internal_order": [
                "exploitation",
                "roi_disagreement",
                "link_disagreement",
                "uncertainty",
            ],
            "return_loss_min_db": batch01.RETURN_LOSS_MIN_DB,
            "optimism_beta": OPTIMISM_BETA,
            "cap_gain_used": False,
            "retained_pareto_objectives": [
                "min worst S11 amplitude",
                "max September Rad_Eff",
                "min normalized substrate area",
            ],
        },
        "new_roi_contract": contract.__dict__,
        "model": model_diagnostics,
        "population": selection_diagnostics,
        "selected": [
            {
                "sample_id": row["sample_id"],
                "acquisition_type": row["acquisition_type"],
                "source": row["source"],
                "source_case_id": row["source_case_id"],
                "predicted_s21_power": float(row["predicted_s21_power"]),
                "predicted_ebn0_db": float(row["predicted_ebn0_db"]),
                "roi_rank_disagreement": float(row["roi_rank_disagreement"]),
                "link_rank_disagreement": float(row["link_rank_disagreement"]),
            }
            for row in rows
        ],
        "inputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in inputs
        },
        "outputs": {
            "worklist": str(OUTPUT_CSV),
            "worklist_sha256": _sha256(OUTPUT_CSV),
            "new_roi_feature_cache": str(feature_cache),
            "new_roi_feature_cache_sha256": _sha256(feature_cache),
            "candidate_predictions_frozen": str(predictions_path),
            "candidate_predictions_frozen_sha256": _sha256(predictions_path),
            "selection_audit": str(audit_path),
            "selection_audit_sha256": _sha256(audit_path),
        },
    }
    _write_json(payload, manifest_path)
    display = [
        "selection_rank",
        "acquisition_type",
        "source",
        "case_id",
        "return_loss_db",
        "roi_rank_disagreement",
        "link_rank_disagreement",
        "predicted_ebn0_db",
        "normalized_uncertainty",
        "three_objective_nondominated",
    ]
    print("[ActiveLearning-02] selected 4 ROI + 4 link + 2 uncertainty + 2 exploit")
    print(selected[display].to_string(index=False))
    print(f"[ActiveLearning-02] worklist -> {OUTPUT_CSV}")
    print(f"[ActiveLearning-02] audit -> {audit_path}")
    print(f"[ActiveLearning-02] manifest -> {manifest_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_batch(
        workers=args.workers,
        bootstraps=args.bootstraps,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
