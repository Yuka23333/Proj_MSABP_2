"""Fit a legacy-metric ROI on 37 cases and validate it on six locked cases.

Training targets are the historical uniform-band mean ``|S21|`` and the
single-wideband-pulse BER=1e-4 Eb/N0 threshold.  The existing 37-case FFS tensor
is used for ROI search.  Active-learning batch 03 is held out completely and
is evaluated only after the ROI is frozen.
"""

from __future__ import annotations

import argparse
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
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing import ber_03_average_s21 as uniform_s21  # noqa: E402
from scripts.postprocessing import ber_04_run_experiment as legacy_ber  # noqa: E402
from scripts.postprocessing.prepare_link_ffs_tensor import prepare_tensor  # noqa: E402
from scripts.postprocessing.search_link_spherical_roi import (  # noqa: E402
    _interval_indices,
    _load_tensor,
    make_field_maps,
    region_feature,
    run_search,
    spherical_theta_cell_weights,
    symmetric_phi_mask,
)


PROXY_ROOT = REPOSITORY_ROOT / "results" / "processed" / "onbody_proxies"
BATCH02_ANALYSIS = (
    PROXY_ROOT / "active_learning_batch02" / "mushroom_analysis_batch02"
)
BATCH03_ROOT = PROXY_ROOT / "active_learning_batch03"

TRAINING_LINK = BATCH02_ANALYSIS / "combined_link_metrics_37.csv"
TRAINING_TENSOR = BATCH02_ANALYSIS / "link_ffs_tensor_37_3p1-4p8GHz.npz"
HOLDOUT_WORKLIST = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_active_learning_6_batch03.csv"
)
HOLDOUT_AUDIT = BATCH03_ROOT / "selection_audit.csv"
HOLDOUT_UNIFORM_S21 = (
    BATCH03_ROOT / "legacy_uniform_metrics" / "average_s21_3p1-4p8GHz.csv"
)
HOLDOUT_BER = (
    BATCH03_ROOT
    / "legacy_uniform_metrics"
    / "ber_v1_single_uwb_5Mbps"
    / "ber_results.csv"
)
TRAINING_BER_FILES = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "propagation_s21_14"
    / "ber_results"
    / "matched_filter_5Mbps"
    / "ber_results.csv",
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "propagation_s21_14"
    / "roblin_wei_2012_reference"
    / "ber_results"
    / "matched_filter_5Mbps"
    / "ber_results.csv",
    PROXY_ROOT
    / "active_learning_batch01"
    / "legacy_ber_matched_filter_5Mbps"
    / "ber_results.csv",
    PROXY_ROOT
    / "active_learning_batch02"
    / "legacy_ber_matched_filter_5Mbps"
    / "ber_results.csv",
)
DEFAULT_OUTPUT_DIRECTORY = BATCH03_ROOT / "legacy_roi_fit37_holdout6"
EXPECTED_TRAINING_CASES = 37
EXPECTED_HOLDOUT_CASES = 6
TARGET_BER = 1.0e-4
FIELD_FAMILY = "radiation_gain"


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


def _ber_thresholds(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    records = [legacy_ber.BerAggregate(**row) for row in frame.to_dict("records")]
    thresholds: dict[str, float] = {}
    for case_name in sorted({record.case_name for record in records}):
        selected = [record for record in records if record.case_name == case_name]
        threshold = legacy_ber._interpolated_threshold_ebn0(selected, TARGET_BER)
        if threshold is not None:
            thresholds[case_name] = float(threshold)
    return thresholds


def _combined_ber_thresholds(paths: Sequence[Path]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for path in paths:
        for case_name, threshold in _ber_thresholds(path).items():
            if case_name in combined:
                raise ValueError(f"duplicate legacy BER case: {case_name}")
            combined[case_name] = threshold
    return combined


def _build_training_labels() -> pd.DataFrame:
    source = pd.read_csv(TRAINING_LINK)
    _require(source, {"case_name", "source_s21_path", "source_s21_sha256"}, "training link")
    if len(source) != EXPECTED_TRAINING_CASES or source["case_name"].duplicated().any():
        raise ValueError("training link table must contain 37 unique cases")

    thresholds = _combined_ber_thresholds(TRAINING_BER_FILES)
    rows: list[dict[str, Any]] = []
    for record in source.itertuples(index=False):
        s21_path = Path(record.source_s21_path).resolve()
        if _sha256(s21_path).casefold() != str(record.source_s21_sha256).casefold():
            raise ValueError(f"training S21 hash mismatch: {record.case_name}")
        average = uniform_s21.summarize_s21_directory(s21_path.parent)
        if record.case_name not in thresholds:
            raise ValueError(f"missing legacy BER threshold: {record.case_name}")
        rows.append(
            {
                "case_name": str(record.case_name),
                "cohort": str(record.cohort),
                "legacy_mean_s21_linear": average.mean_s21_linear,
                "legacy_mean_s21_db": average.mean_s21_db,
                "legacy_ber_ebn0_at_target_db": thresholds[record.case_name],
                # Compatibility aliases consumed by the generic ROI search.
                "s21_power_september": average.mean_s21_linear,
                "ebn0_ber_v2_at_target_db": thresholds[record.case_name],
                "source_s21_path": str(s21_path),
                "source_s21_sha256": str(record.source_s21_sha256),
            }
        )
    return pd.DataFrame(rows)


def _build_holdout_ffs_selection() -> pd.DataFrame:
    worklist = pd.read_csv(HOLDOUT_WORKLIST)
    audit = pd.read_csv(HOLDOUT_AUDIT)
    _require(worklist, {"sample_id", "selection_rank", "source_case_id"}, "holdout worklist")
    _require(audit, {"selection_rank", "case_id", "ffs_path", "ffs_sha256"}, "holdout audit")
    joined = worklist.merge(
        audit[["selection_rank", "case_id", "ffs_path", "ffs_sha256"]],
        on="selection_rank",
        validate="one_to_one",
    )
    if len(joined) != EXPECTED_HOLDOUT_CASES:
        raise ValueError("holdout selection must contain six cases")
    if not joined["source_case_id"].astype(str).eq(joined["case_id"].astype(str)).all():
        raise ValueError("holdout worklist and FFS audit disagree")
    return pd.DataFrame(
        {
            "link_case_name": "case_" + joined["sample_id"].astype(str),
            "source_case_id": joined["source_case_id"].astype(str),
            "ffs_path": joined["ffs_path"].astype(str),
            "ffs_sha256": joined["ffs_sha256"].astype(str),
        }
    )


def _holdout_targets() -> pd.DataFrame:
    s21 = pd.read_csv(HOLDOUT_UNIFORM_S21)
    _require(s21, {"case_name", "mean_s21_linear", "mean_s21_db"}, "holdout S21")
    thresholds = _ber_thresholds(HOLDOUT_BER)
    s21 = s21.copy()
    s21["legacy_ber_ebn0_at_target_db"] = s21["case_name"].map(thresholds)
    if len(s21) != EXPECTED_HOLDOUT_CASES or s21["legacy_ber_ebn0_at_target_db"].isna().any():
        raise ValueError("holdout legacy metrics must cover six cases")
    return s21


def _evaluate_frozen_roi(
    tensor_path: Path,
    optimum: Mapping[str, Any],
) -> pd.DataFrame:
    arrays = _load_tensor(tensor_path)
    maps, _ = make_field_maps(arrays, FIELD_FAMILY)
    frequency = np.asarray(arrays["frequency_ghz"], dtype=np.float64)
    theta = np.asarray(arrays["theta_deg"], dtype=np.float64)
    phi = np.asarray(arrays["phi_deg"], dtype=np.float64)
    theta_indices = _interval_indices(
        theta,
        (float(optimum["theta_min_deg"]), float(optimum["theta_max_deg"])),
        "theta",
    )
    frequency_indices = _interval_indices(
        frequency,
        (
            float(optimum["frequency_min_ghz"]),
            float(optimum["frequency_max_ghz"]),
        ),
        "frequency",
    )
    phi_mask = symmetric_phi_mask(
        phi,
        90.0,
        float(optimum["phi_half_width_deg"]),
    )
    feature = region_feature(
        maps[str(optimum["component"])],
        theta_low=int(theta_indices[0]),
        theta_high=int(theta_indices[-1]),
        phi_mask=phi_mask,
        frequency_interval=(int(frequency_indices[0]), int(frequency_indices[-1])),
        theta_weights=spherical_theta_cell_weights(theta),
    )
    return pd.DataFrame(
        {
            "case_name": np.asarray(arrays["case_name"]).astype(str),
            "legacy_roi_metric_linear": feature,
        }
    )


def _spearman(left: pd.Series, right: pd.Series) -> float:
    value = float(stats.spearmanr(left, right).statistic)
    if not math.isfinite(value):
        raise ValueError("holdout Spearman correlation is not finite")
    return value


def run_analysis(output_directory: Path, *, overwrite: bool = False) -> dict[str, Any]:
    named_inputs = {
        "training_link_37": TRAINING_LINK,
        "training_ffs_tensor_37": TRAINING_TENSOR,
        "holdout_worklist_6": HOLDOUT_WORKLIST,
        "holdout_selection_audit_6": HOLDOUT_AUDIT,
        "holdout_uniform_s21_6": HOLDOUT_UNIFORM_S21,
        "holdout_legacy_ber_6": HOLDOUT_BER,
        "training_ber_original_archive": TRAINING_BER_FILES[0],
        "training_ber_roblin_wei_reference": TRAINING_BER_FILES[1],
        "training_ber_active_learning_batch01": TRAINING_BER_FILES[2],
        "training_ber_active_learning_batch02": TRAINING_BER_FILES[3],
    }
    for path in named_inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_labels": output_directory / "legacy_training_labels_37.csv",
        "holdout_ffs": output_directory / "holdout_ffs_selection_6.csv",
        "holdout_tensor": output_directory / "holdout_ffs_tensor_6_3p1-4p8GHz.npz",
        "holdout_tensor_manifest": output_directory
        / "holdout_ffs_tensor_6_3p1-4p8GHz.manifest.json",
        "holdout_validation": output_directory / "legacy_roi_holdout_validation_6.csv",
        "summary": output_directory / "legacy_roi_holdout_summary.json",
    }
    roi_directory = output_directory / "roi_fit_training37"
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing or roi_directory.exists():
            first = existing[0] if existing else roi_directory
            raise FileExistsError(f"legacy ROI output exists: {first}")

    training = _build_training_labels()
    holdout_ffs = _build_holdout_ffs_selection()
    _write_csv_atomic(training, paths["training_labels"])
    _write_csv_atomic(holdout_ffs, paths["holdout_ffs"])
    prepare_tensor(
        paths["holdout_ffs"],
        paths["holdout_tensor"],
        paths["holdout_tensor_manifest"],
        band_ghz=(3.1, 4.8),
        overwrite=overwrite,
    )
    fitted = run_search(
        TRAINING_TENSOR,
        paths["training_labels"],
        roi_directory,
        field_family=FIELD_FAMILY,
        seed_theta_bounds=(60.0, 120.0),
        seed_frequency_bounds=(3.6, 4.2),
        seed_phi_deg=90.0,
        overwrite=overwrite,
    )
    optimum = dict(fitted["optimal_roi"])
    optimum["component"] = str(fitted["component"])
    holdout = _evaluate_frozen_roi(paths["holdout_tensor"], optimum)
    holdout = holdout.merge(_holdout_targets(), on="case_name", validate="one_to_one")
    holdout = holdout.merge(
        holdout_ffs[["link_case_name", "source_case_id"]].rename(
            columns={"link_case_name": "case_name"}
        ),
        on="case_name",
        validate="one_to_one",
    )
    holdout["roi_rank"] = holdout["legacy_roi_metric_linear"].rank(
        method="min", ascending=False
    ).astype(int)
    holdout["legacy_s21_rank"] = holdout["mean_s21_linear"].rank(
        method="min", ascending=False
    ).astype(int)
    holdout["legacy_ber_rank"] = holdout["legacy_ber_ebn0_at_target_db"].rank(
        method="min", ascending=True
    ).astype(int)
    holdout.sort_values("roi_rank", kind="stable", inplace=True)
    _write_csv_atomic(holdout, paths["holdout_validation"])

    rho_s21 = _spearman(
        holdout["legacy_roi_metric_linear"], holdout["mean_s21_linear"]
    )
    rho_ber = _spearman(
        holdout["legacy_roi_metric_linear"],
        -holdout["legacy_ber_ebn0_at_target_db"],
    )
    exact_s21_order = bool((holdout["roi_rank"] == holdout["legacy_s21_rank"]).all())
    exact_ber_order = bool((holdout["roi_rank"] == holdout["legacy_ber_rank"]).all())
    payload: dict[str, Any] = {
        "schema": "msabp.legacy_roi_fit37_holdout6.v1",
        "leakage_control": {
            "training_cases": EXPECTED_TRAINING_CASES,
            "holdout_cases": EXPECTED_HOLDOUT_CASES,
            "holdout_used_during_roi_search": False,
        },
        "targets": {
            "s21": "uniform arithmetic mean of linear |S21| over 3.1-4.8 GHz",
            "ber": "single 3.1-4.8 GHz UWB pulse at 5 Mbps; Eb/N0 at BER=1e-4",
        },
        "field_family": FIELD_FAMILY,
        "fitted_roi": optimum,
        "training_fit": {
            "rho_s21": float(fitted["optimal_roi"]["rho_s21_power"]),
            "rho_negative_ebn0": float(
                fitted["optimal_roi"]["rho_negative_ebn0"]
            ),
            "robust_joint_rho": float(fitted["optimal_roi"]["robust_joint_rho"]),
        },
        "locked_holdout": {
            "rho_s21": rho_s21,
            "rho_negative_ebn0": rho_ber,
            "joint_rho": min(rho_s21, rho_ber),
            "exact_s21_order": exact_s21_order,
            "exact_ber_order": exact_ber_order,
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in named_inputs.items()
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if name != "summary" and path.is_file()
        },
        "roi_directory": str(roi_directory),
    }
    _write_json_atomic(payload, paths["summary"])

    print(
        "[LegacyROI] fitted on 37: "
        f"{optimum['component']}, theta=[{optimum['theta_min_deg']:.0f},"
        f"{optimum['theta_max_deg']:.0f}] deg, "
        f"phi=90+/-{optimum['phi_half_width_deg']:.0f} deg, "
        f"f=[{optimum['frequency_min_ghz']:.1f},"
        f"{optimum['frequency_max_ghz']:.1f}] GHz"
    )
    print(
        "[LegacyROI] locked holdout-6: "
        f"rho(S21)={rho_s21:.4f}, rho(-Eb/N0)={rho_ber:.4f}"
    )
    print(
        holdout[
            [
                "source_case_id",
                "roi_rank",
                "legacy_s21_rank",
                "legacy_ber_rank",
            ]
        ].to_string(index=False)
    )
    print(f"[LegacyROI] outputs -> {output_directory}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(args.output_directory.expanduser().resolve(), overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
