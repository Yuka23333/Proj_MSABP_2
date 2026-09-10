"""Validate batch 1 and refit the spherical link ROI on all measured cases.

The frozen 13-case ROI and predictor are evaluated on the new 12 propagation
cases before any refitting.  The script then joins the original and new labels,
builds a 25-case FFS tensor, and runs the same constrained spherical ROI search
for the next active-learning round.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing.prepare_link_ffs_tensor import prepare_tensor  # noqa: E402
from scripts.postprocessing.search_link_spherical_roi import run_search  # noqa: E402


PROCESSED_ROOT = REPOSITORY_ROOT / "results" / "processed"
PROXY_ROOT = PROCESSED_ROOT / "onbody_proxies"
ACTIVE_ROOT = PROXY_ROOT / "active_learning_batch01"
DEFAULT_WORKLIST = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_active_learning_12_batch01.csv"
)
DEFAULT_SELECTION_AUDIT = ACTIVE_ROOT / "selection_audit.csv"
DEFAULT_NEW_S21 = ACTIVE_ROOT / "actual_september_s21_lambda0p5.csv"
DEFAULT_NEW_BER = (
    ACTIVE_ROOT
    / "actual_ber_v2_ieee802156_3ch_lambda0p5"
    / "case_summary.csv"
)
DEFAULT_OLD_SELECTED = (
    PROXY_ROOT / "link_correlation_13" / "selected_13_proxy_frequency_table.csv"
)
DEFAULT_OLD_LINK = (
    PROCESSED_ROOT
    / "propagation_s21_12_medoids_plus_roblin_wei_reference"
    / "metrics"
    / "september_s21_and_ebn0_lambda0p5.csv"
)
DEFAULT_OLD_ROI = PROXY_ROOT / "spherical_roi_13" / "optimal_roi_case_metrics.csv"
DEFAULT_OUTPUT_DIRECTORY = ACTIVE_ROOT / "mushroom_analysis_batch01"
EXPECTED_NEW_CASES = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_json_atomic(payload: dict[str, Any], path: Path) -> Path:
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
    return path


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")


def _same_cases(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    actual = set(frame["case_name"].astype(str))
    if actual != expected:
        raise ValueError(
            f"{label} case set mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _rank_correlation(x: pd.Series, y: pd.Series, kind: str) -> float:
    left = x.to_numpy(dtype=np.float64)
    right = y.to_numpy(dtype=np.float64)
    if kind == "spearman":
        value = stats.spearmanr(left, right).statistic
    elif kind == "kendall":
        value = stats.kendalltau(left, right).statistic
    else:
        raise ValueError(f"unknown rank correlation: {kind}")
    return float(value)


def _validation_summary(group_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    actual_log_s21 = np.log10(
        frame["s21_power_september"].to_numpy(dtype=np.float64)
    )
    predicted_log_s21 = np.log10(
        frame["predicted_s21_power"].to_numpy(dtype=np.float64)
    )
    ebn0_error = (
        frame["predicted_ebn0_db"].to_numpy(dtype=np.float64)
        - frame["ebn0_ber_v2_at_target_db"].to_numpy(dtype=np.float64)
    )
    log_s21_error = predicted_log_s21 - actual_log_s21
    return {
        "group": group_name,
        "case_count": len(frame),
        "frozen_roi_spearman_s21": _rank_correlation(
            frame["roi_metric_linear"], frame["s21_power_september"], "spearman"
        ),
        "frozen_roi_spearman_negative_ebn0": _rank_correlation(
            frame["roi_metric_linear"],
            -frame["ebn0_ber_v2_at_target_db"],
            "spearman",
        ),
        "predictor_spearman_s21": _rank_correlation(
            frame["predicted_s21_power"],
            frame["s21_power_september"],
            "spearman",
        ),
        "predictor_spearman_negative_ebn0": _rank_correlation(
            -frame["predicted_ebn0_db"],
            -frame["ebn0_ber_v2_at_target_db"],
            "spearman",
        ),
        "predictor_kendall_s21": _rank_correlation(
            frame["predicted_s21_power"],
            frame["s21_power_september"],
            "kendall",
        ),
        "predictor_kendall_negative_ebn0": _rank_correlation(
            -frame["predicted_ebn0_db"],
            -frame["ebn0_ber_v2_at_target_db"],
            "kendall",
        ),
        "log10_s21_rmse": float(np.sqrt(np.mean(log_s21_error**2))),
        "log10_s21_bias_predicted_minus_actual": float(np.mean(log_s21_error)),
        "s21_multiplicative_rmse_factor": float(
            10.0 ** np.sqrt(np.mean(log_s21_error**2))
        ),
        "ebn0_rmse_db": float(np.sqrt(np.mean(ebn0_error**2))),
        "ebn0_mae_db": float(np.mean(np.abs(ebn0_error))),
        "ebn0_bias_predicted_minus_actual_db": float(np.mean(ebn0_error)),
    }


def _roi_score(group_name: str, roi_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "roi": roi_name,
        "evaluation_group": group_name,
        "case_count": len(frame),
        "rho_s21_power": _rank_correlation(
            frame["roi_metric_linear"], frame["s21_power_september"], "spearman"
        ),
        "rho_negative_ebn0": _rank_correlation(
            frame["roi_metric_linear"],
            -frame["ebn0_ber_v2_at_target_db"],
            "spearman",
        ),
    }


def _load_frozen_validation(
    worklist_path: Path,
    selection_audit_path: Path,
    new_s21_path: Path,
    new_ber_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    worklist = pd.read_csv(worklist_path)
    audit = pd.read_csv(selection_audit_path)
    s21 = pd.read_csv(new_s21_path)
    ber = pd.read_csv(new_ber_path)
    _require_columns(
        worklist,
        {
            "sample_id",
            "selection_rank",
            "acquisition_type",
            "source",
            "source_case_id",
            "source_ffs_sha256",
            "predicted_s21_power",
            "predicted_ebn0_db",
            "roi_metric_linear",
        },
        "worklist",
    )
    _require_columns(
        audit,
        {"selection_rank", "source", "case_id", "ffs_path", "ffs_sha256"},
        "selection audit",
    )
    _require_columns(s21, {"case_name", "s21_power_september"}, "new S21 table")
    _require_columns(
        ber,
        {
            "case_name",
            "ebn0_ch0_at_target_db",
            "ebn0_ch1_mandatory_at_target_db",
            "ebn0_ch2_at_target_db",
            "ebn0_ber_v2_at_target_db",
        },
        "new BER table",
    )
    if len(worklist) != EXPECTED_NEW_CASES:
        raise ValueError(
            f"expected {EXPECTED_NEW_CASES} batch-1 cases, got {len(worklist)}"
        )
    if worklist["sample_id"].duplicated().any():
        raise ValueError("worklist sample_id must be unique")
    worklist = worklist.copy()
    worklist["case_name"] = "case_" + worklist["sample_id"].astype(str)
    expected = set(worklist["case_name"])
    _same_cases(s21, expected, "new S21 table")
    _same_cases(ber, expected, "new BER table")

    audit_fields = audit[
        ["selection_rank", "source", "case_id", "ffs_path", "ffs_sha256"]
    ].rename(
        columns={
            "source": "audit_source",
            "case_id": "audit_source_case_id",
        }
    )
    joined = worklist.merge(
        audit_fields,
        on="selection_rank",
        how="left",
        validate="one_to_one",
    )
    if joined["ffs_path"].isna().any():
        raise ValueError("selection audit does not cover every worklist row")
    if not (
        joined["source"].astype(str) == joined["audit_source"].astype(str)
    ).all():
        raise ValueError("worklist and selection audit source disagree")
    if not (
        joined["source_case_id"].astype(str)
        == joined["audit_source_case_id"].astype(str)
    ).all():
        raise ValueError("worklist and selection audit source case disagree")
    if not (
        joined["source_ffs_sha256"].astype(str).str.casefold()
        == joined["ffs_sha256"].astype(str).str.casefold()
    ).all():
        raise ValueError("worklist and selection audit FFS hash disagree")

    frozen = joined.merge(s21, on="case_name", how="inner", validate="one_to_one")
    frozen = frozen.merge(ber, on="case_name", how="inner", validate="one_to_one")
    frozen["actual_log10_s21_power"] = np.log10(frozen["s21_power_september"])
    frozen["predicted_log10_s21_power_recomputed"] = np.log10(
        frozen["predicted_s21_power"]
    )
    frozen["log10_s21_error_predicted_minus_actual"] = (
        frozen["predicted_log10_s21_power_recomputed"]
        - frozen["actual_log10_s21_power"]
    )
    frozen["ebn0_error_predicted_minus_actual_db"] = (
        frozen["predicted_ebn0_db"] - frozen["ebn0_ber_v2_at_target_db"]
    )
    return frozen, joined


def run_analysis(
    *,
    worklist_path: Path,
    selection_audit_path: Path,
    new_s21_path: Path,
    new_ber_path: Path,
    old_selected_path: Path,
    old_link_path: Path,
    old_roi_path: Path,
    output_directory: Path,
    overwrite: bool,
) -> dict[str, Any]:
    inputs = [
        worklist_path,
        selection_audit_path,
        new_s21_path,
        new_ber_path,
        old_selected_path,
        old_link_path,
        old_roi_path,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "frozen_validation": output_directory / "frozen_validation_12.csv",
        "validation_summary": output_directory / "validation_summary.csv",
        "combined_ffs": output_directory / "combined_ffs_selection_25.csv",
        "combined_link": output_directory / "combined_link_metrics_25.csv",
        "tensor": output_directory / "link_ffs_tensor_25_3p1-4p8GHz.npz",
        "tensor_manifest": output_directory
        / "link_ffs_tensor_25_3p1-4p8GHz.manifest.json",
        "roi_scores": output_directory / "roi_evaluation.csv",
        "manifest": output_directory / "analysis_manifest.json",
    }
    roi_directory = output_directory / "spherical_roi_25_batch01"
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing or roi_directory.exists():
            first = existing[0] if existing else roi_directory
            raise FileExistsError(f"analysis output exists: {first}")

    frozen, joined = _load_frozen_validation(
        worklist_path,
        selection_audit_path,
        new_s21_path,
        new_ber_path,
    )
    summary_rows = [_validation_summary("all_new_12", frozen)]
    for acquisition_type in ("calibration", "exploitation"):
        summary_rows.append(
            _validation_summary(
                acquisition_type,
                frozen[frozen["acquisition_type"] == acquisition_type],
            )
        )
    summary = pd.DataFrame(summary_rows)
    _write_csv_atomic(frozen, paths["frozen_validation"])
    _write_csv_atomic(summary, paths["validation_summary"])

    old_selected = pd.read_csv(old_selected_path)
    _require_columns(
        old_selected,
        {"link_case_name", "ffs_path", "ffs_sha256"},
        "old selected table",
    )
    old_ffs = old_selected[
        ["link_case_name", "ffs_path", "ffs_sha256"]
    ].drop_duplicates("link_case_name")
    new_ffs = joined[["case_name", "ffs_path", "ffs_sha256"]].rename(
        columns={"case_name": "link_case_name"}
    )
    combined_ffs = pd.concat([old_ffs, new_ffs], ignore_index=True)
    if combined_ffs["link_case_name"].duplicated().any() or len(combined_ffs) != 25:
        raise ValueError("combined FFS selection must contain 25 unique cases")
    _write_csv_atomic(combined_ffs, paths["combined_ffs"])

    old_link = pd.read_csv(old_link_path)
    _require_columns(
        old_link,
        {"case_name", "s21_power_september", "ebn0_ber_v2_at_target_db"},
        "old link table",
    )
    old_link = old_link.copy()
    old_link["cohort"] = "original_13"
    old_link["acquisition_type"] = "original"
    new_link = frozen.copy()
    new_link["cohort"] = "active_learning_batch01"
    combined_link = pd.concat([old_link, new_link], ignore_index=True, sort=False)
    if combined_link["case_name"].duplicated().any() or len(combined_link) != 25:
        raise ValueError("combined link table must contain 25 unique cases")
    combined_link["link_rank_lambda0p5"] = (
        combined_link["ebn0_ber_v2_at_target_db"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    combined_link = combined_link.sort_values(
        "link_rank_lambda0p5", kind="stable"
    ).reset_index(drop=True)
    _write_csv_atomic(combined_link, paths["combined_link"])

    prepare_tensor(
        paths["combined_ffs"],
        paths["tensor"],
        paths["tensor_manifest"],
        band_ghz=(3.1, 4.8),
        overwrite=overwrite,
    )
    updated_roi = run_search(
        paths["tensor"],
        paths["combined_link"],
        roi_directory,
        field_family="directivity",
        seed_theta_bounds=(60.0, 120.0),
        seed_frequency_bounds=(3.6, 4.2),
        seed_phi_deg=90.0,
        overwrite=overwrite,
    )

    old_roi = pd.read_csv(old_roi_path)
    old_roi["cohort"] = "original_13"
    old_frozen_new = frozen[
        [
            "case_name",
            "roi_metric_linear",
            "s21_power_september",
            "ebn0_ber_v2_at_target_db",
        ]
    ].copy()
    old_frozen_new["cohort"] = "active_learning_batch01"
    old_roi_all = pd.concat(
        [
            old_roi[
                [
                    "case_name",
                    "roi_metric_linear",
                    "s21_power_september",
                    "ebn0_ber_v2_at_target_db",
                    "cohort",
                ]
            ],
            old_frozen_new,
        ],
        ignore_index=True,
    )
    updated_case_metrics = pd.read_csv(
        roi_directory / "optimal_roi_case_metrics.csv"
    ).merge(
        combined_link[["case_name", "cohort"]],
        on="case_name",
        how="left",
        validate="one_to_one",
    )
    roi_rows: list[dict[str, Any]] = []
    for roi_name, table in (
        ("frozen_old_13_roi", old_roi_all),
        ("refit_all_25_roi", updated_case_metrics),
    ):
        roi_rows.append(_roi_score("all_25", roi_name, table))
        for cohort in ("original_13", "active_learning_batch01"):
            roi_rows.append(
                _roi_score(
                    cohort,
                    roi_name,
                    table[table["cohort"] == cohort],
                )
            )
    roi_scores = pd.DataFrame(roi_rows)
    _write_csv_atomic(roi_scores, paths["roi_scores"])

    payload: dict[str, Any] = {
        "schema": "msabp.mushroom_analysis.batch01.v1",
        "phantom_scope": (
            "open-boundary infinite planar Muscle phantom; conclusions are not "
            "validated for a finite cylindrical phantom"
        ),
        "case_counts": {"original": 13, "new": 12, "combined": 25},
        "frozen_validation": summary_rows[0],
        "updated_roi": {
            "component": updated_roi["component"],
            "seed": updated_roi["seed"],
            "optimal_roi": updated_roi["optimal_roi"],
            "status": "refit_in_sample_requires_next_batch_validation",
        },
        "inputs": {
            str(path.name): {"path": str(path), "sha256": _sha256(path)}
            for path in inputs
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if name != "manifest" and path.is_file()
        },
        "roi_directory": str(roi_directory),
    }
    _write_json_atomic(payload, paths["manifest"])
    print(
        "[Mushroom] frozen new-12 rho: "
        f"S21={summary_rows[0]['frozen_roi_spearman_s21']:.4f}, "
        f"-Eb/N0={summary_rows[0]['frozen_roi_spearman_negative_ebn0']:.4f}"
    )
    optimum = updated_roi["optimal_roi"]
    print(
        "[Mushroom] refit ROI: "
        f"{updated_roi['component']}, "
        f"theta=[{optimum['theta_min_deg']:.0f},{optimum['theta_max_deg']:.0f}] deg, "
        f"phi=90+/-{optimum['phi_half_width_deg']:.0f} deg, "
        f"f=[{optimum['frequency_min_ghz']:.1f},"
        f"{optimum['frequency_max_ghz']:.1f}] GHz"
    )
    print(f"[Mushroom] outputs -> {output_directory}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument(
        "--selection-audit", type=Path, default=DEFAULT_SELECTION_AUDIT
    )
    parser.add_argument("--new-s21", type=Path, default=DEFAULT_NEW_S21)
    parser.add_argument("--new-ber", type=Path, default=DEFAULT_NEW_BER)
    parser.add_argument("--old-selected", type=Path, default=DEFAULT_OLD_SELECTED)
    parser.add_argument("--old-link", type=Path, default=DEFAULT_OLD_LINK)
    parser.add_argument("--old-roi", type=Path, default=DEFAULT_OLD_ROI)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(
        worklist_path=args.worklist.expanduser().resolve(),
        selection_audit_path=args.selection_audit.expanduser().resolve(),
        new_s21_path=args.new_s21.expanduser().resolve(),
        new_ber_path=args.new_ber.expanduser().resolve(),
        old_selected_path=args.old_selected.expanduser().resolve(),
        old_link_path=args.old_link.expanduser().resolve(),
        old_roi_path=args.old_roi.expanduser().resolve(),
        output_directory=args.output_directory.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
