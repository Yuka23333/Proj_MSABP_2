"""Validate active-learning batch 2 and refit the spherical ROI on 37 cases.

Frozen 25-case predictions and both prior ROI definitions are evaluated on the
new 12 cases before all measured cases are combined.  The final 37-case ROI is
therefore reported as an in-sample refit, not as holdout performance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing import (  # noqa: E402
    analyze_propagation_active_learning_batch01 as batch01,
)
from scripts.postprocessing.prepare_link_ffs_tensor import prepare_tensor  # noqa: E402
from scripts.postprocessing.search_link_spherical_roi import run_search  # noqa: E402


PROCESSED_ROOT = REPOSITORY_ROOT / "results" / "processed"
PROXY_ROOT = PROCESSED_ROOT / "onbody_proxies"
BATCH01_ROOT = PROXY_ROOT / "active_learning_batch01"
BATCH01_ANALYSIS_ROOT = BATCH01_ROOT / "mushroom_analysis_batch01"
BATCH02_ROOT = PROXY_ROOT / "active_learning_batch02"

DEFAULT_WORKLIST = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_active_learning_12_batch02.csv"
)
DEFAULT_SELECTION_AUDIT = BATCH02_ROOT / "selection_audit.csv"
DEFAULT_NEW_S21 = BATCH02_ROOT / "actual_september_s21_lambda0p5.csv"
DEFAULT_NEW_BER = (
    BATCH02_ROOT
    / "actual_ber_v2_ieee802156_3ch_lambda0p5"
    / "case_summary.csv"
)
DEFAULT_PRIOR_FFS = BATCH01_ANALYSIS_ROOT / "combined_ffs_selection_25.csv"
DEFAULT_PRIOR_LINK = BATCH01_ANALYSIS_ROOT / "combined_link_metrics_25.csv"
DEFAULT_OLD13_ROI = PROXY_ROOT / "spherical_roi_13" / "optimal_roi_case_metrics.csv"
DEFAULT_BATCH01_FROZEN = BATCH01_ANALYSIS_ROOT / "frozen_validation_12.csv"
DEFAULT_ROI25 = (
    BATCH01_ANALYSIS_ROOT
    / "spherical_roi_25_batch01"
    / "optimal_roi_case_metrics.csv"
)
DEFAULT_BATCH01_SUMMARY = BATCH01_ANALYSIS_ROOT / "validation_summary.csv"
DEFAULT_OUTPUT_DIRECTORY = BATCH02_ROOT / "mushroom_analysis_batch02"
EXPECTED_NEW_CASES = 12
EXPECTED_PRIOR_CASES = 25
EXPECTED_COMBINED_CASES = 37


def _load_batch02(
    worklist_path: Path,
    selection_audit_path: Path,
    new_s21_path: Path,
    new_ber_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    worklist = pd.read_csv(worklist_path)
    audit = pd.read_csv(selection_audit_path)
    s21 = pd.read_csv(new_s21_path)
    ber = pd.read_csv(new_ber_path)
    batch01._require_columns(
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
            "uncertainty_log10_s21_power",
            "uncertainty_negative_ebn0_db",
            "old_roi_metric_linear",
            "new_roi_metric_linear",
        },
        "batch-2 worklist",
    )
    batch01._require_columns(
        audit,
        {"selection_rank", "source", "case_id", "ffs_path", "ffs_sha256"},
        "batch-2 selection audit",
    )
    batch01._require_columns(
        s21,
        {"case_name", "s21_power_september", "s21_power_september_db"},
        "batch-2 S21 table",
    )
    batch01._require_columns(
        ber,
        {
            "case_name",
            "ebn0_ch0_at_target_db",
            "ebn0_ch1_mandatory_at_target_db",
            "ebn0_ch2_at_target_db",
            "ebn0_ber_v2_at_target_db",
        },
        "batch-2 BER table",
    )
    if len(worklist) != EXPECTED_NEW_CASES:
        raise ValueError(
            f"expected {EXPECTED_NEW_CASES} batch-2 cases, got {len(worklist)}"
        )
    if worklist["sample_id"].duplicated().any():
        raise ValueError("batch-2 worklist sample_id must be unique")

    worklist = worklist.copy()
    worklist["case_name"] = "case_" + worklist["sample_id"].astype(str)
    expected = set(worklist["case_name"])
    batch01._same_cases(s21, expected, "batch-2 S21 table")
    batch01._same_cases(ber, expected, "batch-2 BER table")
    if ber["ebn0_ber_v2_at_target_db"].isna().any():
        missing = sorted(
            ber.loc[ber["ebn0_ber_v2_at_target_db"].isna(), "case_name"]
        )
        raise ValueError(f"batch-2 BER target is not bracketed for: {missing}")

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
        raise ValueError("selection audit does not cover every batch-2 row")
    if not (
        joined["source"].astype(str) == joined["audit_source"].astype(str)
    ).all():
        raise ValueError("batch-2 worklist and audit source disagree")
    if not (
        joined["source_case_id"].astype(str)
        == joined["audit_source_case_id"].astype(str)
    ).all():
        raise ValueError("batch-2 worklist and audit source case disagree")
    if not (
        joined["source_ffs_sha256"].astype(str).str.casefold()
        == joined["ffs_sha256"].astype(str).str.casefold()
    ).all():
        raise ValueError("batch-2 worklist and audit FFS hash disagree")

    frozen = joined.merge(s21, on="case_name", how="inner", validate="one_to_one")
    frozen = frozen.merge(ber, on="case_name", how="inner", validate="one_to_one")
    frozen["roi_metric_linear"] = frozen["new_roi_metric_linear"]
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


def _uncertainty_summary(frame: pd.DataFrame) -> dict[str, Any]:
    s21_error = frame["log10_s21_error_predicted_minus_actual"].abs()
    ebn0_error = frame["ebn0_error_predicted_minus_actual_db"].abs()
    return {
        "case_count": len(frame),
        "spearman_uncertainty_vs_abs_log10_s21_error": batch01._rank_correlation(
            frame["uncertainty_log10_s21_power"], s21_error, "spearman"
        ),
        "spearman_uncertainty_vs_abs_ebn0_error": batch01._rank_correlation(
            frame["uncertainty_negative_ebn0_db"], ebn0_error, "spearman"
        ),
        "interpretation": (
            "bootstrap dispersion is tested only as an error-ranking heuristic; "
            "it is not treated as a calibrated predictive interval"
        ),
    }


def _assemble_roi_table(
    name: str,
    frames: Sequence[pd.DataFrame],
    cohort_by_case: pd.Series,
) -> pd.DataFrame:
    table = pd.concat(frames, ignore_index=True)
    if len(table) != EXPECTED_COMBINED_CASES or table["case_name"].duplicated().any():
        raise ValueError(f"{name} must cover 37 unique cases")
    table = table[["case_name", "roi_metric_linear"]].copy()
    table["cohort"] = table["case_name"].map(cohort_by_case)
    if table["cohort"].isna().any():
        raise ValueError(f"{name} contains cases absent from combined link labels")
    return table


def run_analysis(
    *,
    worklist_path: Path,
    selection_audit_path: Path,
    new_s21_path: Path,
    new_ber_path: Path,
    prior_ffs_path: Path,
    prior_link_path: Path,
    old13_roi_path: Path,
    batch01_frozen_path: Path,
    roi25_path: Path,
    batch01_summary_path: Path,
    output_directory: Path,
    overwrite: bool,
) -> dict[str, Any]:
    inputs = [
        worklist_path,
        selection_audit_path,
        new_s21_path,
        new_ber_path,
        prior_ffs_path,
        prior_link_path,
        old13_roi_path,
        batch01_frozen_path,
        roi25_path,
        batch01_summary_path,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "frozen_validation": output_directory / "frozen_validation_12.csv",
        "validation_summary": output_directory / "validation_summary.csv",
        "batch_comparison": output_directory / "batch_validation_comparison.csv",
        "frozen_roi_holdout": output_directory / "frozen_roi_holdout_evaluation.csv",
        "combined_ffs": output_directory / "combined_ffs_selection_37.csv",
        "combined_link": output_directory / "combined_link_metrics_37.csv",
        "tensor": output_directory / "link_ffs_tensor_37_3p1-4p8GHz.npz",
        "tensor_manifest": output_directory
        / "link_ffs_tensor_37_3p1-4p8GHz.manifest.json",
        "roi_scores": output_directory / "roi_evaluation.csv",
        "manifest": output_directory / "analysis_manifest.json",
    }
    roi_directory = output_directory / "spherical_roi_37_batch02"
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing or roi_directory.exists():
            first = existing[0] if existing else roi_directory
            raise FileExistsError(f"analysis output exists: {first}")

    frozen, joined = _load_batch02(
        worklist_path,
        selection_audit_path,
        new_s21_path,
        new_ber_path,
    )
    summary_rows = [batch01._validation_summary("all_new_12", frozen)]
    for acquisition_type in (
        "roi_disagreement",
        "link_disagreement",
        "uncertainty",
        "exploitation",
    ):
        summary_rows.append(
            batch01._validation_summary(
                acquisition_type,
                frozen[frozen["acquisition_type"] == acquisition_type],
            )
        )
    summary = pd.DataFrame(summary_rows)
    batch01._write_csv_atomic(frozen, paths["frozen_validation"])
    batch01._write_csv_atomic(summary, paths["validation_summary"])

    prior_summary = pd.read_csv(batch01_summary_path)
    prior_all = prior_summary.loc[prior_summary["group"] == "all_new_12"].copy()
    prior_all.insert(0, "validation_batch", "batch01_frozen_13_case_model")
    current_all = summary.loc[summary["group"] == "all_new_12"].copy()
    current_all.insert(0, "validation_batch", "batch02_frozen_25_case_model")
    comparison = pd.concat([prior_all, current_all], ignore_index=True, sort=False)
    batch01._write_csv_atomic(comparison, paths["batch_comparison"])

    frozen_roi_rows = []
    for roi_name, column in (
        ("frozen_old_13_roi", "old_roi_metric_linear"),
        ("frozen_refit_25_roi", "new_roi_metric_linear"),
    ):
        table = frozen.copy()
        table["roi_metric_linear"] = table[column]
        frozen_roi_rows.append(
            batch01._roi_score("active_learning_batch02", roi_name, table)
        )
    batch01._write_csv_atomic(
        pd.DataFrame(frozen_roi_rows), paths["frozen_roi_holdout"]
    )

    prior_ffs = pd.read_csv(prior_ffs_path)
    batch01._require_columns(
        prior_ffs,
        {"link_case_name", "ffs_path", "ffs_sha256"},
        "prior 25-case FFS table",
    )
    if len(prior_ffs) != EXPECTED_PRIOR_CASES:
        raise ValueError(f"expected 25 prior FFS cases, got {len(prior_ffs)}")
    new_ffs = joined[["case_name", "ffs_path", "ffs_sha256"]].rename(
        columns={"case_name": "link_case_name"}
    )
    combined_ffs = pd.concat([prior_ffs, new_ffs], ignore_index=True)
    if (
        len(combined_ffs) != EXPECTED_COMBINED_CASES
        or combined_ffs["link_case_name"].duplicated().any()
    ):
        raise ValueError("combined FFS selection must contain 37 unique cases")
    batch01._write_csv_atomic(combined_ffs, paths["combined_ffs"])

    prior_link = pd.read_csv(prior_link_path)
    if len(prior_link) != EXPECTED_PRIOR_CASES:
        raise ValueError(f"expected 25 prior link cases, got {len(prior_link)}")
    new_link = frozen.copy()
    new_link["cohort"] = "active_learning_batch02"
    combined_link = pd.concat([prior_link, new_link], ignore_index=True, sort=False)
    if (
        len(combined_link) != EXPECTED_COMBINED_CASES
        or combined_link["case_name"].duplicated().any()
    ):
        raise ValueError("combined link table must contain 37 unique cases")
    combined_link["link_rank_lambda0p5"] = (
        combined_link["ebn0_ber_v2_at_target_db"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    combined_link = combined_link.sort_values(
        "link_rank_lambda0p5", kind="stable"
    ).reset_index(drop=True)
    batch01._write_csv_atomic(combined_link, paths["combined_link"])

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

    cohort_by_case = combined_link.set_index("case_name")["cohort"]
    old13_original = pd.read_csv(old13_roi_path)
    batch01_frozen = pd.read_csv(batch01_frozen_path)
    roi25 = pd.read_csv(roi25_path)
    batch02_old = frozen[["case_name", "old_roi_metric_linear"]].rename(
        columns={"old_roi_metric_linear": "roi_metric_linear"}
    )
    batch02_new = frozen[["case_name", "new_roi_metric_linear"]].rename(
        columns={"new_roi_metric_linear": "roi_metric_linear"}
    )
    old13_all = _assemble_roi_table(
        "frozen old-13 ROI",
        [old13_original, batch01_frozen, batch02_old],
        cohort_by_case,
    )
    roi25_all = _assemble_roi_table(
        "frozen refit-25 ROI",
        [roi25, batch02_new],
        cohort_by_case,
    )
    link_labels = combined_link[
        ["case_name", "s21_power_september", "ebn0_ber_v2_at_target_db"]
    ]
    old13_all = old13_all.merge(
        link_labels,
        on="case_name",
        how="left",
        validate="one_to_one",
    )
    roi25_all = roi25_all.merge(
        link_labels,
        on="case_name",
        how="left",
        validate="one_to_one",
    )
    roi37 = pd.read_csv(roi_directory / "optimal_roi_case_metrics.csv").merge(
        combined_link[["case_name", "cohort"]],
        on="case_name",
        how="left",
        validate="one_to_one",
    )
    roi_rows: list[dict[str, Any]] = []
    for roi_name, table in (
        ("frozen_old_13_roi", old13_all),
        ("frozen_refit_25_roi", roi25_all),
        ("refit_all_37_roi", roi37),
    ):
        roi_rows.append(batch01._roi_score("all_37", roi_name, table))
        for cohort in (
            "original_13",
            "active_learning_batch01",
            "active_learning_batch02",
        ):
            roi_rows.append(
                batch01._roi_score(
                    cohort,
                    roi_name,
                    table[table["cohort"] == cohort],
                )
            )
    batch01._write_csv_atomic(pd.DataFrame(roi_rows), paths["roi_scores"])

    payload: dict[str, Any] = {
        "schema": "msabp.mushroom_analysis.batch02.v1",
        "phantom_scope": (
            "open-boundary infinite planar Muscle phantom; conclusions are not "
            "validated for a finite cylindrical phantom"
        ),
        "case_counts": {"prior": 25, "new": 12, "combined": 37},
        "frozen_validation": summary_rows[0],
        "frozen_roi_holdout": frozen_roi_rows,
        "uncertainty_diagnostic": _uncertainty_summary(frozen),
        "updated_roi": {
            "component": updated_roi["component"],
            "seed": updated_roi["seed"],
            "optimal_roi": updated_roi["optimal_roi"],
            "status": "refit_in_sample_requires_future_holdout_validation",
        },
        "ber_postprocessing": {
            "robustness_lambda": 0.5,
            "target_ber": 1.0e-4,
            "ebn0_grid_db": list(range(40, 72, 2)),
            "grid_extended_for_batch02": True,
        },
        "inputs": {
            str(path.name): {"path": str(path), "sha256": batch01._sha256(path)}
            for path in inputs
        },
        "outputs": {
            name: {"path": str(path), "sha256": batch01._sha256(path)}
            for name, path in paths.items()
            if name != "manifest" and path.is_file()
        },
        "roi_directory": str(roi_directory),
    }
    batch01._write_json_atomic(payload, paths["manifest"])

    frozen_summary = summary_rows[0]
    print(
        "[Mushroom-02] frozen 25-case predictor on new 12: "
        f"rho(S21)={frozen_summary['predictor_spearman_s21']:.4f}, "
        f"rho(-Eb/N0)={frozen_summary['predictor_spearman_negative_ebn0']:.4f}"
    )
    print(
        "[Mushroom-02] frozen ROI holdout: "
        f"old13 rho={frozen_roi_rows[0]['rho_s21_power']:.4f}/"
        f"{frozen_roi_rows[0]['rho_negative_ebn0']:.4f}, "
        f"refit25 rho={frozen_roi_rows[1]['rho_s21_power']:.4f}/"
        f"{frozen_roi_rows[1]['rho_negative_ebn0']:.4f}"
    )
    optimum = updated_roi["optimal_roi"]
    print(
        "[Mushroom-02] refit ROI: "
        f"{updated_roi['component']}, "
        f"theta=[{optimum['theta_min_deg']:.0f},{optimum['theta_max_deg']:.0f}] deg, "
        f"phi=90+/-{optimum['phi_half_width_deg']:.0f} deg, "
        f"f=[{optimum['frequency_min_ghz']:.1f},"
        f"{optimum['frequency_max_ghz']:.1f}] GHz"
    )
    print(f"[Mushroom-02] outputs -> {output_directory}")
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
    run_analysis(
        worklist_path=DEFAULT_WORKLIST,
        selection_audit_path=DEFAULT_SELECTION_AUDIT,
        new_s21_path=DEFAULT_NEW_S21,
        new_ber_path=DEFAULT_NEW_BER,
        prior_ffs_path=DEFAULT_PRIOR_FFS,
        prior_link_path=DEFAULT_PRIOR_LINK,
        old13_roi_path=DEFAULT_OLD13_ROI,
        batch01_frozen_path=DEFAULT_BATCH01_FROZEN,
        roi25_path=DEFAULT_ROI25,
        batch01_summary_path=DEFAULT_BATCH01_SUMMARY,
        output_directory=args.output_directory.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
