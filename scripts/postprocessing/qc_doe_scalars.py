"""Data QC for the 11-variable branch-up DOE scalar tables.

Checks the three objective columns (``s11_max_lin``, ``rad_eff_dB``,
``cap_gain_dBi``) exported alongside the 512-sample training set and the
64-sample holdout set:

* ``s11_max_lin`` must fall in (0, 1] -- it is |S11| on a linear scale, so it
  is strictly positive and bounded above by total reflection.
* ``rad_eff_dB`` must be <= 0 -- radiation efficiency in dB cannot exceed
  0 dB (100%).
* ``cap_gain_dBi`` is checked against a generous physically-plausible
  envelope for this antenna class (compact WBAN slot antenna: small,
  sometimes mismatched/lossy, so occasional slightly negative gain is
  expected, but double-digit gain is not). See ``CAP_GAIN_DBI_RANGE``.
* All three objective columns must be non-null in both tables.
* Each column is additionally checked for statistical outliers via the
  1.5x-IQR rule, on top of the hard physical-range checks above.

Nothing is dropped from the input tables. Every check only flags rows;
flagged ``sample_id`` values are written to a QC report CSV per table and
printed to the console.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# IDE / F5 configuration.
F5_TABLES = (
    REPOSITORY_ROOT / "results" / "processed" / "training_11var_lhs_512_scalars.csv",
    REPOSITORY_ROOT / "results" / "processed" / "training_11var_lhs_holdout_64_scalars.csv",
)
F5_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "processed" / "qc"

OBJECTIVE_COLUMNS = ("s11_max_lin", "rad_eff_dB", "cap_gain_dBi")

S11_MAX_LIN_RANGE = (0.0, 1.0)  # exclusive lower bound, inclusive upper bound
RAD_EFF_DB_MAX = 0.0  # inclusive upper bound
# Generous physical envelope for a compact WBAN slot antenna's peak
# realized gain; observed data sits well inside this (-3.0 .. 6.9 dBi
# across both tables), so this only catches genuinely implausible values.
CAP_GAIN_DBI_RANGE = (-10.0, 10.0)

IQR_MULTIPLIER = 1.5


@dataclass(frozen=True)
class ColumnCheckResult:
    column: str
    out_of_range: pd.Series
    outlier: pd.Series
    missing: pd.Series


def _iqr_outlier_mask(values: pd.Series) -> pd.Series:
    finite = values.dropna()
    q1, q3 = finite.quantile(0.25), finite.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    return (values < lower) | (values > upper)


def check_s11_max_lin(values: pd.Series) -> ColumnCheckResult:
    missing = values.isna()
    lower, upper = S11_MAX_LIN_RANGE
    out_of_range = ~missing & ((values <= lower) | (values > upper))
    outlier = _iqr_outlier_mask(values) & ~missing
    return ColumnCheckResult("s11_max_lin", out_of_range, outlier, missing)


def check_rad_eff_db(values: pd.Series) -> ColumnCheckResult:
    missing = values.isna()
    out_of_range = ~missing & (values > RAD_EFF_DB_MAX)
    outlier = _iqr_outlier_mask(values) & ~missing
    return ColumnCheckResult("rad_eff_dB", out_of_range, outlier, missing)


def check_cap_gain_dbi(values: pd.Series) -> ColumnCheckResult:
    missing = values.isna()
    lower, upper = CAP_GAIN_DBI_RANGE
    out_of_range = ~missing & ((values < lower) | (values > upper))
    outlier = _iqr_outlier_mask(values) & ~missing
    return ColumnCheckResult("cap_gain_dBi", out_of_range, outlier, missing)


CHECKS = {
    "s11_max_lin": check_s11_max_lin,
    "rad_eff_dB": check_rad_eff_db,
    "cap_gain_dBi": check_cap_gain_dbi,
}


def run_qc(table: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    """Return a per-row QC frame: sample_id + one flag/reason column per check."""

    missing_columns = [c for c in ("sample_id", *OBJECTIVE_COLUMNS) if c not in table.columns]
    if missing_columns:
        raise ValueError(f"{table_name}: missing required column(s): {missing_columns}")

    report = pd.DataFrame({"sample_id": table["sample_id"]})
    reasons = pd.Series([[] for _ in range(len(table))], index=table.index)

    for column, check in CHECKS.items():
        result = check(table[column])
        report[f"{column}_missing"] = result.missing
        report[f"{column}_out_of_range"] = result.out_of_range
        report[f"{column}_outlier"] = result.outlier

        for label, mask in (
            ("missing", result.missing),
            ("out_of_range", result.out_of_range),
            ("outlier", result.outlier),
        ):
            for index in table.index[mask]:
                reasons.loc[index].append(f"{column}:{label}")

    report["flagged"] = reasons.apply(len).gt(0)
    report["reasons"] = reasons.apply(lambda items: "; ".join(items))
    return report


def summarize(report: pd.DataFrame, *, table_name: str, row_count: int) -> str:
    flagged = report.loc[report["flagged"]]
    lines = [f"[{table_name}] {row_count} rows, {len(flagged)} flagged"]
    for column in OBJECTIVE_COLUMNS:
        n_missing = int(report[f"{column}_missing"].sum())
        n_range = int(report[f"{column}_out_of_range"].sum())
        n_outlier = int(report[f"{column}_outlier"].sum())
        lines.append(
            f"  {column}: missing={n_missing} out_of_range={n_range} outlier={n_outlier}"
        )
    if not flagged.empty:
        lines.append(f"  flagged sample_id: {flagged['sample_id'].tolist()}")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        action="append",
        type=Path,
        dest="tables",
        help="Scalar table CSV to QC; repeat for multiple tables.",
    )
    parser.add_argument("--output-dir", type=Path, default=F5_OUTPUT_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    tables = tuple(args.tables) if args.tables else F5_TABLES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    any_flagged = False
    for table_path in tables:
        table_path = Path(table_path)
        table_name = table_path.stem
        frame = pd.read_csv(table_path)
        report = run_qc(frame, table_name=table_name)
        print(summarize(report, table_name=table_name, row_count=len(frame)))

        flagged_only = report.loc[report["flagged"]]
        output_path = output_dir / f"{table_name}_qc.csv"
        flagged_only.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  report: {output_path.resolve()} ({len(flagged_only)} row(s))\n")
        any_flagged = any_flagged or not flagged_only.empty

    return 1 if any_flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
