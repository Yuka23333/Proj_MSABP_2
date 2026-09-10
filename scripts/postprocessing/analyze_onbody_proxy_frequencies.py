"""Analyze on-body far-field proxies independently at each frequency.

This stage deliberately performs no broadband averaging or channel weighting.
It excludes reference rows from population statistics because their historical
optimization objectives are intentionally blank, while retaining them in the
prepared source table for later visual comparison.
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing.ffs_onbody_proxies import _write_csv_atomic  # noqa: E402


DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "onbody_proxies_per_frequency_3p1-4p8GHz.csv"
)
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "processed" / "onbody_proxies"
SUMMARY_FILENAME = "proxy_summary_by_frequency.csv"
OBJECTIVE_CORRELATION_FILENAME = "proxy_objective_spearman_by_frequency.csv"
PAIRWISE_CORRELATION_FILENAME = "proxy_pairwise_spearman_by_frequency.csv"
MANIFEST_FILENAME = "frequency_analysis_manifest.json"

# Analyze one representation for each quantity. dB columns are monotonic
# transforms of their linear partners and would duplicate Spearman results.
PROXY_COLUMNS = (
    "P_hor",
    "P_hor80",
    "P_cap",
    "theta_centroid_deg",
    "G_theta_hor",
    "G_theta_hor_min",
    "chi_TM",
    "G_endfire_phi90_vertical",
    "G_endfire_phi90_horizontal",
    "G_endfire_phi90_total",
    "chi_vertical_endfire_phi90",
    "G_cap_max",
    "Rad_Eff_from_ffs",
    "Tot_Eff_from_ffs",
)
HISTORICAL_OBJECTIVE_COLUMNS = (
    "worst_s11_linear_amplitude",
    "mean_total_efficiency_linear",
    "normalized_substrate_area",
    "cap_realized_gain_linear",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_columns(frame: pd.DataFrame) -> None:
    required = {
        "sample_index",
        "sample_kind",
        "freq_ghz",
        *PROXY_COLUMNS,
        *HISTORICAL_OBJECTIVE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"per-frequency proxy table lacks columns: {missing}")


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def analyze_frequency_table(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-frequency summaries and two Spearman long tables."""

    _validate_columns(frame)
    archive = frame.loc[frame["sample_kind"] == "archive"].copy()
    if archive.empty:
        raise ValueError("per-frequency proxy table contains no archive samples")
    if archive.duplicated(["sample_index", "freq_ghz"]).any():
        raise ValueError("archive contains duplicate sample/frequency rows")

    summary_rows: list[dict[str, float | int | str]] = []
    objective_rows: list[dict[str, float | int | str]] = []
    pairwise_rows: list[dict[str, float | int | str]] = []

    for frequency_ghz, group in archive.groupby("freq_ghz", sort=True):
        proxy_data = _finite_numeric(group, PROXY_COLUMNS)
        for metric in PROXY_COLUMNS:
            values = proxy_data[metric].dropna()
            summary_rows.append(
                {
                    "freq_ghz": float(frequency_ghz),
                    "metric": metric,
                    "n_finite": int(len(values)),
                    "n_missing": int(len(group) - len(values)),
                    "minimum": float(values.min()),
                    "p10": float(values.quantile(0.10)),
                    "median": float(values.median()),
                    "p90": float(values.quantile(0.90)),
                    "maximum": float(values.max()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
            )

        analysis_columns = (*PROXY_COLUMNS, *HISTORICAL_OBJECTIVE_COLUMNS)
        numeric = _finite_numeric(group, analysis_columns)
        correlation = numeric.corr(method="spearman", min_periods=3)
        complete_counts = numeric.notna().astype(np.int64).T @ numeric.notna().astype(
            np.int64
        )

        for metric in PROXY_COLUMNS:
            for objective in HISTORICAL_OBJECTIVE_COLUMNS:
                objective_rows.append(
                    {
                        "freq_ghz": float(frequency_ghz),
                        "proxy": metric,
                        "objective": objective,
                        "spearman_rho": float(correlation.loc[metric, objective]),
                        "n_complete": int(complete_counts.loc[metric, objective]),
                    }
                )

        for left_index, left in enumerate(PROXY_COLUMNS):
            for right in PROXY_COLUMNS[left_index + 1 :]:
                pairwise_rows.append(
                    {
                        "freq_ghz": float(frequency_ghz),
                        "proxy_a": left,
                        "proxy_b": right,
                        "spearman_rho": float(correlation.loc[left, right]),
                        "n_complete": int(complete_counts.loc[left, right]),
                    }
                )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(objective_rows),
        pd.DataFrame(pairwise_rows),
    )


def run_analysis(
    input_path: str | Path = DEFAULT_INPUT,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(input_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    paths = {
        "summary": output / SUMMARY_FILENAME,
        "objective_correlations": output / OBJECTIVE_CORRELATION_FILENAME,
        "pairwise_correlations": output / PAIRWISE_CORRELATION_FILENAME,
        "manifest": output / MANIFEST_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"analysis output already exists; pass --overwrite: {existing[0]}"
        )

    frame = pd.read_csv(source)
    summary, objective_correlations, pairwise_correlations = analyze_frequency_table(
        frame
    )
    _write_csv_atomic(summary, paths["summary"], overwrite=overwrite)
    _write_csv_atomic(
        objective_correlations,
        paths["objective_correlations"],
        overwrite=overwrite,
    )
    _write_csv_atomic(
        pairwise_correlations,
        paths["pairwise_correlations"],
        overwrite=overwrite,
    )

    frequency_grid = sorted(float(value) for value in frame["freq_ghz"].unique())
    payload: dict[str, Any] = {
        "schema": "msabp.onbody_proxies.frequency_analysis.v1",
        "frequency_aggregation": None,
        "population": "archive only; references excluded",
        "archive_sample_count": int(
            frame.loc[frame["sample_kind"] == "archive", "sample_index"].nunique()
        ),
        "frequency_grid_ghz": frequency_grid,
        "proxy_columns": list(PROXY_COLUMNS),
        "historical_objective_columns": list(HISTORICAL_OBJECTIVE_COLUMNS),
        "correlation": "Spearman, independently at each frequency",
        "input": {"path": str(source), "sha256": _sha256(source)},
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    _write_json_atomic(payload, paths["manifest"])
    print(
        f"[OnBodyFrequencyAnalysis] frequencies={len(frequency_grid)}, "
        f"archive_samples={payload['archive_sample_count']}"
    )
    for name, path in paths.items():
        print(f"[OnBodyFrequencyAnalysis] {name}: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(args.input, args.output_dir, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
