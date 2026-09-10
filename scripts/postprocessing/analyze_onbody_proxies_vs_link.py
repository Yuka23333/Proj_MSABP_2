"""Relate per-frequency far-field proxies to measured on-body link metrics.

The population is deliberately small: twelve K-medoids plus the Roblin-Wei
reference. Proxy values remain at their eighteen individual frequencies; no
frequency averaging or channel weighting is applied to the predictors.

Spearman correlation is the primary descriptive statistic. Leave-one-out
variation exposes sensitivity to a single sample, and a max-T permutation test
controls the family-wise error caused by searching all proxy/frequency pairs.
The results remain exploratory because n=13 cannot validate a surrogate.
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
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing.analyze_onbody_proxy_frequencies import (  # noqa: E402
    PROXY_COLUMNS,
)
from scripts.postprocessing.ffs_onbody_proxies import _write_csv_atomic  # noqa: E402


DEFAULT_PROXY_TABLE = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "onbody_proxies_per_frequency_3p1-4p8GHz.csv"
)
DEFAULT_PROPAGATION_SELECTION = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_selected_13.csv"
)
DEFAULT_CLUSTER_TABLE = (
    REPOSITORY_ROOT / "results" / "processed" / "pareto_candidate_clusters_k12.csv"
)
DEFAULT_LINK_TABLE = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "propagation_s21_12_medoids_plus_roblin_wei_reference"
    / "metrics"
    / "september_s21_and_ebn0_lambda0p5.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "processed" / "onbody_proxies" / "link_correlation_13"
)
DEFAULT_FIGURE_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "figures"
    / "onbody_proxy_link_correlation_13_lambda0p5.png"
)
SELECTED_FILENAME = "selected_13_proxy_frequency_table.csv"
CORRELATION_FILENAME = "proxy_link_correlations_by_frequency.csv"
TOP_FILENAME = "top_proxy_link_correlations.csv"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_PERMUTATIONS = 100_000
DEFAULT_RANDOM_SEED = 20260908

TARGETS = {
    "s21_power_september": "higher_is_better",
    "ebn0_ber_v2_at_target_db": "lower_is_better",
}
REFERENCE_CASE_NAME = "roblin_wei_2012_reference"


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


def _save_figure_atomic(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        figure.savefig(temporary, dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return path


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")


def select_link_population(
    proxy_table: pd.DataFrame,
    propagation_selection: pd.DataFrame,
    cluster_table: pd.DataFrame,
    link_table: pd.DataFrame,
) -> pd.DataFrame:
    """Select the twelve medoids and reference, then attach link targets."""

    _require_columns(
        proxy_table,
        {"source", "case_id", "sample_index", "freq_ghz", *PROXY_COLUMNS},
        "proxy table",
    )
    _require_columns(
        propagation_selection,
        {"sample_id", "candidate_rank", "source", "source_case_id"},
        "propagation selection",
    )
    _require_columns(
        cluster_table,
        {"candidate_rank", "is_selected_medoid"},
        "cluster table",
    )
    _require_columns(
        link_table,
        {"case_name", "candidate_rank", *TARGETS},
        "link table",
    )

    selected_flag = cluster_table["is_selected_medoid"]
    if selected_flag.dtype != bool:
        selected_flag = selected_flag.astype(str).str.casefold().eq("true")
    medoid_ranks = set(
        cluster_table.loc[selected_flag, "candidate_rank"].astype(int).tolist()
    )
    if len(medoid_ranks) != 12:
        raise ValueError(f"expected 12 selected medoids, got {len(medoid_ranks)}")
    medoids = propagation_selection.loc[
        propagation_selection["candidate_rank"].astype(int).isin(medoid_ranks)
    ].copy()
    if len(medoids) != 12:
        raise ValueError("propagation selection does not map all 12 medoids")

    populations: list[pd.DataFrame] = []
    for row in medoids.itertuples(index=False):
        case_name = f"case_{row.sample_id}"
        match = proxy_table.loc[
            proxy_table["source"].eq(row.source)
            & proxy_table["case_id"].astype(str).eq(str(row.source_case_id))
        ].copy()
        if match["sample_index"].nunique() != 1:
            raise ValueError(
                f"proxy identity is not unique for candidate rank {row.candidate_rank}"
            )
        match.insert(0, "link_case_name", case_name)
        populations.append(match)

    reference = proxy_table.loc[
        proxy_table["source"].eq(REFERENCE_CASE_NAME)
        & proxy_table["case_id"].eq(REFERENCE_CASE_NAME)
    ].copy()
    if reference["sample_index"].nunique() != 1:
        raise ValueError("Roblin-Wei reference proxy identity is not unique")
    reference.insert(0, "link_case_name", REFERENCE_CASE_NAME)
    populations.append(reference)

    selected = pd.concat(populations, ignore_index=True)
    expected_frequencies = np.sort(selected["freq_ghz"].unique())
    if len(expected_frequencies) != 18 or len(selected) != 13 * 18:
        raise ValueError(
            "selected proxy population must contain 13 samples x 18 frequencies"
        )
    frequency_counts = selected.groupby("link_case_name")["freq_ghz"].nunique()
    if not frequency_counts.eq(18).all():
        raise ValueError("one or more selected samples lacks the 18-point grid")

    if set(link_table["case_name"]) != set(selected["link_case_name"]):
        raise ValueError("link result cases do not match selected proxy cases")
    link_columns = [
        column
        for column in link_table.columns
        if column not in {"candidate_rank", "source", "source_case_id"}
    ]
    selected = selected.merge(
        link_table[link_columns],
        left_on="link_case_name",
        right_on="case_name",
        how="left",
        validate="many_to_one",
    )
    if selected[list(TARGETS)].isna().any().any():
        raise ValueError("selected population has missing link targets")
    return selected.sort_values(
        ["freq_ghz", "link_case_name"],
        kind="stable",
    ).reset_index(drop=True)


def _standardized_ranks(values: np.ndarray) -> np.ndarray:
    ranked = stats.rankdata(np.asarray(values, dtype=np.float64), method="average")
    centered = ranked - ranked.mean()
    norm = np.linalg.norm(centered)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot correlate a constant ranked vector")
    return centered / norm


def _max_t_permutation_pvalues(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    *,
    permutations: int,
    seed: int,
    batch_size: int = 2_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return observed Spearman rho, max-T FWER p, and null 95th percentile."""

    if permutations < 1:
        raise ValueError("permutations must be positive")
    x = np.asarray(feature_matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("feature matrix and target shapes disagree")
    ranked_features = np.column_stack(
        [_standardized_ranks(x[:, index]) for index in range(x.shape[1])]
    )
    ranked_target = _standardized_ranks(y)
    observed = ranked_target @ ranked_features

    rng = np.random.default_rng(seed)
    max_null = np.empty(permutations, dtype=np.float64)
    cursor = 0
    while cursor < permutations:
        count = min(batch_size, permutations - cursor)
        permuted_targets = np.stack(
            [ranked_target[rng.permutation(len(ranked_target))] for _ in range(count)]
        )
        null_correlations = np.abs(permuted_targets @ ranked_features)
        max_null[cursor : cursor + count] = np.max(null_correlations, axis=1)
        cursor += count
    adjusted = (1.0 + np.sum(max_null[:, None] >= np.abs(observed), axis=0)) / (
        permutations + 1.0
    )
    return observed, adjusted, float(np.quantile(max_null, 0.95))


def analyze_correlations(
    selected: pd.DataFrame,
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Analyze all proxy/frequency predictors against each link target."""

    case_order = sorted(selected["link_case_name"].unique())
    try:
        reference_index = case_order.index(REFERENCE_CASE_NAME)
    except ValueError as exc:
        raise ValueError("selected population lacks the Roblin-Wei reference") from exc
    features: list[np.ndarray] = []
    feature_keys: list[tuple[str, float]] = []
    for metric in PROXY_COLUMNS:
        pivot = selected.pivot(
            index="link_case_name",
            columns="freq_ghz",
            values=metric,
        ).reindex(case_order)
        for frequency_ghz in sorted(pivot.columns):
            values = pivot[frequency_ghz].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite predictor: {metric} at {frequency_ghz}")
            features.append(values)
            feature_keys.append((metric, float(frequency_ghz)))
    feature_matrix = np.column_stack(features)

    link_by_case = selected.drop_duplicates("link_case_name").set_index(
        "link_case_name"
    )
    rows: list[dict[str, Any]] = []
    null_thresholds: dict[str, float] = {}
    for target_index, (target_name, direction) in enumerate(TARGETS.items()):
        target = link_by_case.reindex(case_order)[target_name].to_numpy(
            dtype=np.float64
        )
        observed, adjusted_p, null_threshold = _max_t_permutation_pvalues(
            feature_matrix,
            target,
            permutations=permutations,
            seed=seed + target_index,
        )
        null_thresholds[target_name] = null_threshold
        for feature_index, ((metric, frequency_ghz), predictor) in enumerate(
            zip(feature_keys, features, strict=True)
        ):
            spearman = stats.spearmanr(predictor, target)
            spearman_without_reference = stats.spearmanr(
                np.delete(predictor, reference_index),
                np.delete(target, reference_index),
            )
            kendall = stats.kendalltau(predictor, target)
            pearson = stats.pearsonr(predictor, target)
            loo_rho = np.asarray(
                [
                    stats.spearmanr(
                        np.delete(predictor, omitted),
                        np.delete(target, omitted),
                    ).statistic
                    for omitted in range(len(target))
                ],
                dtype=np.float64,
            )
            rho = float(observed[feature_index])
            rho_sign = np.sign(rho)
            rows.append(
                {
                    "target": target_name,
                    "target_direction": direction,
                    "proxy": metric,
                    "freq_ghz": frequency_ghz,
                    "n": len(target),
                    "spearman_rho": rho,
                    "spearman_without_reference": float(
                        spearman_without_reference.statistic
                    ),
                    "spearman_p_asymptotic": float(spearman.pvalue),
                    "spearman_max_t_fwer_p": float(adjusted_p[feature_index]),
                    "kendall_tau": float(kendall.statistic),
                    "kendall_p_asymptotic": float(kendall.pvalue),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "loo_spearman_min": float(np.min(loo_rho)),
                    "loo_spearman_median": float(np.median(loo_rho)),
                    "loo_spearman_max": float(np.max(loo_rho)),
                    "loo_same_sign_fraction": float(
                        np.mean(np.sign(loo_rho) == rho_sign)
                    ),
                }
            )
    return pd.DataFrame(rows), null_thresholds


def _top_correlations(correlations: pd.DataFrame, count: int = 20) -> pd.DataFrame:
    ranked = correlations.assign(
        absolute_spearman=correlations["spearman_rho"].abs()
    ).sort_values(
        ["target", "absolute_spearman", "spearman_max_t_fwer_p"],
        ascending=[True, False, True],
        kind="stable",
    )
    return ranked.groupby("target", sort=False).head(count).reset_index(drop=True)


def _plot_heatmaps(correlations: pd.DataFrame) -> plt.Figure:
    frequencies = sorted(correlations["freq_ghz"].unique())
    figure, axes = plt.subplots(2, 1, figsize=(13.5, 11.0), constrained_layout=True)
    for axis, (target_name, direction) in zip(axes, TARGETS.items(), strict=True):
        subset = correlations.loc[correlations["target"].eq(target_name)]
        heatmap = subset.pivot(index="proxy", columns="freq_ghz", values="spearman_rho")
        heatmap = heatmap.reindex(index=PROXY_COLUMNS, columns=frequencies)
        image = axis.imshow(
            heatmap.to_numpy(),
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.set_title(f"Spearman rho vs {target_name} ({direction}), n=13")
        axis.set_yticks(range(len(PROXY_COLUMNS)), labels=PROXY_COLUMNS)
        axis.set_xticks(
            range(len(frequencies)),
            labels=[f"{value:g}" for value in frequencies],
            rotation=45,
            ha="right",
        )
        axis.set_xlabel("Frequency (GHz); predictors are not frequency-averaged")
        adjusted = subset.pivot(
            index="proxy",
            columns="freq_ghz",
            values="spearman_max_t_fwer_p",
        ).reindex(index=PROXY_COLUMNS, columns=frequencies)
        for row, column in np.argwhere(adjusted.to_numpy() < 0.05):
            axis.text(column, row, "*", ha="center", va="center", fontsize=8)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.85)
    colorbar.set_label("Spearman rho; * max-T FWER p < 0.05")
    return figure


def run_analysis(
    *,
    proxy_table_path: str | Path = DEFAULT_PROXY_TABLE,
    propagation_selection_path: str | Path = DEFAULT_PROPAGATION_SELECTION,
    cluster_table_path: str | Path = DEFAULT_CLUSTER_TABLE,
    link_table_path: str | Path = DEFAULT_LINK_TABLE,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    figure_path: str | Path = DEFAULT_FIGURE_PATH,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    inputs = {
        "proxy_table": Path(proxy_table_path).expanduser().resolve(),
        "propagation_selection": Path(propagation_selection_path)
        .expanduser()
        .resolve(),
        "cluster_table": Path(cluster_table_path).expanduser().resolve(),
        "link_table": Path(link_table_path).expanduser().resolve(),
    }
    output = Path(output_directory).expanduser().resolve()
    figure = Path(figure_path).expanduser().resolve()
    outputs = {
        "selected_population": output / SELECTED_FILENAME,
        "correlations": output / CORRELATION_FILENAME,
        "top_correlations": output / TOP_FILENAME,
        "manifest": output / MANIFEST_FILENAME,
        "figure": figure,
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"analysis output exists; pass --overwrite to replace: {existing[0]}"
        )

    proxy_table = pd.read_csv(inputs["proxy_table"])
    propagation_selection = pd.read_csv(inputs["propagation_selection"])
    cluster_table = pd.read_csv(inputs["cluster_table"])
    link_table = pd.read_csv(inputs["link_table"])
    selected = select_link_population(
        proxy_table,
        propagation_selection,
        cluster_table,
        link_table,
    )
    correlations, null_thresholds = analyze_correlations(
        selected,
        permutations=permutations,
        seed=seed,
    )
    top = _top_correlations(correlations)

    _write_csv_atomic(selected, outputs["selected_population"], overwrite=overwrite)
    _write_csv_atomic(correlations, outputs["correlations"], overwrite=overwrite)
    _write_csv_atomic(top, outputs["top_correlations"], overwrite=overwrite)
    _save_figure_atomic(_plot_heatmaps(correlations), outputs["figure"])

    payload: dict[str, Any] = {
        "schema": "msabp.onbody_proxy_link_correlation.v1",
        "status": "exploratory; n=13",
        "sample_count": 13,
        "population": "12 K-medoids plus Roblin-Wei reference; rank 1 excluded",
        "predictor_frequency_aggregation": None,
        "frequency_count": int(selected["freq_ghz"].nunique()),
        "proxy_count": len(PROXY_COLUMNS),
        "hypothesis_count_per_target": len(PROXY_COLUMNS)
        * int(selected["freq_ghz"].nunique()),
        "targets": TARGETS,
        "primary_statistic": "Spearman rho",
        "stability": "leave-one-out Spearman over 13 omissions",
        "multiplicity_control": "Monte Carlo max-T FWER over all proxy/frequency pairs",
        "permutations": int(permutations),
        "random_seed": int(seed),
        "max_t_null_95pct_absolute_rho": null_thresholds,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    _write_json_atomic(payload, outputs["manifest"])
    print(
        "[ProxyLink] n=13, predictors="
        f"{payload['hypothesis_count_per_target']}/target, "
        f"permutations={permutations}"
    )
    for target_name, threshold in null_thresholds.items():
        significant = int(
            (
                correlations.loc[
                    correlations["target"].eq(target_name),
                    "spearman_max_t_fwer_p",
                ]
                < 0.05
            ).sum()
        )
        print(
            f"[ProxyLink] {target_name}: null max-|rho| p95={threshold:.6f}, "
            f"FWER-significant={significant}"
        )
    for name, path in outputs.items():
        print(f"[ProxyLink] {name}: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-table", type=Path, default=DEFAULT_PROXY_TABLE)
    parser.add_argument(
        "--propagation-selection",
        type=Path,
        default=DEFAULT_PROPAGATION_SELECTION,
    )
    parser.add_argument("--cluster-table", type=Path, default=DEFAULT_CLUSTER_TABLE)
    parser.add_argument("--link-table", type=Path, default=DEFAULT_LINK_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(
        proxy_table_path=args.proxy_table,
        propagation_selection_path=args.propagation_selection,
        cluster_table_path=args.cluster_table,
        link_table_path=args.link_table,
        output_directory=args.output_dir,
        figure_path=args.figure,
        permutations=args.permutations,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
