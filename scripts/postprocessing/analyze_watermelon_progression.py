"""Analyze latitude-belt radiation redistribution across six K-RVEA rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "processed" / "watermelon_latitude_10deg"
)
DEFAULT_OUTPUT_DIRECTORY = DEFAULT_INPUT_DIRECTORY / "progression"
DEFAULT_FIGURE = (
    REPOSITORY_ROOT / "results" / "figures" / "watermelon_latitude_progression.png"
)
DEFAULT_BAND_GHZ = (3.1, 4.8)
DOE_SOURCE = "doe-11var-branch-up-lhs-512-001"
REFERENCE_SOURCE = "roblin_wei_2012_reference"
ROUND_SOURCES: tuple[tuple[str, str], ...] = (
    ("R1 smoke", "msabp-krvea-11var-smoke-128-001"),
    ("R2 calibrated", "msabp-krvea-11var-calibrated-64-002"),
    ("R3 calibrated", "msabp-krvea-11var-calibrated-64-003"),
    ("R4 deep", "msabp-krvea-11var-deep-64-004"),
    ("R5 deep", "msabp-krvea-11var-deep-64-005"),
    ("R6 learned", "msabp-krvea-11var-stage2-learned-64-006"),
)
ROUND_LABEL_BY_SOURCE = {source: label for label, source in ROUND_SOURCES}

SAMPLE_METRICS_FILENAME = "sample_region_metrics.csv"
ROUND_SUMMARY_FILENAME = "round_summary.csv"
PROFILE_FILENAME = "latitude_profiles.csv"
FRONT_SUMMARY_FILENAME = "cumulative_front_summary.csv"
FRONT_MEMBERS_FILENAME = "cumulative_front_members.csv"
MANIFEST_FILENAME = "analysis_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(payload: Mapping[str, Any], path: Path) -> Path:
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


def _region_mask(
    theta_start: np.ndarray,
    theta_stop: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    return (theta_start >= lower - 1e-10) & (theta_stop <= upper + 1e-10)


def _solid_angle_mean(
    values: np.ndarray,
    omega: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    return (values[:, mask] * omega[mask]).sum(axis=1) / omega[mask].sum()


def _pattern_share(
    directivity: np.ndarray,
    omega: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    return (directivity[:, mask] * omega[mask]).sum(axis=1) / (4.0 * np.pi)


def _peak_category(latitude_center: float) -> str:
    if latitude_center > 70.0:
        return "north_polar_70_90"
    if latitude_center >= 20.0:
        return "north_mid_20_70"
    if latitude_center > -20.0:
        return "near_plane_abs_0_20"
    if latitude_center >= -70.0:
        return "south_mid_20_70"
    return "south_polar_70_90"


def build_sample_metrics(
    sample_index: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    *,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return per-sample regional metrics and broadband belt profiles."""

    frequency = np.asarray(arrays["frequency_ghz"], dtype=np.float64)
    theta_start = np.asarray(arrays["theta_start_deg"], dtype=np.float64)
    theta_stop = np.asarray(arrays["theta_stop_deg"], dtype=np.float64)
    omega = np.asarray(arrays["solid_angle_sr"], dtype=np.float64)
    low, high = map(float, band_ghz)
    band_mask = (frequency >= low - 1e-10) & (frequency <= high + 1e-10)
    if not np.any(band_mask):
        raise ValueError("analysis band contains no frequency samples")

    realized = np.asarray(arrays["realized_gain_linear"], dtype=np.float64)
    directivity = np.asarray(arrays["directivity_linear"], dtype=np.float64)
    if realized.shape[0] != len(sample_index):
        raise ValueError("sample index and watermelon arrays have different lengths")
    realized_band = realized[:, band_mask, :].mean(axis=1)
    directivity_band = directivity[:, band_mask, :].mean(axis=1)

    north_plane = _region_mask(theta_start, theta_stop, 70.0, 90.0)
    north_mid = _region_mask(theta_start, theta_stop, 20.0, 70.0)
    equatorial = _region_mask(theta_start, theta_stop, 80.0, 100.0)
    if not (north_plane.sum() == 2 and north_mid.sum() == 5 and equatorial.sum() == 2):
        raise ValueError("10-degree watermelon grid does not contain required regions")

    plane_gain = _solid_angle_mean(realized_band, omega, north_plane)
    mid_gain = _solid_angle_mean(realized_band, omega, north_mid)
    equatorial_gain = _solid_angle_mean(realized_band, omega, equatorial)
    plane_share = _pattern_share(directivity_band, omega, north_plane)
    mid_share = _pattern_share(directivity_band, omega, north_mid)
    equatorial_share = _pattern_share(directivity_band, omega, equatorial)
    latitude_center = 90.0 - (theta_start + theta_stop) / 2.0
    peak_index = np.argmax(realized_band, axis=1)
    peak_latitude = latitude_center[peak_index]

    metrics = sample_index.copy()
    metrics["round_label"] = metrics["source"].map(ROUND_LABEL_BY_SOURCE)
    metrics.loc[metrics["source"].eq(REFERENCE_SOURCE), "round_label"] = (
        "Roblin-Wei reference"
    )
    metrics["north_plane_0_20_gain_linear"] = plane_gain
    metrics["north_plane_0_20_gain_dbi"] = 10.0 * np.log10(plane_gain)
    metrics["north_mid_20_70_gain_linear"] = mid_gain
    metrics["north_mid_20_70_gain_dbi"] = 10.0 * np.log10(mid_gain)
    metrics["equatorial_abs_0_10_gain_linear"] = equatorial_gain
    metrics["equatorial_abs_0_10_gain_dbi"] = 10.0 * np.log10(equatorial_gain)
    metrics["plane_minus_north_mid_dB"] = 10.0 * np.log10(plane_gain / mid_gain)
    metrics["equator_minus_north_mid_dB"] = 10.0 * np.log10(
        equatorial_gain / mid_gain
    )
    metrics["north_plane_0_20_pattern_share"] = plane_share
    metrics["north_mid_20_70_pattern_share"] = mid_share
    metrics["equatorial_abs_0_10_pattern_share"] = equatorial_share
    metrics["peak_latitude_center_deg"] = peak_latitude
    metrics["peak_region"] = [_peak_category(value) for value in peak_latitude]
    return metrics, realized_band, directivity_band


def _quantiles(values: pd.Series, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        f"{prefix}_q25": float(numeric.quantile(0.25)),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_q75": float(numeric.quantile(0.75)),
    }


def summarize_group(frame: pd.DataFrame, label: str, order: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "group_order": order,
        "group_label": label,
        "sample_count": int(len(frame)),
    }
    for column in (
        "north_plane_0_20_gain_dbi",
        "north_mid_20_70_gain_dbi",
        "equatorial_abs_0_10_gain_dbi",
        "plane_minus_north_mid_dB",
        "equator_minus_north_mid_dB",
        "north_plane_0_20_pattern_share",
        "north_mid_20_70_pattern_share",
        "equatorial_abs_0_10_pattern_share",
        "peak_latitude_center_deg",
    ):
        result.update(_quantiles(frame[column], column))
    result["plane_stronger_than_north_mid_fraction"] = float(
        (frame["plane_minus_north_mid_dB"] > 0.0).mean()
    )
    for category in (
        "north_polar_70_90",
        "north_mid_20_70",
        "near_plane_abs_0_20",
        "south_mid_20_70",
        "south_polar_70_90",
    ):
        result[f"peak_fraction_{category}"] = float(
            frame["peak_region"].eq(category).mean()
        )
    return result


def build_round_summary(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference = sample_metrics.loc[sample_metrics["source"].eq(REFERENCE_SOURCE)]
    if len(reference) != 1:
        raise ValueError("expected exactly one Roblin-Wei reference")
    rows.append(summarize_group(reference, "Roblin-Wei reference", 0))
    for order, (label, source) in enumerate(ROUND_SOURCES, start=1):
        group = sample_metrics.loc[sample_metrics["source"].eq(source)]
        if group.empty:
            raise ValueError(f"optimization round has no completed samples: {source}")
        rows.append(summarize_group(group, label, order))
    return pd.DataFrame.from_records(rows)


def build_profiles(
    sample_metrics: pd.DataFrame,
    realized_band: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    theta_start = np.asarray(arrays["theta_start_deg"], dtype=np.float64)
    theta_stop = np.asarray(arrays["theta_stop_deg"], dtype=np.float64)
    latitude_center = 90.0 - (theta_start + theta_stop) / 2.0
    realized_dbi = 10.0 * np.log10(realized_band)
    groups = [("Roblin-Wei reference", REFERENCE_SOURCE), *ROUND_SOURCES]
    rows: list[dict[str, Any]] = []
    for group_order, (label, source) in enumerate(groups):
        positions = np.flatnonzero(sample_metrics["source"].eq(source).to_numpy())
        values = realized_dbi[positions]
        for belt_index in range(len(theta_start)):
            rows.append(
                {
                    "group_order": group_order,
                    "group_label": label,
                    "sample_count": int(len(positions)),
                    "theta_start_deg": float(theta_start[belt_index]),
                    "theta_stop_deg": float(theta_stop[belt_index]),
                    "latitude_center_deg": float(latitude_center[belt_index]),
                    "realized_gain_dbi_q25": float(
                        np.quantile(values[:, belt_index], 0.25)
                    ),
                    "realized_gain_dbi_median": float(
                        np.median(values[:, belt_index])
                    ),
                    "realized_gain_dbi_q75": float(
                        np.quantile(values[:, belt_index], 0.75)
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    values = np.asarray(objectives, dtype=np.float64)
    selected = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        weakly_better = np.all(values <= point, axis=1)
        strictly_better = np.any(values < point, axis=1)
        selected[index] = not np.any(weakly_better & strictly_better)
    return selected


def build_cumulative_fronts(
    sample_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive = sample_metrics.loc[sample_metrics["sample_kind"].eq("archive")].copy()
    required = {
        "worst_s11_linear_amplitude",
        "mean_total_efficiency_linear",
        "normalized_substrate_area",
        "cap_realized_gain_linear",
    }
    missing = required - set(archive.columns)
    if missing:
        raise ValueError(f"sample index is missing objectives: {sorted(missing)}")

    summaries: list[dict[str, Any]] = []
    members: list[pd.DataFrame] = []
    cumulative_sources = {DOE_SOURCE}
    for order, (label, source) in enumerate(ROUND_SOURCES, start=1):
        cumulative_sources.add(source)
        cumulative = archive.loc[archive["source"].isin(cumulative_sources)].copy()
        objectives = np.column_stack(
            [
                cumulative["worst_s11_linear_amplitude"].to_numpy(float),
                1.0 - cumulative["mean_total_efficiency_linear"].to_numpy(float),
                cumulative["normalized_substrate_area"].to_numpy(float),
                cumulative["cap_realized_gain_linear"].to_numpy(float),
            ]
        )
        front = cumulative.loc[_nondominated_mask(objectives)].copy()
        summary = summarize_group(front, f"Cumulative front after {label}", order)
        summary["archive_sample_count"] = int(len(cumulative))
        summary["front_size"] = int(len(front))
        summaries.append(summary)
        front.insert(0, "front_stage_order", order)
        front.insert(1, "front_stage_label", label)
        members.append(front)
    return pd.DataFrame.from_records(summaries), pd.concat(members, ignore_index=True)


def plot_latitude_profiles(profiles: pd.DataFrame, path: Path) -> Path:
    """Plot only the latitude-profile panel retained for the current analysis."""

    figure, axis = plt.subplots(figsize=(11.0, 6.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ROUND_SOURCES)))

    reference_profile = profiles.loc[
        profiles["group_label"].eq("Roblin-Wei reference")
    ].sort_values("latitude_center_deg")
    axis.plot(
        reference_profile["latitude_center_deg"],
        reference_profile["realized_gain_dbi_median"],
        color="black",
        linewidth=2.2,
        linestyle="--",
        label="Roblin-Wei reference",
    )
    for color, (label, _source) in zip(colors, ROUND_SOURCES, strict=True):
        group = profiles.loc[profiles["group_label"].eq(label)].sort_values(
            "latitude_center_deg"
        )
        axis.plot(
            group["latitude_center_deg"],
            group["realized_gain_dbi_median"],
            color=color,
            linewidth=1.8,
            label=label,
        )
    axis.axvspan(0.0, 20.0, color="#38bdf8", alpha=0.12, label="Near-plane")
    axis.axvspan(
        20.0,
        70.0,
        color="#fb923c",
        alpha=0.10,
        label="North midlatitude",
    )
    axis.set(
        title="Watermelon latitude profile: 3.1-4.8 GHz, 10-degree belts",
        xlabel="Latitude center (deg)",
        ylabel="Belt-average realized gain (dBi)",
        xlim=(-90.0, 90.0),
    )
    axis.set_xticks(np.arange(-85.0, 86.0, 10.0))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9, ncol=3, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def analyze(
    input_directory: str | Path = DEFAULT_INPUT_DIRECTORY,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    figure_path: str | Path = DEFAULT_FIGURE,
    *,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
) -> dict[str, Any]:
    input_root = Path(input_directory).expanduser().resolve()
    output_root = Path(output_directory).expanduser().resolve()
    array_path = input_root / "watermelon_latitude_gain.npz"
    index_path = input_root / "sample_index.csv"
    source_manifest_path = input_root / "manifest.json"
    arrays = dict(np.load(array_path))
    sample_index = pd.read_csv(index_path)

    sample_metrics, realized_band, _directivity_band = build_sample_metrics(
        sample_index,
        arrays,
        band_ghz=band_ghz,
    )
    relevant_sources = {REFERENCE_SOURCE, *ROUND_LABEL_BY_SOURCE}
    relevant_metrics = sample_metrics.loc[
        sample_metrics["source"].isin(relevant_sources)
    ].copy()
    round_summary = build_round_summary(relevant_metrics)
    profiles = build_profiles(sample_metrics, realized_band, arrays)
    front_summary, front_members = build_cumulative_fronts(sample_metrics)

    outputs = {
        SAMPLE_METRICS_FILENAME: output_root / SAMPLE_METRICS_FILENAME,
        ROUND_SUMMARY_FILENAME: output_root / ROUND_SUMMARY_FILENAME,
        PROFILE_FILENAME: output_root / PROFILE_FILENAME,
        FRONT_SUMMARY_FILENAME: output_root / FRONT_SUMMARY_FILENAME,
        FRONT_MEMBERS_FILENAME: output_root / FRONT_MEMBERS_FILENAME,
    }
    _atomic_csv(relevant_metrics, outputs[SAMPLE_METRICS_FILENAME])
    _atomic_csv(round_summary, outputs[ROUND_SUMMARY_FILENAME])
    _atomic_csv(profiles, outputs[PROFILE_FILENAME])
    _atomic_csv(front_summary, outputs[FRONT_SUMMARY_FILENAME])
    _atomic_csv(front_members, outputs[FRONT_MEMBERS_FILENAME])
    figure = plot_latitude_profiles(
        profiles,
        Path(figure_path).expanduser().resolve(),
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "input_array": str(array_path),
        "input_array_sha256": _sha256(array_path),
        "input_sample_index": str(index_path),
        "input_sample_index_sha256": _sha256(index_path),
        "input_manifest": str(source_manifest_path),
        "input_manifest_sha256": _sha256(source_manifest_path),
        "band_ghz": [float(band_ghz[0]), float(band_ghz[1])],
        "included_sources": [REFERENCE_SOURCE, *ROUND_LABEL_BY_SOURCE],
        "excluded_from_round_distributions": [
            DOE_SOURCE,
            "current_default_reference",
        ],
        "cumulative_front_initial_source": DOE_SOURCE,
        "regions": {
            "north_plane": {
                "latitude_deg": [0.0, 20.0],
                "theta_deg": [70.0, 90.0],
            },
            "north_mid": {
                "latitude_deg": [20.0, 70.0],
                "theta_deg": [20.0, 70.0],
            },
            "equatorial": {
                "latitude_deg": [-10.0, 10.0],
                "theta_deg": [80.0, 100.0],
            },
        },
        "gain_density": "solid-angle-weighted region mean of broadband realized gain",
        "pattern_share": "solid-angle integral of broadband directivity divided by 4*pi",
        "figure_kind": "single_panel_latitude_profiles",
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in outputs.items()
        }
        | {
            "figure": {
                "path": str(figure),
                "sha256": _sha256(figure),
            }
        },
    }
    manifest_path = output_root / MANIFEST_FILENAME
    _atomic_json(manifest, manifest_path)

    display_columns = [
        "group_label",
        "sample_count",
        "plane_minus_north_mid_dB_median",
        "plane_minus_north_mid_dB_q25",
        "plane_minus_north_mid_dB_q75",
        "plane_stronger_than_north_mid_fraction",
        "peak_fraction_north_mid_20_70",
        "peak_fraction_near_plane_abs_0_20",
        "north_plane_0_20_pattern_share_median",
        "north_mid_20_70_pattern_share_median",
    ]
    print(round_summary.loc[:, display_columns].to_string(index=False))
    print("\nCumulative fronts:")
    print(
        front_summary.loc[
            :,
            [
                "group_label",
                "archive_sample_count",
                "front_size",
                "plane_minus_north_mid_dB_median",
                "peak_fraction_north_mid_20_70",
                "peak_fraction_near_plane_abs_0_20",
            ],
        ].to_string(index=False)
    )
    print(f"\nAnalysis directory: {output_root}")
    print(f"Figure: {figure}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--band", nargs=2, type=float, default=DEFAULT_BAND_GHZ)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyze(
        args.input_dir,
        args.output_dir,
        args.figure,
        band_ghz=(float(args.band[0]), float(args.band[1])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
