"""Plot the sampled three-objective Pareto set from one or more CST runs.

The three objectives are:

* minimize the total substrate area;
* minimize the worst (maximum, least-negative) S11 value inside the band;
* maximize the arithmetic mean of the exported Tot_Eff dB samples in the band.

Reference/origin cases are plotted separately and deliberately excluded from
the sampled Pareto calculation.  Edit the ``F5_*`` constants for an IDE run,
or pass one or more ``--source`` arguments on the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# IDE / F5 configuration.
F5_SOURCE_DIRECTORIES = (
    REPOSITORY_ROOT / "results" / "raw" / "doe-round1-lhs-512",
    REPOSITORY_ROOT / "results" / "raw" / "msabp-qlogehvi-gpu-001",
)
F5_BAND_GHZ = (3.1, 4.8)
F5_OUTPUT_FIGURE = (
    REPOSITORY_ROOT
    / "results"
    / "figures"
    / "doe_round1_pareto_3d_3p1_4p8GHz.png"
)
F5_OUTPUT_CSV = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "doe_round1_pareto_metrics_3p1_4p8GHz.csv"
)
F5_SHOW_FIGURE = True

S11_FILENAME = "S11.csv"
TOTAL_EFFICIENCY_FILENAME = "Tot_Eff.csv"
MANIFEST_FILENAME = "manifest.json"
FLOAT_PAIR_RE = re.compile(
    r"^\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)


@dataclass(frozen=True)
class CaseMetrics:
    """One completed CST case reduced to the requested objective values."""

    source: str
    source_directory: str
    case_id: str
    case_directory: str
    substrate_area_mm2: float
    max_s11_db: float
    mean_tot_eff_db: float
    s11_band_points: int
    tot_eff_band_points: int
    is_reference: bool


class IncompleteCaseError(FileNotFoundError):
    """A manifest exists, but one or more required solver curves do not."""


def read_cst_1d_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the two numeric columns from a CST ASCII 1D export."""

    frequencies: list[float] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            match = FLOAT_PAIR_RE.match(line)
            if match is None:
                continue
            frequency = float(match.group(1))
            value = float(match.group(2))
            if math.isfinite(frequency) and math.isfinite(value):
                frequencies.append(frequency)
                values.append(value)
    if not frequencies:
        raise ValueError(f"No numeric curve samples found in {path}")

    frequency_array = np.asarray(frequencies, dtype=float)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(frequency_array, kind="stable")
    return frequency_array[order], value_array[order]


def values_in_band(
    frequencies_ghz: np.ndarray,
    values: np.ndarray,
    band_ghz: tuple[float, float],
) -> np.ndarray:
    """Return samples inside an inclusive frequency band."""

    lower, upper = band_ghz
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError(f"Invalid band {band_ghz!r}; expected lower < upper")
    mask = (frequencies_ghz >= lower) & (frequencies_ghz <= upper)
    selected = np.asarray(values, dtype=float)[mask]
    if selected.size == 0:
        raise ValueError(
            f"Curve has no samples inside the inclusive band [{lower}, {upper}] GHz"
        )
    return selected


def _case_is_reference(manifest: dict[str, object], case_directory: Path) -> bool:
    case_id = str(manifest.get("case_id", ""))
    if case_id.casefold() == "origin":
        return True
    if case_directory.name.casefold() in {"case_origin", "origin"}:
        return True
    return str(manifest.get("doe_source", "")).casefold() in {
        "origin",
        "reference",
    }


def _substrate_area_mm2(manifest: dict[str, object], manifest_path: Path) -> float:
    geometry = manifest.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"Missing geometry object in {manifest_path}")
    area = float(geometry["substrate_area_mm2"])
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError(f"Invalid substrate_area_mm2={area!r} in {manifest_path}")
    return area


def case_metrics_from_manifest(
    manifest_path: Path,
    *,
    source_directory: Path,
    band_ghz: tuple[float, float],
) -> CaseMetrics:
    """Load one case manifest and reduce its S11/Tot_Eff curves."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    case_directory = manifest_path.parent
    required_curves = (
        case_directory / S11_FILENAME,
        case_directory / TOTAL_EFFICIENCY_FILENAME,
    )
    missing_curves = [path.name for path in required_curves if not path.is_file()]
    if missing_curves:
        raise IncompleteCaseError(
            "Manifest-only/incomplete case is missing required curve(s): "
            + ", ".join(missing_curves)
        )
    s11_frequency, s11_db = read_cst_1d_curve(required_curves[0])
    eff_frequency, total_efficiency_db = read_cst_1d_curve(
        required_curves[1]
    )
    s11_band = values_in_band(s11_frequency, s11_db, band_ghz)
    efficiency_band = values_in_band(
        eff_frequency,
        total_efficiency_db,
        band_ghz,
    )
    source_directory = source_directory.resolve()
    return CaseMetrics(
        source=source_directory.name,
        source_directory=str(source_directory),
        case_id=str(manifest.get("case_id", case_directory.name)),
        case_directory=str(case_directory.resolve()),
        substrate_area_mm2=_substrate_area_mm2(manifest, manifest_path),
        max_s11_db=float(np.max(s11_band)),
        mean_tot_eff_db=float(np.mean(efficiency_band)),
        s11_band_points=int(s11_band.size),
        tot_eff_band_points=int(efficiency_band.size),
        is_reference=_case_is_reference(manifest, case_directory),
    )


def _manifest_paths(source_directory: Path) -> list[Path]:
    source_directory = source_directory.resolve()
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Result source directory does not exist: {source_directory}")
    direct_manifest = source_directory / MANIFEST_FILENAME
    if direct_manifest.is_file():
        return [direct_manifest]
    manifests = sorted(source_directory.rglob(MANIFEST_FILENAME))
    if not manifests:
        raise FileNotFoundError(
            f"No {MANIFEST_FILENAME} files found below {source_directory}"
        )
    return manifests


def collect_metrics(
    source_directories: Sequence[Path],
    *,
    band_ghz: tuple[float, float],
    skip_invalid_cases: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Collect cases from multiple roots and return one merged objective table."""

    if not source_directories:
        raise ValueError("At least one source directory is required")

    records: list[dict[str, object]] = []
    skipped: list[str] = []
    seen_manifests: set[Path] = set()
    for source_directory in source_directories:
        source_directory = Path(source_directory).resolve()
        for manifest_path in _manifest_paths(source_directory):
            resolved_manifest = manifest_path.resolve()
            if resolved_manifest in seen_manifests:
                continue
            seen_manifests.add(resolved_manifest)
            try:
                metrics = case_metrics_from_manifest(
                    resolved_manifest,
                    source_directory=source_directory,
                    band_ghz=band_ghz,
                )
            except IncompleteCaseError as exc:
                skipped.append(f"{resolved_manifest}: {type(exc).__name__}: {exc}")
                continue
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                message = f"{resolved_manifest}: {type(exc).__name__}: {exc}"
                if not skip_invalid_cases:
                    raise RuntimeError(message) from exc
                skipped.append(message)
                continue
            records.append(metrics.__dict__)

    if not records:
        raise RuntimeError("No complete cases could be loaded")
    frame = pd.DataFrame.from_records(records)
    frame.sort_values(
        ["source", "is_reference", "case_id"],
        kind="stable",
        inplace=True,
    )
    frame.reset_index(drop=True, inplace=True)
    return frame, skipped


def nondominated_mask(
    substrate_area_mm2: Iterable[float],
    max_s11_db: Iterable[float],
    mean_tot_eff_db: Iterable[float],
) -> np.ndarray:
    """Return a mask for min(area), min(max S11), max(mean Tot_Eff)."""

    objectives = np.column_stack(
        (
            np.asarray(list(substrate_area_mm2), dtype=float),
            np.asarray(list(max_s11_db), dtype=float),
            -np.asarray(list(mean_tot_eff_db), dtype=float),
        )
    )
    if objectives.ndim != 2 or objectives.shape[1] != 3:
        raise ValueError("Expected three equally sized one-dimensional objectives")
    if not np.isfinite(objectives).all():
        raise ValueError("Objective values must all be finite")

    result = np.ones(len(objectives), dtype=bool)
    for index, candidate in enumerate(objectives):
        dominated = np.any(
            np.all(objectives <= candidate, axis=1)
            & np.any(objectives < candidate, axis=1)
        )
        result[index] = not dominated
    return result


def label_sampled_pareto(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mark the Pareto set while excluding all reference rows."""

    labelled = metrics.copy()
    labelled["is_pareto"] = False
    sampled_indices = labelled.index[~labelled["is_reference"].astype(bool)]
    if len(sampled_indices) == 0:
        raise ValueError("No non-reference sampled cases are available")
    sampled = labelled.loc[sampled_indices]
    mask = nondominated_mask(
        sampled["substrate_area_mm2"],
        sampled["max_s11_db"],
        sampled["mean_tot_eff_db"],
    )
    labelled.loc[sampled_indices[mask], "is_pareto"] = True
    return labelled


def plot_pareto_3d(
    metrics: pd.DataFrame,
    *,
    band_ghz: tuple[float, float],
    output_figure: Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Create and optionally save/show the three-objective scatter plot."""

    figure = plt.figure(figsize=(12.0, 8.8))
    axis = figure.add_subplot(111, projection="3d")
    # mplot3d does not participate reliably in constrained_layout.  Reserve
    # explicit room for the long objective labels, especially the z label.
    figure.subplots_adjust(left=0.02, right=0.86, bottom=0.08, top=0.88)
    sources = list(dict.fromkeys(metrics["source"].astype(str)))
    color_map = plt.get_cmap("tab10")

    sampled = metrics.loc[~metrics["is_reference"].astype(bool)]
    for source_index, source in enumerate(sources):
        points = sampled.loc[sampled["source"].astype(str) == source]
        if points.empty:
            continue
        axis.scatter(
            points["substrate_area_mm2"],
            points["max_s11_db"],
            points["mean_tot_eff_db"],
            s=24,
            alpha=0.48,
            color=color_map(source_index % 10),
            linewidths=0.0,
            label=f"Samples: {source}",
            depthshade=True,
        )

    pareto = metrics.loc[metrics["is_pareto"].astype(bool)]
    axis.scatter(
        pareto["substrate_area_mm2"],
        pareto["max_s11_db"],
        pareto["mean_tot_eff_db"],
        s=66,
        marker="o",
        facecolors="#e63946",
        edgecolors="#3a0a0e",
        linewidths=0.8,
        alpha=0.95,
        label=f"Sampled Pareto set (n={len(pareto)})",
        depthshade=False,
    )

    references = metrics.loc[metrics["is_reference"].astype(bool)]
    if not references.empty:
        axis.scatter(
            references["substrate_area_mm2"],
            references["max_s11_db"],
            references["mean_tot_eff_db"],
            s=190,
            marker="*",
            facecolors="#ffd166",
            edgecolors="#111111",
            linewidths=1.2,
            label=f"Reference / origin (n={len(references)})",
            depthshade=False,
        )
        for _, row in references.iterrows():
            axis.text(
                float(row["substrate_area_mm2"]),
                float(row["max_s11_db"]),
                float(row["mean_tot_eff_db"]),
                f"  {row['case_id']}",
                fontsize=8,
            )

    lower, upper = band_ghz
    axis.set_xlabel("Total substrate area (mm²)  ↓", labelpad=10)
    axis.set_ylabel("Maximum S11 (dB)  ↓", labelpad=10)
    axis.set_zlabel("Mean Tot_Eff (dB)  ↑", labelpad=10)
    axis.set_title(
        "DOE sampling quality: three-objective scatter\n"
        f"[{lower:g}, {upper:g}] GHz · {len(sampled)} sampled cases · "
        f"{len(pareto)} non-dominated",
        pad=18,
    )
    axis.view_init(elev=24, azim=-56)
    axis.grid(True, alpha=0.35)
    axis.legend(loc="upper left", framealpha=0.94)

    if output_figure is not None:
        output_figure = Path(output_figure)
        output_figure.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_figure, dpi=220)
    if show:
        plt.show()
    return figure, axis


def save_metrics_csv(metrics: pd.DataFrame, output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False, encoding="utf-8-sig")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        dest="sources",
        help="Result root or case directory; repeat to merge multiple sources.",
    )
    parser.add_argument(
        "--band",
        nargs=2,
        type=float,
        metavar=("LOW_GHZ", "HIGH_GHZ"),
        default=F5_BAND_GHZ,
        help="Inclusive analysis band in GHz (default: 3.1 4.8).",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=F5_OUTPUT_FIGURE,
    )
    parser.add_argument("--output-csv", type=Path, default=F5_OUTPUT_CSV)
    parser.add_argument(
        "--skip-invalid-cases",
        action="store_true",
        help=(
            "Report and skip malformed cases instead of failing. Cases missing "
            "S11.csv or Tot_Eff.csv are always skipped automatically."
        ),
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=F5_SHOW_FIGURE,
        help="Open the figure window (use --no-show for a headless run).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    sources = tuple(args.sources) if args.sources else F5_SOURCE_DIRECTORIES
    band_ghz = (float(args.band[0]), float(args.band[1]))
    metrics, skipped = collect_metrics(
        sources,
        band_ghz=band_ghz,
        skip_invalid_cases=args.skip_invalid_cases,
    )
    metrics = label_sampled_pareto(metrics)
    save_metrics_csv(metrics, args.output_csv)
    figure, _ = plot_pareto_3d(
        metrics,
        band_ghz=band_ghz,
        output_figure=args.output_figure,
        show=args.show,
    )
    if not args.show:
        plt.close(figure)

    sampled_count = int((~metrics["is_reference"].astype(bool)).sum())
    reference_count = int(metrics["is_reference"].astype(bool).sum())
    pareto_count = int(metrics["is_pareto"].astype(bool).sum())
    print(
        f"Loaded {len(metrics)} cases from {len(sources)} source(s): "
        f"sampled={sampled_count}, reference={reference_count}, "
        f"Pareto={pareto_count}, skipped={len(skipped)}"
    )
    for message in skipped:
        print(f"[skip] {message}")
    print(f"Metrics: {Path(args.output_csv).resolve()}")
    print(f"Figure:  {Path(args.output_figure).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
