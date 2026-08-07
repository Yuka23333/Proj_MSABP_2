from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from scripts.postprocessing import plot_doe_pareto_3d


def _write_curve(path: Path, rows: list[tuple[float, float]]) -> None:
    text = "Frequency / GHz    value / dB\n" + "-" * 40 + "\n"
    text += "\n".join(f"{frequency} {value}" for frequency, value in rows)
    path.write_text(text + "\n", encoding="utf-8")


def _write_case(
    root: Path,
    name: str,
    *,
    case_id: str,
    area: float,
    s11: list[tuple[float, float]],
    efficiency: list[tuple[float, float]],
) -> None:
    case_directory = root / name
    case_directory.mkdir(parents=True)
    manifest = {
        "case_id": case_id,
        "geometry": {"substrate_area_mm2": area},
    }
    (case_directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    _write_curve(case_directory / "S11.csv", s11)
    _write_curve(case_directory / "Tot_Eff.csv", efficiency)


def test_read_curve_and_band_are_numeric_sorted_and_inclusive(tmp_path: Path) -> None:
    curve_path = tmp_path / "curve.csv"
    _write_curve(curve_path, [(4.8, -3.0), (3.1, -1.0), (3.0, -9.0)])

    frequency, values = plot_doe_pareto_3d.read_cst_1d_curve(curve_path)

    assert frequency.tolist() == [3.0, 3.1, 4.8]
    assert plot_doe_pareto_3d.values_in_band(
        frequency, values, (3.1, 4.8)
    ).tolist() == [-1.0, -3.0]


def test_pareto_directions_minimize_area_and_s11_maximize_efficiency() -> None:
    mask = plot_doe_pareto_3d.nondominated_mask(
        substrate_area_mm2=[1.0, 2.0, 1.0, 1.0],
        max_s11_db=[-10.0, -9.0, -10.0, -10.0],
        mean_tot_eff_db=[-3.0, -4.0, -2.0, -2.0],
    )

    # Row 2 dominates rows 0 and 1. Row 3 is its objective duplicate and is
    # deliberately retained on the same non-dominated front.
    assert mask.tolist() == [False, False, True, True]


def test_multiple_sources_reference_exclusion_and_metric_reduction(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "run_a"
    source_b = tmp_path / "run_b"
    common_s11 = [(3.0, -20.0), (3.1, -12.0), (4.0, -5.0), (4.8, -8.0)]
    _write_case(
        source_a,
        "case_0000",
        case_id="0",
        area=100.0,
        s11=common_s11,
        efficiency=[(3.1, -4.0), (4.8, -2.0)],
    )
    _write_case(
        source_b,
        "case_origin",
        case_id="origin",
        area=50.0,
        s11=common_s11,
        efficiency=[(3.1, -1.0), (4.8, -1.0)],
    )

    frame, skipped = plot_doe_pareto_3d.collect_metrics(
        [source_a, source_b],
        band_ghz=(3.1, 4.8),
    )
    labelled = plot_doe_pareto_3d.label_sampled_pareto(frame)

    assert skipped == []
    assert len(labelled) == 2
    sample = labelled.loc[labelled["case_id"] == "0"].iloc[0]
    origin = labelled.loc[labelled["case_id"] == "origin"].iloc[0]
    assert sample["max_s11_db"] == -5.0
    assert sample["mean_tot_eff_db"] == -3.0
    assert bool(sample["is_pareto"])
    assert bool(origin["is_reference"])
    assert not bool(origin["is_pareto"])


def test_plot_saves_headless_figure(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        {
            "source": ["run", "run"],
            "case_id": ["0", "origin"],
            "substrate_area_mm2": [100.0, 110.0],
            "max_s11_db": [-5.0, -4.0],
            "mean_tot_eff_db": [-3.0, -2.0],
            "is_reference": [False, True],
            "is_pareto": [True, False],
        }
    )
    destination = tmp_path / "plot.png"

    figure, _ = plot_doe_pareto_3d.plot_pareto_3d(
        metrics,
        band_ghz=(3.1, 4.8),
        output_figure=destination,
        show=False,
    )

    assert destination.is_file()
    assert destination.stat().st_size > 0
    figure.clear()
