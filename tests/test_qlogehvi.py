from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msabp_opt.optimization import qlogehvi
from scripts.automation import antenna_sampler, cst_build_msabp_geometry
from scripts.geometry import shapely_antenna_model
from scripts.optimization import run_qlogehvi
from scripts.simulation import princess


def _default_parameters() -> dict[str, float]:
    return {
        name: float(value)
        for name, value in asdict(shapely_antenna_model.DEFAULT_PARAMETERS).items()
    }


def _input_space() -> qlogehvi.InputSpace:
    return qlogehvi.input_space_from_sampling_config(
        antenna_sampler.DEFAULT_CONFIG_PATH
    )


def _write_curve(path: Path, rows: list[tuple[float, float]]) -> None:
    text = "Frequency / GHz  value / dB\n" + "-" * 40 + "\n"
    text += "\n".join(f"{frequency} {value}" for frequency, value in rows)
    path.write_text(text + "\n", encoding="utf-8")


def _write_completed_case(
    root: Path,
    case_id: str,
    *,
    parameters: dict[str, float] | None = None,
) -> Path:
    case_directory = root / f"case_{case_id}"
    case_directory.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "status": "completed",
        "parameters": parameters or _default_parameters(),
        "artifacts": {
            "s11": {"path": "S11.csv"},
            "tot_eff": {"path": "Tot_Eff.csv"},
        },
    }
    (case_directory / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    _write_curve(
        case_directory / "S11.csv",
        [(3.0, -20.0), (3.1, -12.0), (4.0, -6.0), (4.8, -10.0)],
    )
    _write_curve(
        case_directory / "Tot_Eff.csv",
        [(3.0, -4.0), (3.1, -3.0), (4.0, 10.0), (4.8, -1.0)],
    )
    return case_directory


def test_exact_substrate_formula_matches_full_geometry_report() -> None:
    parameters = shapely_antenna_model.DEFAULT_PARAMETERS
    values = asdict(parameters)

    width, height, area = qlogehvi.substrate_dimensions_from_values(values)
    _, report = cst_build_msabp_geometry.build_sampled_polygon_specs(parameters)

    assert width == pytest.approx(67.0)
    assert height == pytest.approx(40.6)
    assert area == pytest.approx(report.substrate_area_mm2)


def test_linear_rf_metrics_discard_only_tot_eff_samples_above_one(
    tmp_path: Path,
) -> None:
    case_directory = _write_completed_case(tmp_path, "0")

    worst_s11, mean_efficiency, kept, removed = qlogehvi.rf_objectives_from_curves(
        case_directory / "S11.csv",
        case_directory / "Tot_Eff.csv",
        band_ghz=(3.1, 4.8),
    )

    assert worst_s11 == pytest.approx(10.0 ** (-6.0 / 20.0))
    assert mean_efficiency == pytest.approx(
        np.mean([10.0 ** (-3.0 / 10.0), 10.0 ** (-1.0 / 10.0)])
    )
    assert kept == 2
    assert removed == 1


def test_penalty_values_lie_on_rf_reference_boundary() -> None:
    input_space = _input_space()
    reference = qlogehvi.reference_point(input_space)

    assert reference[0] == -qlogehvi.PENALTY_WORST_S11
    assert reference[1] == qlogehvi.PENALTY_MEAN_TOT_EFF
    assert reference[2] < -qlogehvi.maximum_substrate_area(input_space)


def test_collect_observations_merges_sources_and_penalty_sidecar(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    case_a = _write_completed_case(source_a, "0")
    _write_completed_case(source_b, "1")
    penalty = qlogehvi.penalty_manifest_payload(
        case_id="0",
        parameters=_default_parameters(),
        failure_stage="objective_extraction",
        failure_message="solver defect",
        band_ghz=(3.1, 4.8),
    )
    sidecar = {
        "schema_version": 1,
        "case_id": "0",
        "failure": penalty["failure"],
        "optimization_objectives": penalty["optimization_objectives"],
    }
    (case_a / qlogehvi.OPTIMIZATION_PENALTY_FILENAME).write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    observations = qlogehvi.collect_observations(
        [source_a, source_b], band_ghz=(3.1, 4.8)
    )

    assert len(observations) == 2
    penalized = observations.loc[observations["case_id"] == "0"].iloc[0]
    assert bool(penalized["is_penalty"])
    assert penalized[qlogehvi.WORST_S11_COLUMN] == 1.0
    assert penalized[qlogehvi.MEAN_TOT_EFF_COLUMN] == 0.0


def test_training_arrays_prefer_completed_duplicate_over_penalty() -> None:
    input_space = _input_space()
    parameters = _default_parameters()
    rows = [
        {
            **parameters,
            "is_penalty": True,
            qlogehvi.WORST_S11_COLUMN: 1.0,
            qlogehvi.MEAN_TOT_EFF_COLUMN: 0.0,
            qlogehvi.AREA_COLUMN: 2720.2,
        },
        {
            **parameters,
            "is_penalty": False,
            qlogehvi.WORST_S11_COLUMN: 0.2,
            qlogehvi.MEAN_TOT_EFF_COLUMN: 0.8,
            qlogehvi.AREA_COLUMN: 2720.2,
        },
    ]

    train_x, train_y_rf, train_y_full, aggregate = qlogehvi.training_arrays(
        pd.DataFrame(rows), input_space
    )

    assert train_x.shape == (1, 23)
    assert train_y_rf.tolist() == [[-0.2, 0.8]]
    assert train_y_full.tolist() == [[-0.2, 0.8, -2720.2]]
    assert bool(aggregate.iloc[0]["has_completed_result"])


def test_preflight_default_candidate_is_valid() -> None:
    input_space = _input_space()
    raw = np.asarray(
        [_default_parameters()[name] for name in input_space.names], dtype=float
    )

    valid, error, report = qlogehvi.preflight_candidate(raw, input_space)

    assert valid, error
    assert report[qlogehvi.AREA_COLUMN] == pytest.approx(2720.2)


def test_plan_rejects_target_raw_count_above_budget(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = run_qlogehvi.CampaignConfig(
        plan_id="test-plan",
        source_directories=(source,),
        output_directory=tmp_path / "target",
        total_budget=1,
        device_ids=("maid-a",),
    )
    config.output_directory.mkdir()
    for case_id in ("0", "1"):
        case_directory = config.output_directory / f"case_{case_id}"
        case_directory.mkdir()
        (case_directory / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exceeds planned budget"):
        run_qlogehvi.target_case_directories(config)


def test_princess_cli_and_bo_command_support_explicit_results_root(
    tmp_path: Path,
) -> None:
    parsed = princess.build_parser().parse_args(
        [
            "start",
            "--csv",
            str(tmp_path / "batch.csv"),
            "--results-root",
            str(tmp_path / "results"),
        ]
    )
    assert parsed.results_root == tmp_path / "results"

    source = tmp_path / "source"
    source.mkdir()
    config = run_qlogehvi.CampaignConfig(
        plan_id="plan",
        source_directories=(source,),
        output_directory=tmp_path / "results",
        device_ids=("maid-a", "maid-b"),
    )
    command = run_qlogehvi.build_princess_command(
        config,
        {
            "worklist_csv": str(tmp_path / "batch.csv"),
            "run_id": "plan-batch-0000",
        },
    )
    assert "--results-root" in command
    assert command[command.index("--results-root") + 1] == str(config.output_directory)
    assert command.count("--device") == 2


def test_existing_plan_supplies_resume_defaults_without_repeating_cli(
    tmp_path: Path,
) -> None:
    source = tmp_path / "historical"
    source.mkdir()
    output = tmp_path / "custom-plan-output"
    output.mkdir()
    payload = {
        "plan_id": "custom-plan",
        "total_budget": 17,
        "q": 3,
        "band_ghz": [3.2, 4.7],
        "source_directories": [str(source)],
        "sampling_config": str(antenna_sampler.DEFAULT_CONFIG_PATH),
        "proposal_settings": {
            "q": 3,
            "seed": 99,
            "raw_samples": 40,
            "num_restarts": 4,
            "mc_samples": 12,
            "optimization_batch_limit": 1,
            "optimization_maxiter": 30,
            "gp_training_steps": 7,
            "gp_fixed_noise_variance": 1e-6,
        },
        "simulation": {
            "device_ids": ["maid-custom"],
            "device_config": str(run_qlogehvi.DEVICE_CONFIG),
            "project_template": str(run_qlogehvi.PROJECT_TEMPLATE),
            "coordinate_quantum_mm": 0.01,
            "allow_disconnected_conductor": False,
            "max_attempts": 4,
        },
    }
    (output / run_qlogehvi.PLAN_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    args = run_qlogehvi.parse_args(["--output", str(output), "--prepare-only"])
    config = run_qlogehvi._config_from_args(args)

    assert config.plan_id == "custom-plan"
    assert config.total_budget == 17
    assert config.band_ghz == (3.2, 4.7)
    assert config.source_directories == (source,)
    assert config.device_ids == ("maid-custom",)
    assert config.max_attempts == 4
    assert config.proposal.q == 3
    assert config.proposal.raw_samples == 40
    assert config.proposal.gp_training_steps == 7
