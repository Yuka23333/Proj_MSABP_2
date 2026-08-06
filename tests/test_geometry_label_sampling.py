from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.automation import antenna_sampler
from scripts.automation import sample_geometry_labels
from scripts.automation import scan_geometry_feasibility
from scripts.geometry import shapely_antenna_model


def test_default_config_defines_exact_23_variable_space() -> None:
    config = antenna_sampler.load_sampling_config()
    plan = antenna_sampler.resolve_sampling_plan(config, n_samples=2)

    assert config["schema_version"] == 2
    assert tuple(config["sampling"]["parameters"]) == (
        shapely_antenna_model.PARAMETER_NAMES
    )
    assert len(plan.resolved_parameters) == 23
    assert sum(item.effective_sample for item in plan.resolved_parameters) == 23


def test_redesigned_scenario_has_23_active_dimensions() -> None:
    config = antenna_sampler.load_sampling_config()
    configured = sample_geometry_labels.configure_scenario(config, "redesigned_23d")
    plan = antenna_sampler.resolve_sampling_plan(configured, n_samples=2)

    assert tuple(sample_geometry_labels.SCENARIO_BRANCH_STATES) == ("redesigned_23d",)
    assert sum(item.effective_sample for item in plan.resolved_parameters) == 23


def test_parameter_frame_matches_serial_sampler_inputs() -> None:
    config = sample_geometry_labels.configure_scenario(
        antenna_sampler.load_sampling_config(),
        "redesigned_23d",
    )
    plan = antenna_sampler.resolve_sampling_plan(config, n_samples=4)
    frame = antenna_sampler.generate_parameter_frame(plan)
    serial = antenna_sampler.generate_samples(plan).frame

    assert len(frame) == 4
    assert "geometry_valid" not in frame
    for name in ("sample_id", *antenna_sampler.PARAMETER_REGISTRY):
        np.testing.assert_array_equal(frame[name].to_numpy(), serial[name].to_numpy())


def test_chunk_labels_and_pickle_free_npy_round_trip(tmp_path: Path) -> None:
    defaults = asdict(shapely_antenna_model.DEFAULT_PARAMETERS)
    invalid = dict(defaults)
    invalid["SLOT_MAIN_LENGTH"] = 0.0
    rows = tuple(
        tuple(values[name] for name in sample_geometry_labels.PARAMETER_NAMES)
        for values in (defaults, invalid)
    )
    labels = sample_geometry_labels._validate_parameter_chunk(
        (rows, shapely_antenna_model.QUANTIZE_STEP_MM, False)
    )
    assert labels.tolist() == [True, False]

    frame = antenna_sampler.generate_parameter_frame(
        antenna_sampler.resolve_sampling_plan(
            sample_geometry_labels.configure_scenario(
                antenna_sampler.load_sampling_config(),
                "redesigned_23d",
            ),
            n_samples=2,
        )
    )
    labeled = sample_geometry_labels.build_labeled_array(frame, labels)
    output = sample_geometry_labels.save_labeled_array(
        labeled,
        tmp_path / "labels.npy",
    )
    loaded = np.load(output, allow_pickle=False)

    assert loaded.shape == (2,)
    assert loaded.dtype.hasobject is False
    assert loaded.dtype.names is not None
    assert loaded.dtype.names[-1] == "geometry_valid"
    assert loaded["geometry_valid"].tolist() == [True, False]


def test_feasibility_scan_windows_are_inclusive() -> None:
    assert scan_geometry_feasibility.build_half_width_percentages(5.0, 50.0, 5.0) == (
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
    )


def test_scan_only_widens_not_yet_relative_parameters() -> None:
    config = scan_geometry_feasibility.configure_scan_window(
        antenna_sampler.load_sampling_config(),
        "redesigned_23d",
        5.0,
    )

    plan = antenna_sampler.resolve_sampling_plan(config, n_samples=2)
    resolved = {item.spec.name: item for item in plan.resolved_parameters}
    margin = resolved["PATCH_BRICK_1_SIDE_MARGIN"]
    assert margin.range_source == "parameter"
    assert margin.lower == 6.0 * 0.95
    assert margin.upper == 6.0 * 1.05

    special_absolute = resolved["SLOT_MAIN_LENGTH"]
    assert special_absolute.lower == 15.0
    assert special_absolute.upper == 60.0

    relative = resolved["UPPER_CORNER_NOTCH_1_K1"]
    assert relative.range_source == "parameter"
    assert relative.lower == 0.0
    assert relative.upper == 1.0
    assert set(scan_geometry_feasibility.globally_scanned_parameter_names(plan)) == (
        set(scan_geometry_feasibility.PERCENTAGE_SCANNED_PARAMETER_NAMES)
    )
