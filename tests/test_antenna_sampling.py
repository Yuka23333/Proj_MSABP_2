from __future__ import annotations

import copy
import csv
from dataclasses import replace
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from scripts.automation import antenna_sampler
from scripts.automation import cst_build_msabp_geometry
from scripts.automation import cst_test_outgrown_branch
from scripts.geometry import antenna_outline
from scripts.geometry import shapely_antenna_model


def _default_config() -> dict:
    return antenna_sampler.load_sampling_config()


def _resolved_by_name(plan: antenna_sampler.SamplingPlan) -> dict:
    return {item.spec.name: item for item in plan.resolved_parameters}


def test_coordinate_quantization_uses_half_up_ties() -> None:
    assert antenna_outline.quantize_coordinate_mm(1.375) == 1.38
    assert antenna_outline.quantize_coordinate_mm(-1.375) == -1.38
    assert antenna_outline.quantize_coordinate_mm(-0.004) == 0.0


def test_individual_curve_rejects_self_intersection() -> None:
    bow_tie = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    with pytest.raises(ValueError, match="self-intersects"):
        antenna_outline.quantize_and_validate_closed_polygon_points(
            bow_tie,
            curve_name="bow tie",
        )


def test_individual_curve_rejects_edge_collapsed_by_quantization() -> None:
    narrow_edge = [
        (0.0, 0.0),
        (0.004, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]
    with pytest.raises(ValueError, match="collapsed"):
        antenna_outline.quantize_and_validate_closed_polygon_points(
            narrow_edge,
            curve_name="narrow edge",
        )


def test_default_cst_curves_are_quantized_and_individually_valid() -> None:
    specs, report = cst_build_msabp_geometry.build_polygon_specs()
    assert len(specs) == 6
    assert report.coordinate_quantum_mm == 0.01
    assert report.final_conductor_component_count == 1
    assert all(
        abs(coordinate * 100.0 - round(coordinate * 100.0)) < 1e-9
        for spec in specs
        for point in spec.points
        for coordinate in point
    )


def test_sampled_rectangles_send_quantized_points_to_cst() -> None:
    parameters = replace(
        antenna_outline.DEFAULT_ANTENNA_PARAMETERS,
        rectangle_length_mm=68.0636,
        rectangle_width_mm=39.9188,
    )
    specs, report = cst_build_msabp_geometry.build_polygon_specs(
        parameters=parameters,
    )

    substrate, _, _, _, reflector, _ = specs
    expected_points = [
        (-34.03, 0.0),
        (34.03, 0.0),
        (34.03, 39.92),
        (-34.03, 39.92),
        (-34.03, 0.0),
    ]
    assert substrate.points == expected_points
    assert reflector.points == expected_points
    assert report.substrate_area_mm2 == pytest.approx(68.06 * 39.92)


def test_branch_anchor_ratio_tracks_parent_trunk() -> None:
    parameters = replace(
        antenna_outline.DEFAULT_ANTENNA_PARAMETERS,
        rectangle_length_mm=134.0,
    )
    reservations = antenna_outline.generate_inner_slot_order2_reservations(parameters)
    assert reservations[0].anchor == pytest.approx((44.0, 21.1))
    assert reservations[1].anchor == pytest.approx((10.0, 21.1))
    assert reservations[2].anchor == pytest.approx((40.0, 21.1))


def test_active_slot_branch_cannot_cover_sma_solder_keepout() -> None:
    parameters = replace(
        antenna_outline.DEFAULT_ANTENNA_PARAMETERS,
        inner_slot_order2_reserved_down1_enabled=True,
        inner_slot_order2_reserved_down1_length_mm=18.0,
        inner_slot_order2_reserved_down1_width_mm=1.0,
    )
    with pytest.raises(ValueError, match="SMA solder keepout"):
        antenna_outline.build_antenna_closed_polygons(parameters)


def test_disconnected_conductor_requires_explicit_policy() -> None:
    trial = cst_test_outgrown_branch.select_outgrown_branch_trial()
    with pytest.raises(ValueError, match="disconnected conductor"):
        cst_build_msabp_geometry.build_polygon_specs(parameters=trial.parameters)
    _, report = cst_build_msabp_geometry.build_polygon_specs(
        parameters=trial.parameters,
        allow_disconnected_conductor=True,
    )
    assert report.final_conductor_component_count > 1


def test_sampler_registry_has_exactly_seven_absolute_and_sixteen_ratio_variables() -> None:
    registry = antenna_sampler.PARAMETER_REGISTRY
    grouped = {
        name
        for parameter_names in antenna_sampler.PARAMETER_GROUPS.values()
        for name in parameter_names
    }

    assert tuple(registry) == shapely_antenna_model.PARAMETER_NAMES
    assert antenna_sampler.PARAMETER_GROUPS["absolute_mm"] == (
        shapely_antenna_model.ABSOLUTE_PARAMETER_NAMES
    )
    assert grouped == set(shapely_antenna_model.PARAMETER_NAMES)
    assert sum(spec.kind == "absolute" for spec in registry.values()) == 7
    assert sum(spec.kind == "ratio" for spec in registry.values()) == 16


def test_explorer_slider_modes_and_accumulated_parameter_updates() -> None:
    defaults = antenna_outline.DEFAULT_ANTENNA_PARAMETERS
    percent = antenna_outline.explorer_slider_spec("rectangle_length_mm", defaults)
    ratio = antenna_outline.explorer_slider_spec(
        "inner_slot_order2_reserved_up_anchor_t",
        defaults,
    )

    assert percent.mode == "percent"
    assert percent.parameter_value(150.0) == pytest.approx(100.5)
    assert percent.slider_value(100.5) == pytest.approx(150.0)
    assert ratio.mode == "ratio"
    assert ratio.parameter_value(0.25) == pytest.approx(0.25)

    after_length = replace(
        defaults,
        rectangle_length_mm=percent.parameter_value(110.0),
    )
    width = antenna_outline.explorer_slider_spec("rectangle_width_mm", defaults)
    after_width = replace(
        after_length,
        rectangle_width_mm=width.parameter_value(105.0),
    )
    assert after_width.rectangle_length_mm == pytest.approx(73.7)
    assert after_width.rectangle_width_mm == pytest.approx(41.79)


def test_explorer_rejects_undrawable_geometry_and_supports_embedded_plot() -> None:
    invalid = replace(
        antenna_outline.DEFAULT_ANTENNA_PARAMETERS,
        rectangle_length_mm=0.0,
    )
    with pytest.raises((TypeError, ValueError)):
        antenna_outline.validate_explorer_parameters(invalid)

    figure = Figure(figsize=(6.0, 4.0))
    axes = figure.add_subplot(111)
    returned_figure, returned_axes = antenna_outline.plot_complete_antenna(
        show=False,
        figure=figure,
        axes=axes,
    )
    assert returned_figure is figure
    assert returned_axes is axes
    figure.clear()


def test_range_precedence_is_resolved_per_parameter() -> None:
    config = copy.deepcopy(_default_config())
    config["sampling"]["parameters"]["SLOT_MAIN_LENGTH"] = {
        "value": 50.0,
        "range": {"mode": "absolute", "min": 10.5, "max": 11.5}
    }
    plan = antenna_sampler.resolve_sampling_plan(config)
    resolved = _resolved_by_name(plan)
    main_length = resolved["SLOT_MAIN_LENGTH"]
    assert main_length.nominal == pytest.approx(50.0)
    assert main_length.lower == pytest.approx(10.5)
    assert main_length.upper == pytest.approx(11.5)
    assert main_length.range_source == "parameter"
    assert all(item.range_source == "parameter" for item in resolved.values())
    assert all(item.effective_sample for item in resolved.values())


def test_ratio_nominals_and_ranges_stay_inside_unit_interval() -> None:
    resolved = _resolved_by_name(antenna_sampler.resolve_sampling_plan(_default_config()))

    for name in shapely_antenna_model.RATIO_PARAMETER_NAMES:
        item = resolved[name]
        assert 0.0 <= item.nominal <= 1.0
        assert item.lower == pytest.approx(0.0)
        assert item.upper == pytest.approx(1.0)
        assert item.spec.hard_min == pytest.approx(0.0)
        assert item.spec.hard_max == pytest.approx(1.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"lower_outer_slot_order1_opposite_corner_x_mm": 9.99},
    ],
)
def test_lower_outer_slots_cannot_enter_central_cpw_corridor(changes: dict) -> None:
    parameters = replace(antenna_outline.DEFAULT_ANTENNA_PARAMETERS, **changes)

    with pytest.raises(ValueError, match=antenna_outline.CPW_FEEDING_INTERFERENCE_ERROR):
        antenna_outline.build_antenna_closed_polygons(parameters)


def test_main_slot_uses_explicit_absolute_ranges() -> None:
    resolved = _resolved_by_name(antenna_sampler.resolve_sampling_plan(_default_config()))

    assert resolved["SLOT_MAIN_LENGTH"].lower == pytest.approx(15.0)
    assert resolved["SLOT_MAIN_LENGTH"].upper == pytest.approx(60.0)
    assert resolved["SLOT_MAIN_HEIGHT"].lower == pytest.approx(1.0)
    assert resolved["SLOT_MAIN_HEIGHT"].upper == pytest.approx(3.0)


def test_outer_order1_heights_are_substrate_y_ratios() -> None:
    parameters = replace(
        antenna_outline.DEFAULT_ANTENNA_PARAMETERS,
        rectangle_width_mm=100.0,
        upper_outer_slot_order1_height_ratio=0.25,
        lower_outer_slot_order1_height_ratio=0.3,
    )

    assert parameters.upper_outer_slot_order1_depth_mm == pytest.approx(25.0)
    assert parameters.upper_outer_slot_order1_bottom_y_mm == pytest.approx(75.0)
    assert parameters.lower_outer_slot_order1_height_mm == pytest.approx(30.0)


def test_outer_order2_vertical_ranges_are_parent_relative() -> None:
    upper_low = replace(
        antenna_outline.DEFAULT_ANTENNA_PARAMETERS,
        rectangle_width_mm=100.0,
        upper_outer_slot_order2_lower_y_ratio=0.0,
    )
    upper_high = replace(upper_low, upper_outer_slot_order2_lower_y_ratio=1.0)
    lower = replace(
        upper_low,
        lower_outer_slot_order1_height_ratio=0.4,
        lower_outer_slot_order2_branch1_width_ratio=0.25,
        lower_outer_slot_order2_branch1_offset_ratio=0.5,
        lower_outer_slot_order2_branch2_width_ratio=0.5,
        lower_outer_slot_order2_branch2_offset_ratio=-0.5,
    )

    assert upper_low.upper_outer_slot_order2_lower_y_mm == pytest.approx(60.0)
    assert upper_high.upper_outer_slot_order2_lower_y_mm == pytest.approx(100.0)
    assert lower.lower_outer_slot_order2_branch1_width_mm == pytest.approx(10.0)
    assert lower.lower_outer_slot_order2_branch1_center_y_mm == pytest.approx(35.0)
    assert lower.lower_outer_slot_order2_branch1_lower_y_mm == pytest.approx(30.0)
    assert lower.lower_outer_slot_order2_branch1_upper_y_mm == pytest.approx(40.0)
    assert lower.lower_outer_slot_order2_branch2_width_mm == pytest.approx(20.0)
    assert lower.lower_outer_slot_order2_branch2_center_y_mm == pytest.approx(10.0)
    assert lower.lower_outer_slot_order2_branch2_lower_y_mm == pytest.approx(0.0)
    assert lower.lower_outer_slot_order2_branch2_upper_y_mm == pytest.approx(20.0)


def test_three_outer_order2_lengths_are_ratios_of_width_minus_a_minus_b() -> None:
    parameters = antenna_outline.DEFAULT_ANTENNA_PARAMETERS
    half_board_x = parameters.rectangle_length_mm / 2.0
    protection_b = antenna_outline.OUTER_SLOT_CENTERLINE_PROTECTION_B_FIXED_MM
    expected_upper = (
        half_board_x - parameters.upper_outer_slot_order1_width_mm - protection_b
    )
    expected_lower = (
        half_board_x - parameters.lower_outer_slot_order1_width_mm - protection_b
    )

    assert parameters.upper_outer_slot_order2_max_length_mm == pytest.approx(
        expected_upper
    )
    assert parameters.upper_outer_slot_order2_line_length_mm == pytest.approx(
        parameters.upper_outer_slot_order2_length_ratio * expected_upper
    )
    assert parameters.lower_outer_slot_order2_max_length_mm == pytest.approx(
        expected_lower
    )
    assert parameters.lower_outer_slot_order2_branch1_line_length_mm == pytest.approx(
        parameters.lower_outer_slot_order2_branch1_length_ratio * expected_lower
    )
    assert parameters.lower_outer_slot_order2_branch2_line_length_mm == pytest.approx(
        parameters.lower_outer_slot_order2_branch2_length_ratio * expected_lower
    )


def test_sampler_uses_requested_ranges_for_all_23_variables() -> None:
    resolved = _resolved_by_name(
        antenna_sampler.resolve_sampling_plan(copy.deepcopy(_default_config()))
    )

    expected_absolute_ranges = {
        "SLOT_MAIN_LENGTH": (15.0, 60.0),
        "SLOT_MAIN_HEIGHT": (1.0, 3.0),
        "PATCH_BRICK_1_SIDE_MARGIN": (3.0, 9.0),
        "PATCH_BRICK_1_TOP_MARGIN": (1.3, 3.9),
        "PATCH_BRICK_3_BOTTOM_MARGIN": (1.0, 3.0),
        "PATCH_BRICK_2_HEIGHT_MARGIN": (7.5, 22.5),
        "PATCH_BRICK_4_MARGIN": (2.0, 6.0),
    }
    for name, (lower, upper) in expected_absolute_ranges.items():
        assert resolved[name].lower == pytest.approx(lower)
        assert resolved[name].upper == pytest.approx(upper)
        assert resolved[name].range_source == "parameter"
    for name in shapely_antenna_model.RATIO_PARAMETER_NAMES:
        assert resolved[name].lower == pytest.approx(0.0)
        assert resolved[name].upper == pytest.approx(1.0)
        assert resolved[name].range_source == "parameter"


def test_downward_inner_branch_requires_five_mm_from_cpw_slot_and_stubs() -> None:
    parameters = replace(
        antenna_outline.DEFAULT_ANTENNA_PARAMETERS,
        inner_slot_order2_reserved_down1_enabled=True,
        inner_slot_order2_reserved_down1_anchor_t=8.0 / 26.5,
        inner_slot_order2_reserved_down1_length_mm=12.0,
        inner_slot_order2_reserved_down1_width_mm=1.0,
    )

    with pytest.raises(ValueError, match=antenna_outline.CPW_FEEDING_INTERFERENCE_ERROR):
        antenna_outline.build_antenna_closed_polygons(parameters)


def test_generated_parameter_frame_respects_all_resolved_ranges() -> None:
    plan = antenna_sampler.resolve_sampling_plan(_default_config(), n_samples=32)
    frame = antenna_sampler.generate_samples(plan).frame

    for item in plan.resolved_parameters:
        assert item.lower is not None
        assert item.upper is not None
        assert frame[item.spec.name].between(item.lower, item.upper).all()


def test_small_sampler_run_preserves_requested_row_count() -> None:
    plan = antenna_sampler.resolve_sampling_plan(_default_config(), n_samples=2)
    result = antenna_sampler.generate_samples(plan)
    assert len(result.frame) == 2
    assert {"sample_id", "geometry_valid", "geometry_error"}.issubset(
        result.frame.columns
    )


def test_saved_valid_rows_remain_valid_after_csv_float_round_trip(
    tmp_path: Path,
) -> None:
    plan = antenna_sampler.resolve_sampling_plan(_default_config(), n_samples=16)
    result = antenna_sampler.generate_samples(plan)
    output, _resolved = antenna_sampler.save_sampling_result(
        result,
        tmp_path / "samples.csv",
        valid_only=True,
    )

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    for row in rows:
        parameters = antenna_sampler.parameters_from_csv_row(row)
        cst_build_msabp_geometry.build_sampled_polygon_specs(
            parameters,
            coordinate_quantum_mm=plan.geometry_policy.coordinate_quantum_mm,
        )
