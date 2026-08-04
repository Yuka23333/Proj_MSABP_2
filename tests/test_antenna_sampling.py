from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from scripts.automation import antenna_sampler
from scripts.automation import cst_build_msabp_geometry
from scripts.automation import cst_test_outgrown_branch
from scripts.geometry import antenna_outline


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
        inner_slot_order1_right_x_mm=53.0,
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


def test_range_precedence_is_resolved_per_parameter() -> None:
    config = _default_config()
    config["sampling"]["groups"]["cpw.guide.design"] = {
        "range": {
            "mode": "relative",
            "lower": -0.01,
            "upper": 0.01,
            "reference": "nominal",
        }
    }
    config["sampling"]["parameters"]["cpw_guide_p4_y_mm"] = {
        "range": {"mode": "absolute", "min": 10.5, "max": 11.5}
    }
    plan = antenna_sampler.resolve_sampling_plan(config)
    resolved = _resolved_by_name(plan)
    assert resolved["cpw_guide_p4_y_mm"].range_source == "parameter"
    assert resolved["cpw_guide_p3_p4_x_mm"].range_source == "group"
    assert resolved["rectangle_length_mm"].range_source == "global"
    assert not resolved["cpw_guide_p1_x_mm"].effective_sample


def test_enabled_zero_nominal_branch_requires_absolute_size_range() -> None:
    config = copy.deepcopy(_default_config())
    config["branches"]["reserved_up_1"]["enabled"] = True
    with pytest.raises(ValueError, match="use an absolute range"):
        antenna_sampler.resolve_sampling_plan(config)

    config["sampling"]["groups"]["inner.order2.reserved.up.size"] = {
        "range": {"mode": "absolute", "min": 0.5, "max": 1.0}
    }
    plan = antenna_sampler.resolve_sampling_plan(config)
    resolved = _resolved_by_name(plan)
    assert resolved["inner_slot_order2_reserved_up_length_mm"].range_source == "group"
    assert resolved["inner_slot_order2_reserved_up_width_mm"].range_source == "group"


def test_small_sampler_run_preserves_requested_row_count() -> None:
    plan = antenna_sampler.resolve_sampling_plan(_default_config(), n_samples=2)
    result = antenna_sampler.generate_samples(plan)
    assert len(result.frame) == 2
    assert {"sample_id", "geometry_valid", "geometry_error"}.issubset(
        result.frame.columns
    )
