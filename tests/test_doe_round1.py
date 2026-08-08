from __future__ import annotations

from dataclasses import asdict

import pytest

from scripts.automation import antenna_sampler
from scripts.geometry import shapely_antenna_model
from scripts.optimization import prepare_doe_round1


ROUND2_CONFIG_PATH = (
    prepare_doe_round1.REPOSITORY_ROOT
    / "configs"
    / "optimization"
    / "doe_round2_sobol_local_512.json"
)


def _valid_audit_row() -> dict[str, object]:
    return {
        **asdict(shapely_antenna_model.DEFAULT_PARAMETERS),
        "sample_id": 0,
        "geometry_valid": True,
        "geometry_error": "",
        "non_bottom_intersection_pairs": "Slot__CPW_Feed_Pin",
        "non_bottom_crossing_pairs": "",
        "non_bottom_touching_pairs": "",
        "non_bottom_overlapping_pairs": "Slot__CPW_Feed_Pin",
    }


def test_round1_config_is_512_candidate_latin_hypercube() -> None:
    config = prepare_doe_round1.load_round_config()

    assert config["sampling"] == {
        "method": "latin",
        "candidate_count": 512,
        "seed": 20260806,
    }
    assert config["eligibility"]["allowed_non_bottom_overlap_pairs"] == [
        "Slot__CPW_Feed_Pin"
    ]


def test_round1_rejects_invalid_and_unexpected_intersections() -> None:
    allowed = {"Slot__CPW_Feed_Pin"}
    valid = _valid_audit_row()
    assert (
        prepare_doe_round1._initial_rejection_reason(
            valid,
            allowed_non_bottom_overlap_pairs=allowed,
        )
        == ""
    )

    invalid = {**valid, "geometry_valid": False, "geometry_error": "bad Patch"}
    assert prepare_doe_round1._initial_rejection_reason(
        invalid,
        allowed_non_bottom_overlap_pairs=allowed,
    ) == "bad Patch"

    crossing = {
        **valid,
        "non_bottom_intersection_pairs": "Slot__Patch;Slot__CPW_Feed_Pin",
        "non_bottom_crossing_pairs": "Slot__Patch",
    }
    assert "Slot__Patch" in prepare_doe_round1._initial_rejection_reason(
        crossing,
        allowed_non_bottom_overlap_pairs=allowed,
    )


def test_default_row_preflights_all_full_cst_model_parts() -> None:
    dimensions = prepare_doe_round1._full_model_preflight(_valid_audit_row())

    assert dimensions["substrate_width_mm"] == pytest.approx(67.0)
    assert dimensions["substrate_height_mm"] == pytest.approx(40.6)
    assert dimensions["reflector_cutout_width_mm"] == pytest.approx(51.96)
    assert dimensions["reflector_cutout_depth_mm"] == pytest.approx(0.5)


def test_origin_row_is_named_tracked_and_full_model_valid() -> None:
    row, dimensions = prepare_doe_round1._build_origin_audit_row(
        allowed_non_bottom_overlap_pairs={"Slot__CPW_Feed_Pin"},
    )

    assert row["sample_id"] == "origin"
    assert row["doe_source"] == "origin"
    assert row["geometry_valid"] is True
    assert row["BRANCH_DOWN_1_K3"] == pytest.approx(0.0)
    assert dimensions["substrate_width_mm"] == pytest.approx(67.0)


def test_latin_candidate_frame_occupies_every_one_dimensional_stratum() -> None:
    config = antenna_sampler.load_sampling_config()
    plan = antenna_sampler.resolve_sampling_plan(
        config,
        n_samples=4,
        method="latin",
        seed=7,
    )
    candidates = antenna_sampler.generate_parameter_frame(plan)

    prepare_doe_round1._verify_latin_hypercube(candidates, plan)
    candidates.loc[1, "SLOT_MAIN_LENGTH"] = candidates.loc[0, "SLOT_MAIN_LENGTH"]
    with pytest.raises(ValueError, match="SLOT_MAIN_LENGTH"):
        prepare_doe_round1._verify_latin_hypercube(candidates, plan)


def test_round2_sobol_uses_local_absolute_ranges_and_full_k_ranges() -> None:
    round_config = prepare_doe_round1.load_round_config(ROUND2_CONFIG_PATH)
    sampling = round_config["sampling"]
    assert sampling == {
        "method": "sobol",
        "candidate_count": 512,
        "seed": 20260808,
        "include_origin": False,
    }
    base_config = antenna_sampler.load_sampling_config(
        prepare_doe_round1.REPOSITORY_ROOT
        / round_config["base_sampling_config"]
    )
    plan = antenna_sampler.resolve_sampling_plan(base_config)
    assert plan.method == "sobol"
    assert plan.n_samples == 512

    for item in plan.resolved_parameters:
        assert item.lower is not None
        assert item.upper is not None
        if item.spec.kind == "absolute":
            assert item.lower == pytest.approx(0.95 * item.nominal)
            assert item.upper == pytest.approx(1.05 * item.nominal)
        else:
            assert item.lower == 0.0
            assert item.upper == 1.0

    candidates = antenna_sampler.generate_parameter_frame(plan)
    prepare_doe_round1._verify_sobol_design(candidates, plan)
    assert len(candidates) == 512
