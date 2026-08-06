from __future__ import annotations

from scripts.automation import check_sampled_curve_intersections
from scripts.geometry import shapely_antenna_model


def test_bottom_edge_contact_is_the_only_ignored_intersection() -> None:
    payload = {
        "vertices": {
            "Slot": [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)],
            "Patch": [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)],
            "CPW_Feed_Pin": [
                (1.25, 0.0),
                (1.75, 0.0),
                (1.75, 0.5),
                (1.25, 0.5),
            ],
        }
    }

    result = check_sampled_curve_intersections.inspect_curve_boundaries(payload)

    assert result["bottom_contact_pairs"]
    assert result["non_bottom_intersection"] is False
    assert result["non_bottom_intersection_pairs"] == ""


def test_current_baseline_reports_the_fixed_slot_pin_top_overlap() -> None:
    result = check_sampled_curve_intersections.inspect_curve_boundaries(
        shapely_antenna_model.polygon_export_payload()
    )

    assert result["non_bottom_intersection"] is True
    assert result["non_bottom_intersection_pairs"] == "Slot__CPW_Feed_Pin"
    assert result["non_bottom_crossing_pairs"] == ""
    assert result["non_bottom_overlapping_pairs"] == "Slot__CPW_Feed_Pin"
