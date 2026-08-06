from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.automation import cst_build_msabp_geometry
from scripts.geometry import antenna_polygon_export
from scripts.geometry import shapely_antenna_model


def test_current_export_is_read_without_closing_or_requantizing() -> None:
    exported = antenna_polygon_export.load_antenna_polygon_export()

    assert exported.quantize_step_mm == pytest.approx(0.01)
    assert exported.substrate_bounds_mm == pytest.approx((-33.5, 0.0, 33.5, 40.6))
    assert exported.substrate_size_mm == pytest.approx((67.0, 40.6))
    assert len(exported.vertices["Patch"]) == 40
    assert len(exported.vertices["Slot"]) == 37
    assert len(exported.vertices["CPW_Feed_Pin"]) == 12
    assert exported.vertices["Patch"][0] != exported.vertices["Patch"][-1]


def test_parameterized_defaults_reproduce_current_export_exactly() -> None:
    exported = antenna_polygon_export.load_antenna_polygon_export()
    generated = shapely_antenna_model.polygon_export_payload()

    for curve_name in ("Slot", "Patch", "CPW_Feed_Pin"):
        assert generated["vertices"][curve_name] == list(
            exported.vertices[curve_name]
        )


def test_exported_specs_reuse_substrate_reflector_and_clearance() -> None:
    exported = antenna_polygon_export.load_antenna_polygon_export()
    specs, report = cst_build_msabp_geometry.build_exported_polygon_specs()
    substrate, patch, slot, guide, reflector, clearance = specs

    assert patch.points == exported.points("Patch")
    assert slot.points == exported.points("Slot")
    assert guide.points == exported.points("CPW_Feed_Pin")
    assert substrate.points == [
        (-33.5, 0.0),
        (33.5, 0.0),
        (33.5, 40.6),
        (-33.5, 40.6),
        (-33.5, 0.0),
    ]
    assert reflector.points == substrate.points
    assert clearance.points == [
        (-25.98, 0.0),
        (25.98, 0.0),
        (25.98, 0.5),
        (-25.98, 0.5),
        (-25.98, 0.0),
    ]
    assert report.point_counts == (5, 40, 37, 12, 5, 5)
    assert report.substrate_area_mm2 == pytest.approx(67.0 * 40.6)
    assert report.reflector_cutout_width_mm == pytest.approx(51.96)


def test_reader_does_not_repeat_geometry_legality_checks(tmp_path: Path) -> None:
    payload = {
        "meta": {"quantize_step": 0.01},
        "vertices": {
            "Patch": [[-33.5, 0.0], [33.5, 0.0], [33.5, 40.6], [-33.5, 40.6]],
            "Slot": [[-1.0, 1.0], [1.0, 3.0], [-1.0, 3.0], [1.0, 1.0]],
            "CPW_Feed_Pin": [[-0.5, 0.0], [0.5, 0.0], [0.0, 2.0]],
        },
    }
    path = tmp_path / "self-crossing-transport-example.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    exported = antenna_polygon_export.load_antenna_polygon_export(path)
    specs, _ = cst_build_msabp_geometry.build_exported_polygon_specs(path)

    assert exported.points("Slot") == [
        (-1.0, 1.0),
        (1.0, 3.0),
        (-1.0, 3.0),
        (1.0, 1.0),
    ]
    assert specs[2].points == exported.points("Slot")
