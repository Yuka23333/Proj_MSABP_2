"""Disconnected copper must survive export, extrusion, and later cleanup."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from shapely.affinity import scale
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from scripts.automation import cst_build_msabp_geometry as builder
from scripts.geometry import antenna_polygon_export as exchange
from scripts.geometry.conductor_components import (
    extract_conductor_components,
    validate_component_curves,
)
from scripts.geometry.prepare_copper_island_case import (
    stress_parameters,
    stress_payload,
)


def decode(payload):
    return exchange.polygon_export_from_payload(payload)


def test_stress_case_retains_all_five_regions_and_left_right_symmetry():
    payload = stress_payload()
    exported = decode(payload)
    parts = validate_component_curves(exported.vertices, exported.conductor_components)
    assert len(parts) == 5
    assert sum(p.area for p in parts) == pytest.approx(1356.648)
    assert [p.area for p in parts].count(pytest.approx(186.638)) == 2
    assert (
        sum(c["role"] == "floating_island" for c in payload["conductor_components"])
        == 2
    )
    copper = unary_union(parts)
    assert (
        copper.symmetric_difference(scale(copper, xfact=-1, origin=(0, 0))).area < 1e-9
    )
    for component in exported.conductor_components:
        assert Polygon(component.exterior).exterior.is_ccw


@pytest.mark.parametrize("mutation", ["missing", "overlap", "stale", "main_order"])
def test_reject_incomplete_stale_overlapping_or_misordered_export(mutation):
    payload = deepcopy(stress_payload())
    components = payload["conductor_components"]
    if mutation == "missing":
        components.pop()
    elif mutation == "overlap":
        components.append(deepcopy(components[-1]))
    elif mutation == "stale":
        components[-1]["exterior"][0][0] += 0.1
    else:
        components.reverse()
    exported = decode(payload)
    with pytest.raises(ValueError):
        validate_component_curves(exported.vertices, exported.conductor_components)


def enclosed_island_payload():
    # A closed rectangular slot loop stands in for intersecting second-order
    # branches. Its inner copper is entirely inside an uncut outer patch.
    vertices = {
        "Patch": list(box(-30, 0, 30, 40).exterior.coords)[:-1],
        "Slot": list(box(-8, 10, 8, 25).exterior.coords)[:-1],
        "CPW_Feed_Pin": list(box(-0.5, 0, 0.5, 5).exterior.coords)[:-1],
    }
    holes = {"Slot": [list(box(-6, 12, 6, 23).exterior.coords)[:-1]]}
    return {
        "meta": {"quantize_step": 0.01},
        "vertices": vertices,
        "source_holes": holes,
        "conductor_components": extract_conductor_components(vertices, holes),
    }


def test_enclosed_island_retains_hole_and_uses_explicit_tool_subtraction():
    exported = decode(enclosed_island_payload())
    parts = validate_component_curves(
        exported.vertices, exported.conductor_components, exported.source_holes
    )
    assert len(parts) == 2
    assert len(parts[0].interiors) == 1
    assert parts[1].area == pytest.approx(132)
    assert parts[1].distance(parts[0]) == pytest.approx(2)
    assert parts[0].exterior.equals(Polygon(exported.vertices["Patch"]).exterior)
    specs, report = builder._build_direct_polygon_specs(exported)
    commands = builder._build_live_vba_sequence(specs, report, "component1")
    code = "\n".join(vba for _, vba in commands)
    assert (
        'Solid.Subtract "component1:msabp_patch_solid", "component1:msabp_patch_solid_hole_001"'
        in code
    )
    assert (
        'actualVolume = actualVolume + Solid.GetVolume("component1_msabp_islands:msabp_island_001")'
        in code
    )


def test_island_build_uses_recorded_history_and_preserves_old_source_curves():
    project = SimpleNamespace(model3d=Mock(), schematic=Mock(), save=Mock())
    report = builder.build_msabp_in_cst(parameters=stress_parameters(), project=project)
    assert report.final_conductor_component_count == 5
    project.schematic.execute_vba_code.assert_not_called()
    project.save.assert_called_once()
    history = project.model3d.add_to_history.call_args_list
    labels = [call.args[0] for call in history]
    assert "boolean Patch - slot + guide" not in labels
    assert len([x for x in labels if x.startswith("extrude msabp_island_")]) == 4
    assert sum("restore copper_" in x for x in labels) == 5
    code = "\n".join(call.args[1] for call in history)
    assert "Sub Main()" not in code
    assert "msabp_symmetric_slot_curve" in code
    assert 'Solid.Insert "Connector' not in code


def test_default_build_also_cleans_previous_islands_and_keeps_old_route():
    specs, report = builder.build_sampled_polygon_specs(
        builder.shapely_antenna_model.DEFAULT_PARAMETERS
    )
    assert report.conductor_components == ()
    commands = builder._build_live_vba_sequence(specs, report, "component1")
    assert 'Component.Delete "component1_msabp_islands"' in commands[0][1]
    assert 'Curve.DeleteCurve "msabp_conductor_components"' in commands[0][1]
    assert "boolean Patch - slot + guide" in [label for label, _ in commands]


def test_source_holes_cannot_be_silently_ignored():
    payload = enclosed_island_payload()
    payload.pop("conductor_components")
    with pytest.raises(ValueError, match="source holes require"):
        decode(payload)


def test_producer_preserves_closed_slot_loop_instead_of_filling_the_island(monkeypatch):
    source = enclosed_island_payload()
    geometry = SimpleNamespace(
        patch=Polygon(source["vertices"]["Patch"]),
        slot=Polygon(source["vertices"]["Slot"], source["source_holes"]["Slot"]),
        cpw_feed_pin=Polygon(source["vertices"]["CPW_Feed_Pin"]),
    )
    monkeypatch.setattr(
        builder.shapely_antenna_model, "build_antenna_geometry", lambda _: geometry
    )
    payload = builder.shapely_antenna_model.polygon_export_payload(
        include_conductor_components=True
    )
    exported = decode(payload)
    parts = validate_component_curves(
        exported.vertices, exported.conductor_components, exported.source_holes
    )
    assert len(parts) == 2
    assert len(exported.source_holes["Slot"]) == 1
    assert parts[1].area == pytest.approx(132)
