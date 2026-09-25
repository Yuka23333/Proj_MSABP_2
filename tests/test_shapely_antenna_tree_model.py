"""The arbitrary-tree geometry model must remain independent of its GUI."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
from shapely.affinity import scale
from shapely.geometry import box

from scripts.geometry import shapely_antenna_tree_model as model


def test_import_and_build_without_gui_modules_or_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = f'''
import builtins, sys
sys.path.insert(0, {str(root)!r})
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('tkinter', 'matplotlib', 'cst'):
        raise AssertionError('Headless model attempted GUI/CST import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from scripts.geometry import shapely_antenna_tree_model as model
model.build(model.default_params(), model.default_tree())
'''
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_defaults_are_independent_and_build_does_not_mutate_inputs():
    params, tree = model.default_params(), model.default_tree()
    before = deepcopy((params, tree))
    geometry = model.build(params, tree)
    assert (params, tree) == before
    assert geometry["Slot"].is_valid
    assert geometry["Patch"].is_valid
    tree["U1"]["k"][0] = .8
    assert model.default_tree()["U1"]["k"][0] == .5
    params["SLOT_MAIN_LENGTH"] = 55
    assert model.default_params()["SLOT_MAIN_LENGTH"] == 53


def test_arbitrary_depth_mirroring_and_subtree_removal():
    tree = model.default_tree()
    parent = "U1"
    for side in ("R", "U", "L", "D", "R"):
        parent = model.add_branch(tree, parent, side)
    geometry = model.build(model.default_params(), tree)
    assert len(geometry["branches"]) == 7
    slot = geometry["Slot"]
    assert slot.symmetric_difference(scale(slot, xfact=-1, origin=(0, 0))).area < 1e-8
    model.delete_subtree(tree, "U1/R1")
    assert list(tree) == ["U1", "D1"]


def test_empty_tree_and_zero_width_parent_with_children():
    params = model.default_params()
    empty = model.build(params, {})
    assert empty["Slot"].is_valid
    tree = model.default_tree()
    tree["U1"]["k"][1] = 0
    child = model.add_branch(tree, "U1", "R")
    geometry = model.build(params, tree)
    assert geometry["branches"]["U1"]["branch"].area == 0
    assert geometry["branches"][child]["branch"].area > 0
    assert model.k_range(tree, "U1", 1) == (0, 1)
    assert model.k_range(tree, child, 1) == (.05, 1)


def test_demo_reexports_shared_model_functions():
    from scripts.geometry import shapely_antenna_tree_demo as demo

    for name in ("build", "default_tree", "default_params", "child_sides",
                 "add_branch", "delete_subtree", "k_range", "strip_reach"):
        if name == "strip_reach":
            # Internal geometric helpers intentionally stay in the model layer.
            assert hasattr(model, name)
        else:
            assert getattr(demo, name) is getattr(model, name)


def test_known_default_extent():
    geometry = model.build(model.default_params(), model.default_tree())
    bounds = geometry["Substrate_Full"].bounds
    assert bounds[2] - bounds[0] == pytest.approx(67)
    assert bounds[3] - bounds[1] == pytest.approx(40.6)


@pytest.mark.parametrize("raw,effective", [(0, 0), (.5, .5), (.9, .9), (.925, .95), (.95, 1), (.97, 1), (1, 1)])
def test_k2_relaxation(raw, effective):
    assert model.relax_upper_k(raw) == pytest.approx(effective)
    assert model.relax_upper_k(raw, 0) == raw


@pytest.mark.parametrize("raw,s", [(float('nan'), .1), (-.1, .1), (1.1, .1),
                                  (.5, -.1), (.5, .6), (.5, float('inf'))])
def test_invalid_relaxation(raw, s):
    with pytest.raises(ValueError):
        model.relax_upper_k(raw, s)


@pytest.mark.parametrize("direction", ['up', 'down', 'left', 'right'])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("k1", [0, .25, .5, .75, 1])
def test_snapped_face_bounds_in_all_directions(direction, reverse, k1):
    span = (2.3, 12.7) if not reverse else (12.7, 2.3)
    face = {'span': span, 'cap': span, 'base': 15}
    info = model._build_branch(k1, .97, .5, face, direction, box(0, 0, 30, 30))
    axis = 0 if direction in ('up', 'down') else 1
    lo, hi = [p[axis] for p in info['endpoints']]
    assert 2.3 - 1e-12 <= lo <= hi <= 12.7 + 1e-12
    if k1 < .5:
        assert (hi if reverse else lo) == span[0]
    elif k1 > .5:
        assert (lo if reverse else hi) == span[1]
    else:
        assert lo == 2.3
        assert hi == 12.7
    assert info['effective_k2'] == 1
    assert info['k'][1] == .97


def test_saturation_applies_recursively_without_mutating_tree():
    tree = model.default_tree()
    parent = 'U1'
    for side in ('R', 'D', 'L', 'U'):
        parent = model.add_branch(tree, parent, side, k=(.75, .97, .5))
    before = deepcopy(tree)
    saturated = model.build(model.default_params(), tree)
    linear = model.build(model.default_params(), tree, snap_fraction=0)
    assert tree == before
    for node in tree:
        assert saturated['branches'][node]['effective_k2'] == model.relax_upper_k(tree[node]['k'][1])
        assert linear['branches'][node]['effective_k2'] == tree[node]['k'][1]
    slot = saturated['Slot']
    assert slot.is_valid
    assert slot.symmetric_difference(scale(slot, xfact=-1, origin=(0, 0))).area < 1e-8


def _extend_tip(info, distance):
    """The branch rectangle pushed `distance` further along its growth direction."""
    dx, dy = model.DIRECTIONS[info["direction"]]
    min_x, min_y, max_x, max_y = info["branch"].bounds
    return box(min_x + min(dx, 0) * distance, min_y + min(dy, 0) * distance,
               max_x + max(dx, 0) * distance, max_y + max(dy, 0) * distance)


def test_full_length_branches_keep_tip_clearance_to_patch_edge():
    tree = model.default_tree()
    for node in tree.values():
        node["k"][2] = 1.0
    parent = "U1"
    for side in ("R", "U", "L", "D"):
        parent = model.add_branch(tree, parent, side, k=(.6, .5, 1.0))
    model.add_branch(tree, "D1", "R", k=(.4, .5, 1.0))
    geometry = model.build(model.default_params(), tree)
    patch = geometry["Patch"]
    clearance = model.BRANCH_TIP_CLEARANCE
    for node_id, info in geometry["branches"].items():
        if info["length"] > 0:
            # One clearance beyond the tip is still copper, never outside the patch.
            assert _extend_tip(info, clearance).difference(patch).area < 1e-9, node_id


def test_zero_tip_clearance_restores_flush_length():
    params, tree = model.default_params(), model.default_tree()
    guarded = model.build(params, tree)["branches"]["U1"]["max_length"]
    flush = model.build(params, tree, tip_clearance=0)["branches"]["U1"]["max_length"]
    assert flush - guarded == pytest.approx(model.BRANCH_TIP_CLEARANCE)


@pytest.mark.parametrize("clearance", [-0.1, float("nan"), float("inf")])
def test_invalid_tip_clearance(clearance):
    with pytest.raises(ValueError):
        model.build(model.default_params(), model.default_tree(), tip_clearance=clearance)


def test_inward_lower_branch_stops_clearance_short_of_feed():
    tree = model.default_tree()
    tree["D1"]["k"][2] = .8
    child = model.add_branch(tree, "D1", "L", k=(.5, .5, 1.0))
    geometry = model.build(model.default_params(), tree)
    branch = geometry["branches"][child]["branch"]
    assert branch.intersection(geometry["Feed_Region"]).area == 0
    assert branch.distance(geometry["Feed_Region"]) == pytest.approx(model.BRANCH_TIP_CLEARANCE)
    flush = model.build(model.default_params(), tree, tip_clearance=0)
    assert flush["branches"][child]["branch"].distance(flush["Feed_Region"]) == pytest.approx(0)
    assert flush["branches"][child]["branch"].intersection(flush["Feed_Region"]).area == 0


def test_feed_region_is_an_obstacle_at_every_depth():
    import random

    rng = random.Random(4)
    for _ in range(40):
        tree = model.default_tree()
        for _ in range(12):
            parent = rng.choice(["SLOT", *tree])
            side = rng.choice(model.child_sides(tree, parent))
            node = model.add_branch(tree, parent, side)
            tree[node]["k"] = [rng.uniform(*model.k_range(tree, node, i)) for i in range(3)]
            tree[node]["k"][2] = rng.choice([1.0, tree[node]["k"][2]])
        for clearance in (0, model.BRANCH_TIP_CLEARANCE):
            geometry = model.build(model.default_params(), tree, tip_clearance=clearance)
            for node_id, info in geometry["branches"].items():
                assert info["branch"].intersection(geometry["Feed_Region"]).area < 1e-9, node_id
