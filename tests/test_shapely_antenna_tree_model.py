"""The arbitrary-tree geometry model must remain independent of its GUI."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
from shapely.affinity import scale

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
