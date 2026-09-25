# Arbitrary-tree geometry model

`scripts/geometry/shapely_antenna_tree_model.py` is the headless geometry layer
extracted from the G5 tree demo at commit `b503ff1`. It requires Shapely, not
Tkinter, Matplotlib or CST. Importing and building do not open windows, write
files or run simulations.

## Minimal use

Run from the repository root:

```python
from scripts.geometry import shapely_antenna_tree_model as model

params = model.default_params()
tree = model.default_tree()
child = model.add_branch(tree, "U1", "R", k=(0.5, 0.5, 0.5))
shapes = model.build(params, tree)

copper = shapes["Patch"].difference(shapes["Slot"]).union(
    shapes["CPW_Feed_Pin"]
)
substrate = shapes["Substrate_Full"]

model.delete_subtree(tree, child)
```

## Data contract

- `params`: 17 named nonbranch parameters returned by `default_params()`.
- `tree`: an insertion-ordered dictionary, with parents before their children.
  Each node has `parent`, `side` (`U`, `D`, `L`, `R`) and `k` (three numbers).
  The root is `SLOT`; IDs such as `U1/R1/U1` identify descendant paths.
- The default tree has two branches, hence 23 scalar parameters in total.
  Each added node adds three K parameters; topology remains a separate choice.
- `child_sides(tree, parent)` supplies the two perpendicular growth directions.
- `add_branch` mutates the tree and returns the new node ID.
- `delete_subtree` removes the node and all descendants, rather than zeroing K.
- `param_range` and `k_range` preserve the demo's slider ranges; they do not
  validate or clamp values passed to `build`. Only root-attached branch K2
  permits zero in those slider ranges. Children of a zero-width branch retain
  the original center-line attachment behavior.
- `build` returns `Substrate_Full`, `Patch`, `Slot`, `CPW_Feed_Pin`, `SMA_Pads`,
  `slot_main`, `Upper_Substrate`, `Lower_Substrate`, `branches`, and `metal_area`.
  Geometry values are native Shapely objects; `branches` contains per-node
  construction information and `metal_area` is a scalar.
- Left/right mirroring remains `(x, y) -> (-x, y)`. Polygon holes and separate
  copper islands remain represented by Shapely, without flattening to one ring.

## Integration boundary

### K2 saturation

All tree depths use an upper saturation band on branch K2 (not corner K2).
For `s = 0.10`, raw K2 up to 0.8 stays unchanged, 0.8--0.9 maps linearly
to 0.8--1, and 0.9--1 maps to exactly 1. K1 and K3 are unchanged. Raw K2
is retained in the tree and `branches[id]["k"]`; the transformed value is
available as `branches[id]["effective_k2"]`. The demo displays both.

Use `build(params, tree, snap_fraction=0)` to disable saturation, or supply
another fraction in `[0, 0.5]`. K2 must be finite and in `[0, 1]`.
At saturation the width touches the nearer end of the attachment face exactly;
for a child positioned in the tipward half this is the parent's tip. K1 at
either endpoint can still give zero width. No growth boundary is relaxed.
The mapping is continuous but has derivative corners and a flat upper band.

The tree demo now imports this model, and its existing autoplay stays compatible.
The production `shapely_antenna_model.py`, sampler, polygon export schema and CST
build routing are not switched to this model yet. No extra geometry rejection,
quantization or export adapter is introduced by this extraction.

## Verification

```powershell
python -m pytest tests/test_shapely_antenna_tree_model.py -q
```

Before adding K2 saturation, extraction was checked against the original demo on 100 seeded
parameter/tree cases: exact geometry WKB and returned metadata matched. A
withdrawn-window GUI/autoplay smoke check passed 30 edits. Neither check used CST.
