# Arbitrary-tree manufacturing projection

The tree geometry path now defaults to `repair`. Legacy fixed-tree and old
antenna scripts are unchanged. No CST templates, boundary settings, or existing
simulation results are modified by this change.

Current implementation version: **`tree-manufacturing-v2`**. It adds local joint
collars and facing-boundary classification; see the [300-point regression
report](MFG_CORNER_REPAIR_AUDIT_20260927.md).

## Contract

- Minimum **local** copper width and empty gap width: **0.1 mm**. An island is
  checked by width, not by area alone. Coordinate quantization remains **0.01 mm**.
- Width below 0.05 mm: remove that material feature. Width from 0.05 mm inclusive
  to 0.1 mm exclusive: widen to at least 0.1 mm. Width >= 0.1 mm is retained.
  Outward rounding can produce 0.11 mm rather than shrinking below the minimum.
- Removing a copper bridge disconnects copper; removing a gap fills copper.
  Tiny rectangular islands/holes are removed or enlarged in both axes together.
- Work on the full left/right mirrored final Boolean geometry, never each half
  separately. A 0.02 mm central bridge is one feature, not two 0.01 mm features.
- Preserve the feed region plus a minimum-feature-sized halo and SMA pads.
  Reject repairs touching that mask; do not patch protected material back later.
- If a widened strip still joins its parent through a narrow axial throat in
  [threshold, minimum), add a local rectangular collar to the copper (or cut it
  out to widen a gap). The collar extends minimum/2 past both endpoints, rounds
  outward onto the coordinate grid, and is mirrored before applying it. This is
  a bounded local edit, not global dilation and not necessarily minimum-area repair.
- Check both copper and its complement again after each repair and after final
  quantization. Reject cycles, iteration exhaustion, invalid polygons, point
  contacts, loss of symmetry/feed integrity, or remaining narrow proximities.

The implemented local-width detector is an exact slab test for axis-aligned
features in this model. It is **not** Shapely `minimum_clearance` (which also flags
harmless short edges), and not a minimum-area rule. The proximity check first
removes redundant collinear vertices for detection only, then uses consistently
oriented boundary normals and the material along the connecting segment. Both
edges must face into the same copper throat, or away from the same empty gap.
A short step with no opposing faces is not a minimum-width violation.

True oblique narrow features, diagonal proximities, and sub-threshold residual
joints needing a new topological removal choice remain conservatively rejected.
There are at most 8 joint-collar repairs and 12 total repair passes. Each edit is
checked against the original protected feed mask (on the same coordinate grid
after quantization), never against a progressively relaxed mask. `reject` mode
does not add collars; cycles, limit exhaustion and no-progress edits are errors.
This is a model-specific DRC, not a general curved-PCB certification.
Shared polygon corners and boundary-collinear short edges are not treated as
zero-width features. A 1e-9 mm numeric cleanup grid and 1e-8 tolerances only handle
floating-point Boolean noise; they are not fabrication allowances.

## Configuration

Tree request JSON accepts these `build_options` in addition to existing K2/tip
options (defaults are now written into normalized requests):

```json
{
  "build_options": {
    "snap_fraction": 0.05,
    "tip_clearance": 1.0,
    "manufacturing_mode": "repair",
    "min_feature_mm": 0.1,
    "feature_snap_mm": 0.05
  }
}
```

`manufacturing_mode="reject"` checks without feature repair. `"off"` explicitly
reproduces historical geometry and does NOT certify manufacturability. It is
needed when comparing to old 005/007 results: default processing now changes
their geometry. Use a NEW run ID for new manufacturing-aware simulations.
No optimizer strategy or remote deployment is changed in this update.

## Single source of truth and provenance

`shapely_antenna_tree_model.build()` returns:

- `Copper_Raw`: exact unquantized final Boolean copper before repair.
- `Copper`: authoritative manufactured copper for the demo and CST exporter.
- `Manufacturing`: version, settings, actions, before/after topology and areas.
- Existing `Patch`, `Slot`, `branches`: raw construction guides, NOT a way to
  reconstruct the manufactured copper.

The CST payload retains `copper_raw`, repaired `copper`, the manufacturing report
and `manufactured_copper_sha256`. The manifest records the report, copper hash
and manufacturing source hash. Local timing sampling deduplicates by the repaired
copper hash, so different raw parameters collapsed onto one result are not run
as distinct geometries. Future optimizer integrations must use this same key.

Action bounds use X and Y measured from the substrate bottom. Area differences
include both repair and coordinate quantization. The tree/K values are retained
unchanged so the repair is visible rather than hidden in mutated parameters.
`joint_cap` actions record the material phase, input distance, endpoint pair,
mirrored collar bounds, and added/removed area. `joint_repairs` records the count.
`ManufacturingError.report` is JSON-serializable; the Maid runner saves rejection
reports under `manufacturing_rejections` before touching CST. Demo rejection clears
the stale plot and displays a warning; random autoplay checks raw shaper rules.

## Offline verification

```powershell
python -m pytest tests/test_manufacturing.py tests/test_shapely_antenna_tree_model.py tests/test_tree_workflow.py
python scripts/geometry/validate_tree_manufacturing.py
```

The audit uses frozen 005 requests, writes NEW results under
`results/processed/manufacturing_validation_20260927`, and plots sample_008 before
and after. It does not open CST or run a solver. Passing DRC does not establish
mesh convergence or physical equivalence to a previous thin-feature design.

Frozen paper-audit requests can additionally be replayed with:

```powershell
python scripts/geometry/audit_tree_manufacturing.py --requests <requests.jsonl> --output <new-output-directory>
```

Add `--baseline-module <archived-manufacturing.py>` to compare previously passing
geometries, not just recorded pass/fail labels. The archived source and model
hashes must match the input audit metadata. Full replay checks all passing
geometries for repeat-repair idempotence, mirror symmetry and remaining features.
