# Manufacturing v2: frozen EuCAP audit regression

Date: 2026-09-27. Geometry-only offline replay; no CST solve, no changes to paper
inputs, templates, protected-feed rules, or optimizer strategy.

## Dataset and source identity

- Paper audit: 300 uniform samples, NumPy `default_rng(2027)`, 38 variables:
  seven absolute parameters at ±10%, ten corner K parameters, and seven tree
  nodes (`U1`, `U2`, `U1/L1`, `U1/R1`, `D1`, `D1/L1`, `D1/R1`).
- Frozen input: `D:/Academic/CSI_2026/deliverables/2027-04_EuCAP/data/mfg_audit_20260927/requests.jsonl`.
- Input SHA256: `f3f648042d67d7c6ba1102bd389e7f754ee755d91602bfb5e5e55bffc2ccb7f6`.
- Unchanged tree model SHA256: `8e3c3c4667e213de01a6cd99d9699bf41c30389c0ddd1a67f05c4f8f49d3c3dd`.
- Baseline manufacturing SHA256: `85d09a7ea3c4c232391812325b80e7e319df65d1f020f0f738491cf05bdef90c`.
- New manufacturing SHA256 at replay: `98b65588f0295dde75622c6bb711cca2ce0c8a303281b2a406ca731f5f7552d2`.
- Both baseline source hashes were verified against the paper audit metadata;
  every baseline pass/fail outcome was reproduced, not inferred from labels alone.

## What changed

1. Proximity requires **opposing boundary faces** and a connecting segment within
   the relevant material. Short steps and redundant collinear vertices no longer
   masquerade as narrow gaps.
2. A real axial endpoint throat can get a small mirrored rectangular collar:
   add copper for a copper neck, remove copper for a gap. Recheck the entire
   geometry after each edit and after quantization.
3. No unconditional diagonal repair, no feed-mask relaxation, no cycle bypass.

The 0.1 mm minimum, 0.05 mm snap threshold, and 0.01 mm coordinate grid are unchanged.

## Full 300-point result

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Passed | 273 | 294 |
| Rejected | 27 (9.0%) | 6 (2.0%) |
| Previously passing samples newly rejected | — | 0 |
| Previously passing copper geometries changed (area difference > 1e-8 mm²) | — | 0 |

All 294 passing v2 geometries passed repeat-repair idempotence (no further repair
actions; symmetric-difference area < 1e-8 mm²), symmetry and final feature checks.

**21 recovered samples**, using the paper dataset's **zero-based IDs**:
`8, 21, 22, 37, 89, 96, 107, 108, 110, 115, 122, 147, 159, 169, 180, 198, 223, 236, 247, 256, 269`.

- Four (`122, 147, 180, 247`) pass after the corrected proximity classification,
  without a joint collar (some retain ordinary pre-existing strip repairs).
- Seventeen pass with local joint-collar repair.

**Six remain rejected:**

| Zero-based sample IDs | Reason |
| --- | --- |
| 39, 116, 277 | Would modify protected feed/SMA region |
| 158 | Repair cycle; no safe projection found |
| 172, 251 | Remaining unsafe diagonal proximity |

The 2.0% rejection rate is descriptive of this fixed audit, not a general
acceptance guarantee. Passing these model-specific geometric tests is not a
fabrication certification or evidence of electromagnetic mesh convergence.

## Artifacts and replay

Local output directory: `results/processed/mfg_jointfix_audit_20260927/`.
`summary.json` contains provenance, outcomes, all repair reports and unchanged-
geometry comparisons. Each `sample_NNN.json` contains the original request and
either a failure report or raw/repaired copper GeoJSON (original model coordinates).
Repair-action bounds are measured from the substrate bottom, as documented in
the manufacturing contract. Do not confuse these IDs with benchmark `sample_008`.

```powershell
python scripts/geometry/audit_tree_manufacturing.py `
  --requests D:/Academic/CSI_2026/deliverables/2027-04_EuCAP/data/mfg_audit_20260927/requests.jsonl `
  --baseline-module tmp/manufacturing_v1_before_jointfix.py `
  --output results/processed/mfg_jointfix_audit_20260927
```

The baseline snapshot is local; omit `--baseline-module` if only checking recorded
outcome labels. A small portable subset of the paper requests is retained in
`tests/fixtures/manufacturing_corner_audit.json` for routine regression tests.
