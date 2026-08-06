# Optimization scripts

## Round 1: 512-candidate LHS

`prepare_doe_round1.py` reads the shared 23-variable ranges from
`configs/optimization/antenna_sampling.json` and the round settings from
`configs/optimization/doe_round1_lhs_512.json`.

```powershell
python scripts\optimization\prepare_doe_round1.py
```

The script verifies one candidate per one-dimensional LHS stratum, checks all
three quantized antenna curves, rejects unexpected pairwise intersections, and
preflights the complete six-part CST geometry: dielectric substrate, Patch,
Slot, CPW feed pin, reflector, and reflector connector-clearance slot.

Outputs:

- `data/samples/doe_round1_lhs_512.csv`: accepted Princess/Maid worklist;
- `data/samples/doe_round1_lhs_512_candidates.csv`: all candidates and audit fields;
- `data/samples/doe_round1_lhs_512_rejected.csv`: rejected candidates only;
- `results/processed/doe_round1_lhs_512_summary.json`: counts, hashes, and model ranges.

The accepted worklist appends one `sample_id=origin` baseline row and records
`doe_source=lhs/origin`. Rejected candidate IDs are intentionally not
renumbered, so every accepted LHS row can be traced back to its original point.
