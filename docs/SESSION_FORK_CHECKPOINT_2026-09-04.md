# Codex session fork checkpoint — 2026-09-04

This file is the shared handoff point for two Codex continuations:

1. continue the late-stage four-objective optimization research;
2. consolidate the successful, reproducible results for a conference paper.

Use the Git commit containing this file as the common code baseline. Do not
assume that a fresh clone contains the simulation archive: `results/raw/`,
`results/processed/`, and `results/figures/` are intentionally ignored.

## Shared verified baseline

- Full test suite at the checkpoint: `282 passed`.
- Ruff passes for every Python file added or modified in this checkpoint.
- Optimization evaluation band: 3.1--4.8 GHz.
- The authoritative CST template for the single-antenna workflow remains
  `simulations/models/msa-bp.cst`.
- Keep working CST infrastructure unchanged when it already solves correctly.
  In particular, do not infer material or monitor validity solely from how an
  independently opened project is rendered in the CST GUI.

## Track A — deep optimization research

Primary entrypoints and immutable round configurations:

- `scripts/optimization/run_krvea.py`: original four-objective K-RVEA line;
- `scripts/optimization/深度优化_krvea.py` with
  `configs/optimization/deep_krvea_round5.json`: isolated late-stage policy;
- `scripts/optimization/深度优化_阶段2_krvea.py` and
  `scripts/optimization/深度优化_阶段2_relay.py` with
  `configs/optimization/deep_krvea_stage2_round6.json`: learned-noise stage;
- `scripts/optimization/深度优化_阶段3_krvea.py` with
  `configs/optimization/deep_krvea_stage3_round7.json`: geometry-feasible
  proposal-pool stage.

Completed local campaign archives contain 128 cases in the smoke round and 64
cases in each of rounds 2--6:

```text
results/raw/msabp-krvea-11var-smoke-128-001
results/raw/msabp-krvea-11var-calibrated-64-002
results/raw/msabp-krvea-11var-calibrated-64-003
results/raw/msabp-krvea-11var-deep-64-004
results/raw/msabp-krvea-11var-deep-64-005
results/raw/msabp-krvea-11var-stage2-learned-64-006
```

Stage 3 is deliberately not described as complete. Its configured budget is
32, while the local archive currently contains 8 case manifests. A later batch
failed before campaign-state mutation because the feasibility pool produced
only 3 valid exploitation candidates for a requested 4. Decide whether to
enlarge/fallback the feasible proposal pool or stop optimization before
resuming this round.

The exact campaign policy, objective definitions, relay topology, and resume
rules are documented in `scripts/optimization/README.md`. Never edit a JSON
file after its immutable plan has been created; copy it to a new round config.

## Track B — conference consolidation

The current single-antenna default reference is recorded by
`data/samples/current_default_reference.csv` and its local completed result is:

```text
results/raw/current-default-reference-001/case_current_default_reference
```

The measured four-objective values at this checkpoint are:

| Metric | Value |
|---|---:|
| Worst in-band S11 amplitude | 0.457962165 |
| Return loss corresponding to worst S11 | 6.783 dB |
| Mean in-band total efficiency | 0.867175033 |
| Normalized substrate area | 1.0 (2720.2 mm2) |
| Polar-cap average realized gain | 1.904418 linear (2.797623 dBi) |

The propagation comparison workspace is local at
`results/processed/propagation_s21_14/`. It contains 13 selected designs plus
the earlier baseline, common 5 Mbps UWB BER inputs/results, and two additional
Touchstone comparisons. The reproducible pipeline is:

```text
scripts/postprocessing/ber_01_generate_binary_sequences.py
scripts/postprocessing/ber_02_modulate_uwb_pulses.py
scripts/postprocessing/ber_03_average_s21.py
scripts/postprocessing/ber_04_run_experiment.py
scripts/postprocessing/ber_05_evaluate_touchstone_reference.py
```

Reference naming is now authoritative as follows:

- `MSA-BP_New-Notch_22-10-2013_CST2023.s2p` is the Roblin--Wei 2012 reference;
- `MSA-BP_David_2_body.s2p` is an unverified historical trial and must not be
  presented as the literature reference.

The corresponding processed directories are
`roblin_wei_2012_reference/` and `david_2_body_unverified_trial/`. Raw filenames
remain unchanged for provenance. Local raw-file identities are:

```text
771BB226D6C41EF8EC49B298F95C81F871498896AFE80B7E549FD51CCDFE7B83  MSA-BP_New-Notch_22-10-2013_CST2023.s2p
A43A62B288611F9E568BFAE0BF82738A1085552CCEFD9B027DAC0DFBA9281F57  MSA-BP_David_2_body.s2p
```

Current comparable 5 Mbps matched-filter results include:

| Case | Average S21 | Required Tx-reference Eb/N0 at BER 1e-4 |
|---|---:|---:|
| Selected rank 20 | -38.313770 dB | 43.883 dB |
| Roblin--Wei 2012 reference | -39.952858 dB | 45.432165 dB |
| Unverified David trial | -46.486641 dB | 50.164303 dB |

Treat the local raw/processed files as experimental evidence, not Git-backed
artifacts. Before moving either continuation to another clone or machine, copy
the required result directories separately and verify the two hashes above.
