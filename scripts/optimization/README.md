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

## Resumable qLogEHVI campaign

`run_qlogehvi.py` uses the completed DoE and any additional repeated
`--source` directories as historical observations, then automatically adds its
own output directory to that source set. The default campaign is
`q=4`, 200 new target evaluations, and `[3.1, 4.8] GHz`.

The controller always runs BoTorch on CPU in `float64`; CUDA is never selected.
Two fixed-noise `SingleTaskGP` outputs model the unstandardized linear RF
objectives. The third acquisition output is exact negative substrate area,
computed directly from the candidate dimensions with zero posterior variance.
The joint `q=4` acquisition is optimized continuously in the normalized
23-dimensional unit cube; it is not restricted to a pre-generated candidate
pool. The default CPU settings use 256 raw starts, 8 restarts, 64 MC samples,
and at most 100 optimizer iterations.
The maximization-form objectives used by qLogEHVI are therefore:

1. `-max(|S11|)` in the band;
2. mean linear `Tot_Eff` in the band;
3. negative substrate area in mm².

Individual linear Tot_Eff samples above 1 are discarded before the arithmetic
mean is taken. A
geometry-preflight failure or a CST task that exhausts its configured attempts
is recorded as `S11=1`, `Tot_Eff=0`; these RF values lie on the fixed
hypervolume reference boundary, so an invalid small board gains no
hypervolume. Infrastructure failures that leave Princess tasks pending or
running do not consume points and keep the batch resumable.

First validate the sources and create the local plan without starting CST:

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe `
  scripts\optimization\run_qlogehvi.py --prepare-only
```

The target directory then contains `optimization_plan.json` with the immutable
budget and objective contract, plus `optimization_state.json`. If the direct
`case_*` count in that target ever exceeds the plan budget, the controller
refuses to continue and asks for a new plan directory.

To start or resume the real campaign, run the same script and type `RUN` (or
pass `--yes`). Re-running is the resume operation: an unfinished batch keeps
the same Princess run id and frozen worklist.

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe `
  scripts\optimization\run_qlogehvi.py `
  --source results\raw\doe-round1-lhs-512 `
  --source results\raw\another-compatible-run `
  --output results\raw\msabp-qlogehvi-001 `
  --budget 200 --q 4
```

`--stop-after-proposal` fits qLogEHVI and persists one resumable batch without
starting Princess. The interpreter used for proposal generation must contain
`torch`, `gpytorch`, and `botorch`; CST bindings are required only because the
same F5 entrypoint also starts Princess.

On Windows the controller adds the active Conda environment's `Scripts`
directory to PATH for `ninja`, discovers Visual Studio Build Tools with
`vswhere`, and loads the x64 compiler variables into the controller process.
This enables BoTorch's fused qLogEHVI extension without changing system-wide
environment variables.
