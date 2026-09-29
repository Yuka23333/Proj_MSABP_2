# Arbitrary-tree Morris screening

This workflow is separate from BO/K-RVEA and does not change their failure policies.
No CST solver runs during sampling or analysis. The simulation entry point requires
`RUN` (or an explicit `--yes`). Master-template distribution remains the default.

## Initial experiment

Edit `configs/optimization/morris_tree_35d.json` **before** preparation:

- 7 absolute dimensions: default value times `[0.9, 1.1]`.
- 10 corner ratios and 18 branch ratios: `[0, 1]`.
- Six nodes: `U1`, `U1/L1`, `U1/R1`, `D1`, `D1/L1`, `D1/R1`.
  The generator works on the right half: `L1` is inward (I), `R1` outward (O).
  The existing left/right mirror shares every parameter; it adds no variables.
- 16 trajectories, each containing **35 + 1 = 36** positions: **576 positions**.
- Four-level grid; normalized step `Delta = p/(2*(p-1)) = 2/3`.
  This is global Morris screening, not a small-step local derivative.
- Ratios equal to zero are allowed. Collapsed/inactive features can produce
  genuine zero elementary effects; these observations are retained.
- Manufacturing repair v2, 0.1 mm minimum feature, 0.05 mm snap threshold,
  0.01 mm final coordinate grid. Other build options are explicit in the JSON.
- Device defaults are frozen in the campaign plan at preparation. The supplied
  config now selects local/G2/G5, while the existing first campaign was prepared
  with only `local`. Use explicit repeated `--device` to change a new dispatch;
  editing the input JSON does not rewrite an existing plan or dispatch.

`range_overrides` maps an exact variable name to an absolute `[lower, upper]` pair,
e.g. `"U1.K3": [0.05, 1]`. The resolved 35-name list is in `plan.json`.

## Prepare (offline)

```powershell
python scripts/optimization/prepare_morris_tree.py
```

F5 on this script performs preparation only. Outputs live under
`simulations/runs/morris-tree-35d-016-001/`:

- `plan.json`: immutable resolved parameters, ranges, metrics, source hashes.
- `template.cst`: frozen copy of the master at preparation, never opened here.
- `batches/batch_0001/candidates.jsonl`: every tested candidate trajectory,
  all its normalized coordinates, manufacturing labels and repair/rejection reports.
- `batch.json`: selected trajectory positions, physical requests, geometry reuse map.
- `sample.csv`: canonical unique physical models in Princess `antenna_tree` format.

Generate random Morris trajectories; inspect **every position** using the actual
tree builder and manufacturing checks. Reject a whole candidate if any position
fails manufacturing. Unexpected programming/geometry exceptions stop preparation
instead of being silently reclassified. Collect 32 valid trajectories by default,
then select 16 using greedy maximin mean inter-point Euclidean distance. This is
a spread heuristic, not a globally optimal Campolongo selection. Sampling and
selection use geometry/input coordinates only, never performance outputs.

An explicit candidate limit prevents an endless search. Incomplete preparation
cannot be dispatched. Its diagnostics are kept. Retrying preparation reserves the
next unused batch number; it never overwrites an interrupted batch. If the ranges
or model must change, use a new campaign directory/id instead.

Identical **full** physical models may reuse one simulation: the key contains
final copper, substrate bounds, reflector and quantum; the template and build
code are fixed for the campaign. Every original input position remains in the
analysis matrix. The actual solve count can therefore be smaller than 576.

## Run/resume/retry

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts/simulation/run_morris_tree.py --prepare-only
C:\Users\David\.conda\envs\cstpy\python.exe scripts/simulation/run_morris_tree.py
```

F5 settings are `CAMPAIGN` and `DISPATCH_ID` at the top of the launch script.
Confirm `RUN` to start Princess. Remote example, only when devices are ready:

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts/simulation/run_morris_tree.py --device convallariag5 --device coconutg2
```

The launcher checks frozen inputs and performs a fresh-process, read-only source
hash/import probe on each selected Maid **before** launching any solver. It does
not push/pull code or change licenses, CST templates, ports, monitors or boundaries.
Synchronize required code first. Do not edit implementation files during a run;
post-solve disk hashes cannot identify code already cached in a live process.
The probe sets `CONDA_PREFIX` and prepends the environment root, `Library/bin`,
and `Scripts` to its child-process PATH, matching Maid Bell. Calling conda's
absolute python.exe directly is insufficient for some Windows native DLL imports.
No permanent environment changes are made. To check without a solve:

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts/simulation/run_morris_tree.py --dispatch-id d002 --device local --device coconutg2 --device convallariag5 --preflight-only
```

- Reuse `d001` to resume its exact frozen Princess worklist and durable state.
- `max_attempts=3` means three total attempts, not three retries.
- A CST failure is an unresolved execution/infrastructure problem, **not** a
  manufacturing rejection, and is never assigned an artificial objective value.
- When attempts are exhausted, fix the cause, then retry unresolved **same points**:

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts/simulation/run_morris_tree.py --dispatch-id d002
```

The new dispatch has a new run id/result folder. It does not edit the previous
Princess database, replace trajectories, or resimulate verified completed points.
Do not run two campaign dispatches simultaneously. Other runs with running leases
or unstopped pending tasks block dispatch; resume/stop the previous one deliberately
first. An OS-released campaign lock also prevents concurrent launcher instances.

## Analyze and extend

```powershell
python scripts/postprocessing/analyze_morris_tree.py --campaign simulations/runs/morris-tree-35d-016-001
python scripts/optimization/prepare_morris_tree.py --append-trajectories 8
C:\Users\David\.conda\envs\cstpy\python.exe scripts/simulation/run_morris_tree.py --dispatch-id d003
```

Appending freezes a **new** batch, retains old trajectories/results and selects
additional spread relative to previous trajectories. A 35D increment of eight
trajectories adds 288 positions, potentially fewer new physical solves. Changing
topology, ranges, grid, repair, model code or template requires a new campaign.

Analysis verifies requests, model/source hashes and artifact hashes. It reports:

- worst linear `|S11|` over 3.1-4.8 GHz (`10**(dB/20)`, maximum, no mean);
- mean linear Rad_Eff and Tot_Eff (`10**(dB/10)`, uniform frequency integral);
- rectangular substrate area normalized by the frozen default reference area.

Efficiency values outside physical bounds are flagged and omitted, not clipped;
remaining valid data must cover the band without extrapolation. Removed counts
and gaps are reported. These are uniform-band screening metrics, not September
three-channel weighting or a reinstatement of retired Cap Gain objectives.

For each output: normalized-input signed elementary effects, `mu`, `mu_star`,
`sigma`, and trajectory-bootstrap confidence intervals. Effects are computed on
the original parameter coordinates, never inferred from repaired geometry.
No failed point is removed and its former neighbours joined into a false edge.
By default incomplete trajectories prevent final ranking; explicit partial mode
is exploratory, reports effective counts, and never marks the campaign complete.

Interpretation is **conditional on the selected fully manufacturable trajectories**,
not an unbiased sensitivity estimate over the entire original hypercube.
Pre-filter candidate labels preserve manufacturing-feasibility information.
Confidence intervals quantify sampled-trajectory variability, not selection bias,
CST discretization error or guaranteed global convergence. Sensitivity rankings
support screening; they do not prove a variable can safely be fixed everywhere.

References: [SALib Morris sampler](https://salib.readthedocs.io/en/latest/_modules/SALib/sample/morris/morris.html),
[SALib missing-output discussion](https://github.com/SALib/SALib/issues/273).
