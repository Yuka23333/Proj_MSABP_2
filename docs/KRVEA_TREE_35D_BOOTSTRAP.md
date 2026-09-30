# 35D tree K-RVEA: bootstrap and staged validation

Status (2026-09-30): offline adaptation and bootstrap generation completed.
No new CST solve or remote GP fit has been started by this adaptation.
The old 11D scripts and the geometry/CST workflow are unchanged.

## Frozen experiment

- Topology: `U1`, `U1/L1`, `U1/R1`, `D1`, `D1/L1`, `D1/R1`.
- 35 variables: 7 absolute dimensions (baseline ±10%), 10 corner K values,
  and 18 branch K values. All K bounds remain `[0,1]`.
- Geometry/manufacturing contract is inherited from
  `morris-tree-35d-016-001`: repair mode, 0.1 mm minimum feature,
  0.05 mm snap threshold, 0.01 mm coordinate quantum.
- The controller distributes its **frozen template**, not a device-local model.
  Template, geometry implementation, metric implementation, historical RoI report,
  manufactured geometry, artifact hashes, solver type and baseline history are
  checked. The latter is a recorded-history identity check, not an independent
  measurement of every internal CST solver setting.
- Devices: `coconutg2`, `convallariag5`. Local is deliberately excluded until its
  geometry environment is reconciled; no global device registry change was made.

Three objectives, all represented as minimization:

1. Maximum **linear amplitude** `|S11|` over `[3.1,4.8] GHz`.
2. Negative frozen-RoI radiation gain in dBi. This retains the historical
   `3.6 GHz`, `theta=[55,85]°`, `phi=[40,140]°`, `E_theta` definition:
   solid-angle-weighted linear power spatial average, then `10 log10`.
   Radiation efficiency is included once; impedance mismatch is excluded.
   There is no RoI refit or new propagation-validation claim here.
3. Exact quantized substrate area divided by reference area `2720.2 mm²`.
   All seven absolute parameters enter the area calculation. No GP is trained
   for area and its predicted standard deviation is exactly zero.

S11 and RoI alone use the existing Phase-2 surrogate machinery, including its
bounded S11 transform and physical-domain prediction conversion. The GPU request
uses `float64`; the existing V100/bocuda relay is retained.

## Dataset split

| Dataset | Physical geometries | Use before validation |
| --- | ---: | --- |
| Morris seed | 316 | Training |
| Supplemental training | 128 | Training; includes reference `tree_train_0000` |
| Holdout | 64 | Validation only |
| Total | 508 | 444 training + 64 held out |

Morris has 576 parameter positions but only 316 independently simulated physical
geometries. The 260 aliases remain in the audit, not as independent GP observations.
The seed is reused, not rerun.

Train and holdout are independently seeded LHS **candidate** streams. Manufacturing
filtering and physical-geometry deduplication mean the retained sets are **not a
strict LHS**. Current preparation accepted 192 unique new geometries and rejected
6 manufacturing-infeasible candidates. All three sets are physically disjoint.
The existing seed was re-audited against raw curves/FFS before provenance sealing.

The generated local-only campaign is:
`simulations/runs/krvea-tree-35d-roi-001/`.
It contains `train.csv`, `holdout.csv`, `points.json`, `candidate_audit.json`,
`morris_seed.json`, `template.cst`, `plan.json`, `seed_provenance.json`, and
`baseline_provenance.json`. These are intentionally ignored runtime artifacts.
Do not edit frozen files in place or regenerate a different sample under the same ID.

## Commands (PowerShell at repository root)

Use the local Python 3.12 geometry environment for these wrappers. They explicitly
launch Princess using `cstpy`; the proposal worker runs remotely in `bocuda`.

Inspect only; no SSH, fitting or solver:

```powershell
& "C:\Program Files\Python312\python.exe" scripts/optimization/run_phase2_krvea_tree.py
```

Print both Princess commands without launching them:

```powershell
& "C:\Program Files\Python312\python.exe" scripts/optimization/prepare_krvea_tree.py --dispatch all
```

After normal Git push/pull deployment, start 128 training points and then 64 holdout
points. This asks for `RUN`; only this command starts CST:

```powershell
& "C:\Program Files\Python312\python.exe" scripts/optimization/prepare_krvea_tree.py --dispatch all --execute
```

`--dispatch train` / `--dispatch holdout` select a single split. IDs are stable for
Princess resume. Do not launch separate competing controllers on the same Maids.
After forced termination, explicitly stop/recover the old dispatch before switching
to another. Exhausted cases remain unresolved and require inspection; they are not
silently converted into poor electromagnetic values or auto-resubmitted forever.

After both splits complete, audit curves/FFS and derive the frozen RoI (no GP):

```powershell
& "C:\Program Files\Python312\python.exe" scripts/optimization/run_phase2_krvea_tree.py --action collect
```

Prepare the remote validation request without fitting:

```powershell
& "C:\Program Files\Python312\python.exe" scripts/optimization/run_phase2_krvea_tree.py --action validate
```

To actually fit the two GPs on 444 observations on CoconutG2 and predict the 64
held-out inputs, add `--execute`. Holdout labels are **not sent** to the GPU worker;
the controller subsequently computes physical-domain RMSE, MAE, normalized RMSE,
Spearman correlation, and nominal 95% interval coverage. Outputs are under
`validation/`. No automatic pass threshold or automatic optimization start exists.
If the holdout informs model changes, it becomes calibration evidence, not a fresh
independent final test set; a new locked test would be needed for that claim.

## Optimization gate and continuation

Configuration: `configs/optimization/phase2_krvea_tree_35d.json`.
Currently `optimization_budget=null`, `validation_approved=false`.
The new-point optimization budget has **not** been chosen or authorized.

Only after reviewing the saved validation report should those two fields be set.
Approval permits assimilating the holdout into training (508 initial observations).
The report/request/response hashes and current bootstrap data must still agree.
Other frozen configuration changes require a new campaign; this prevents reusing
an approval after silently changing the surrogate settings.

```powershell
# Prepare a proposal request only (still requires reviewed validation + budget).
& "C:\Program Files\Python312\python.exe" scripts/optimization/run_phase2_krvea_tree.py --action propose

# Explicitly fit/propose q=4 on CoconutG2; prepare a CSV, do not launch CST.
& "C:\Program Files\Python312\python.exe" scripts/optimization/run_phase2_krvea_tree.py --action propose --execute

# Explicitly run the fit -> propose -> Princess -> collect loop to the chosen budget.
& "C:\Program Files\Python312\python.exe" scripts/optimization/run_phase2_krvea_tree.py --action run --execute
```

The budget counts new registered optimization geometries only, not the Morris seed
or the 192 bootstrap solves. Resume reuses an unfinished batch/request, including
K-RVEA's previous empty-reference-vector count. Complete batch metadata/CSV are
published together; interrupted `.staging_*` directories are not registered jobs.

Manufacturing checks and full-geometry deduplication run before committing proposed
jobs. If fewer than q proposals survive, the script stops with an audit and consumes
no new-point budget. This is intentionally not a claim of a fully automatic
feasibility-aware candidate replenisher. Likewise, unresolved CST failures stop the
loop rather than train false labels. Review is required in either case.

## New code and deployment boundary

- `scripts/optimization/prepare_krvea_tree.py`: offline bootstrap + explicit Princess dispatch.
- `scripts/optimization/run_phase2_krvea_tree.py`: data audit, validation gate and batch loop.
- `scripts/optimization/phase2_krvea_tree_gpu_worker.py`: isolated remote request worker.
- `src/msabp_opt/optimization/phase2_krvea_tree_data.py`: dynamic tree input mapping and Morris import.
- `src/msabp_opt/optimization/phase2_krvea_tree_relay.py`: distinct tree wire protocol and exact area.
- `scripts/postprocessing/analyze_morris_tree_roi.py`: historical-RoI analysis dependency;
  include it in the next scoped commit if not already tracked.

Preparation never launches a solver or a real GP fit. Remote validation requires a
matching code deployment; its request embeds normalized source fingerprints.
Deployment checks use `--validate-only`, not a fit or a solve. Unit tests use fake
predictions where necessary; offline request checks are not evidence of remote
numerical convergence.
