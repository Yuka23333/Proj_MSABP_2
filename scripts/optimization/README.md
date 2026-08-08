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

## Round 2: local-size 512-candidate Sobol design

`doe_round2_sobol_local_512.json` keeps all sixteen K variables on `[0,1]`
while restricting each of the seven absolute millimetre variables to
`[0.95,1.05]` times its nominal value. The design is one reproducible,
scrambled, power-of-two Sobol block with 512 candidates and no repeated origin
row. Prepare and geometry-audit it with:

```powershell
python scripts\optimization\prepare_doe_round1.py `
  --config configs\optimization\doe_round2_sobol_local_512.json
```

The current frozen design contains 499 simulation-eligible cases and 13
rejected geometries. Start or resume its two-Maid Princess run with:

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe `
  scripts\simulation\princess.py start `
  --csv data\samples\doe_round2_sobol_local_512.csv `
  --run-id doe-round2-sobol-local-512 `
  --device convallariag5 `
  --device coconutg2
```

## Resumable qLogEHVI campaign

`run_qlogehvi.py` uses the completed DoE and any additional repeated
`--source` directories as historical observations, then automatically adds its
own output directory to that source set. The current continuation campaign is
`q=4`, 100 new target evaluations, and `[3.1, 4.8] GHz`; it uses both the
original 512-point DoE and `msabp-qlogehvi-gpu-001` as historical sources.

The default controller uses a relay topology: campaign ownership remains on
the Princess host, while qLogEHVI fitting and acquisition optimization run in
`float64` on the `coconutg2` V100 through its `bocuda` environment.

```text
coconutg2 GPU proposal
        │ 4 candidates
        ▼
Princess host (authoritative plan/state/results)
        │ worklist
        ├──────────────► convallariag5 Maid/CST ──┐
        └──────────────► coconutg2 Maid/CST ──────┤
                                                  ▼
                                      Princess host results
                                                  │ compact training arrays
                                                  └────────► next GPU proposal
```

The two roles on `coconutg2` are independent processes and environments:
`bocuda` only calculates BO candidates, while `maid` controls local CST. The
GPU worker never owns the budget, reads CST artifacts, or writes final case
results. The Princess host remains the only campaign state authority.
The `bocuda` environment must contain `numpy`, `pandas`, `torch`, `botorch`,
`gpytorch`, and `ninja`; the smoke-tested interpreter is
`C:\Users\telecom\miniforge3\envs\bocuda\python.exe`.

Two fixed-noise `SingleTaskGP` outputs model the unstandardized linear RF
objectives. The third acquisition output is exact negative scaled substrate
area, computed directly from the candidate dimensions with zero posterior
variance. Its fixed physical reference is 1.01 times the maximum area allowed
by the input space; dividing physical area by this value makes the
minimization-form area reference exactly `1` (and the internal maximization
reference exactly `-1`).
The joint `q=4` acquisition is optimized continuously in the normalized
23-dimensional unit cube; it is not restricted to a pre-generated candidate
pool. The default settings use 256 raw starts, 8 restarts, 64 MC samples,
and at most 100 optimizer iterations.
The maximization-form objectives used by qLogEHVI are therefore:

1. `-max(|S11|)` in the band;
2. mean linear `Tot_Eff` in the band;
3. `-substrate_area_mm2 / area_reference_mm2`.

Individual linear Tot_Eff samples above 1 are discarded before the arithmetic
mean is taken. A
geometry-preflight failure or a CST task that exhausts its configured attempts
is recorded as `S11=1`, `Tot_Eff=0`; these RF values lie on the fixed
hypervolume reference boundary, so an invalid small board gains no
hypervolume. A geometry-preflight rejection intentionally has only a manifest,
not CST curves; its embedded penalty objectives are nevertheless retained as
training data. A manifest that has neither complete curves nor penalty
objectives is treated as genuinely incomplete and skipped. Infrastructure
failures that leave Princess tasks pending or running do not consume points
and keep the batch resumable.

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
  --source results\raw\msabp-qlogehvi-gpu-001 `
  --output results\raw\msabp-qlogehvi-area-scaled-001 `
  --budget 100 --q 4
```

`--stop-after-proposal` asks coconutg2 for one proposal and persists the
resumable batch without starting Princess. The local controller interpreter no
longer performs the heavy BoTorch calculation. The default proposal settings
are equivalent to:

```powershell
--proposal-backend remote_cuda `
--proposal-device coconutg2 `
--proposal-python C:\Users\telecom\miniforge3\envs\bocuda\python.exe `
--proposal-compute-device cuda
```

`--proposal-backend local_cpu` remains available as a diagnostic fallback.

## RF-GP IMSE comparison

`analyze_bo_imse.py` compares the latent RF-GP posterior variance after the
initial DoE, after BO1, and after BO2. It freezes one 1024-point scrambled Sobol
set in the normalized 23-dimensional search cube, then performs the two RF
objective fits independently at all three cumulative stages (six fits total).
The local controller only parses results and transfers compact arrays; all
float64 fitting and posterior evaluation run in `bocuda` on the coconutg2 V100.
It does not connect to CST or modify optimization state.

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe `
  scripts\optimization\analyze_bo_imse.py
```

The JSON and tabular CSV outputs are written under
`results/processed/bo_imse_1024/`. Use `--prepare-only` to build the immutable
request without contacting coconutg2, or `--summarize-only` to re-render an
already retrieved response. `--doe-parity-split` instead divides the stable
498-row DoE order into 1-based odd and even 249-row subsets and compares both
subsets on the same integration points. `--demi-full-source PATH` compares the
stable 1-based odd rows, even rows, and full observation set from any one result
directory on that same fixed integration set.

For each batch, the controller writes
`_qlogehvi/batch_NNNN_proposal_request.json`, atomically uploads it, invokes
`qlogehvi_gpu_worker.py` through a short synchronous SSH command, and atomically
retrieves `batch_NNNN_proposal_response.json`. Only normalized training arrays,
objective arrays, bounds, settings, and summary counts are transferred. A
SHA-256 request identity is embedded in the remote filenames and response, so
an interrupted retry can safely reuse an already completed proposal without
creating a second batch.
The request also fingerprints the normalized qLogEHVI and relay Python sources;
a GPU host on a different Git revision is rejected instead of silently reusing
or producing candidates with different code.
Each completed batch records the actual remote Python, Torch, BoTorch,
GPyTorch, Ninja, CUDA runtime, GPU name, and compute capability in proposal
diagnostics; the plan's software block describes only the local controller.

On Windows the GPU worker adds the active Conda environment's `Scripts`
directory to PATH for `ninja`, discovers Visual Studio Build Tools with
`vswhere`, and loads the x64 compiler variables into its own process.
This enables BoTorch's fused qLogEHVI extension without changing system-wide
environment variables.
