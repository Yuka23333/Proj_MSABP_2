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
objectives. The third acquisition output is exact negative substrate area,
computed directly from the candidate dimensions with zero posterior variance.
The joint `q=4` acquisition is optimized continuously in the normalized
23-dimensional unit cube; it is not restricted to a pre-generated candidate
pool. The default settings use 256 raw starts, 8 restarts, 64 MC samples,
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
  --output results\raw\msabp-qlogehvi-gpu-001 `
  --budget 200 --q 4
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
