# CoconutG2 SYSTEM CST solve probe

This directory archives the first successful real CST solve through the Shore
controller path. The one-shot task ran as `NT AUTHORITY\SYSTEM` in session 0,
built the standard default arbitrary-tree antenna on an isolated template copy,
ran the HF Time Domain solver, exported the required result closure, and closed
both the project and CST Design Environment.

- Request ID: `50fffd9d668640f08b7b5cdce49580e9`
- Host: `CoconutG2`
- Python: `C:\Users\telecom\miniforge3\envs\maid\python.exe` (3.11.15)
- Task result: `0`; the one-shot task was removed after completion
- Total invocation time: 77.863 s
- CST probe time: 76.850 s
- Solver stage: 56.084 s
- Frequency range: 2.0--6.0 GHz
- Source/template SHA-256 before and after:
  `d1962a2d16f976b7d1e4e9eb6d44baf1a460ae324897b4b2c892c0109715d3a8`
- Remote evidence root:
  `C:\ProgramData\ShorePrincess\project-msabp\diagnostics\cst-system-solve-50fffd9d668640f08b7b5cdce49580e9`

Required result closure:

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| `S11.csv` | 58,184 | `85c692d3b4696ce317a38648ccc56c2570e47709bf38b234f5cee0aaeaa726b2` |
| `Rad_Eff.csv` | 2,519 | `68cc6a0b2fd0152aba75d883bdaacc329261524478cb4071cbb82dcf9ce26205` |
| `Tot_Eff.csv` | 2,519 | `161ba96661dc7c5e31fe7c8dfdf84ac201afd04b33158c471b5959bed91544bd` |
| `Farfield Source [1].ffs` | 9,642,535 | `fa04b4e835d6ccf2a3ef01b526c0fe5be496247de9bc8e28368760da24100c01` |
| `manifest.json` | 30,207 | `4ca09e23cbbaed7025587dcf6a986e7bf5ada0e232f2440ba7455767842cedc9` |

Evidence hashes:

- `invocation.json`: `7fa8d775b6bdb4ba58ffd852e98ee0e7a3b1d8b20d91991239bbf654d1a41117`
- `report.json`: `58dd83dfb26684fffc1e4d31dd82cf7088753e77a3f45942cb3ac76f925c179c`
- `source_bundle.zip`: `44b5bc0cc67deab7235a3209380310da3201fab4a54a74f74110a8804c1ae3d4`
- `probe_cst_system_solve.py`: `c49bbc1fd5cb8c55a262a7d5e92f7c6d29805b2504e6292fba84341877ab152a`
- `Invoke-CstSystemSolveProbeRemote.ps1`: `06a477dc30ec5caed70dcac1e7fd874bb094c7e84ecca53c6bce87653590457d`

The Git-tracked evidence keeps this README, `invocation.json`, `report.json`,
`manifest.json`, and the three small CSV curves. The 9.6 MB FFS, source bundle,
and duplicate source snapshots remain in this local evidence directory and at
the remote evidence root, but are intentionally not committed; the hashes above
bind those retained files to the successful invocation without duplicating
large or already tracked payloads in Git.

## Root cause fixed during validation

The first two task launches exited before producing a report with Windows code
`0xC06D007F`, the MSVC delay-load `ERROR_PROC_NOT_FOUND` exception. The same
code was reproduced with a minimal non-interactive `import shapely`. An
absolute Conda `python.exe` path is not sufficient on Windows: its native DLL
directories were absent from the child process search path.

The task bootstrap now sets process-local `CONDA_PREFIX` and prepends the Conda
environment root, `Library\bin`, and `Scripts` to `PATH` before starting Python.
No user or system environment variable is changed. A minimal Shapely import
then returned 0, and the real SYSTEM solve completed successfully. A separate
PowerShell 5.1 compatibility fault in `Split-Path -LiteralPath ... -Parent` was
also replaced with `.Directory.FullName` before this successful run.

Safety invariants remained intact: there was no automatic retry, no process was
force-terminated, the source template and source archive hashes were unchanged,
the task removed itself only after completion, and no non-infrastructure CST
process remained afterward.
