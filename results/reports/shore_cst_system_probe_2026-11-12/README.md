# CoconutG2 SYSTEM CST open/close probe

This directory archives the first successful CST integration probe through the
Shore controller path.  The probe opened an isolated copy of `msa-bp.cst` as
`NT AUTHORITY\SYSTEM` in session 0 and closed both the project and the CST
Design Environment.  It did not start a solver.

- Request ID: `3c67b8ea5a0c4296b7fcfbef33c03bdd`
- Host: `CoconutG2`
- Python: `C:\Users\telecom\miniforge3\envs\maid\python.exe` (3.11.15)
- Task result: `0`; the one-shot task was removed after completion
- Probe elapsed time: 12.234 s
- Source/template SHA-256 before and after:
  `d1962a2d16f976b7d1e4e9eb6d44baf1a460ae324897b4b2c892c0109715d3a8`
- Remote evidence root:
  `C:\ProgramData\ShorePrincess\project-msabp\diagnostics\cst-system-open-3c67b8ea5a0c4296b7fcfbef33c03bdd`

Archived evidence hashes:

- `invocation.json`: `c8483414df749622f910a27e7102a59a6d810ba806587db558ed28c03fc79e9a`
- `report.json`: `3fe9ede41f2b5d6d99421ebd1dfdf6bd82d40aced734a52f0fdee8f3298105da`

The read-only preflight initially rejected three boot-time CST-named processes.
They were subsequently proven to be the two automatic CST Distributed
Computing services and the CST license daemon.  The final guard allows only
those exact executable names at their exact CST 2025 installation paths; all
other CST/design-environment/solver matches remain blockers.

The next integration boundary is a separately approved minimal real solve.
