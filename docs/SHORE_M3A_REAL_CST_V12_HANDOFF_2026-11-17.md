# Shore M3a real-CST v12 handoff

Date: 2026-11-17

Status: **offline candidate only**. Nothing in this handoff was installed,
activated, sent to CoconutG2, or allowed to start CST.

## Outcome

The existing Supervisor release can now carry either of two closed workloads:

- `shore-remote-synthetic-v1`: the unchanged synthetic acceptance workload;
- `shore-remote-cst-v1`: one physical `antenna_tree` row against the fixed
  `msa-bp.cst` template.

The Bell transport, wake/startup/idle signed domains, release-profile outer
mode, and installed service lifecycle remain `shore-remote-synthetic-v1`.
Only the hash-bound Maid runtime selects real CST. This is deliberate: v12 must
not invent a second Bell protocol or change the already reviewed channel.

## Candidate package

Root:

`D:\Academic\Proj_MSABP_2\results\reports\shore_supervisor_m3a_2026-11-17\candidate-v12`

| Item | SHA-256 |
|---|---|
| Release ID / `release-manifest.json` | `e1163290ef46570896898280233f23040f7e4bfaad3cbdbc808c2728cf499c26` |
| `public-release-profile.json` | `98b867e6bc95e1af0b44260098906486e344edc8762d5988b8f41d3ae0f362e9` |
| `candidate-package.json` | `dce6a9880148785ecd30db83cf4b6b18cdc3d5a1455bd67c6917f6a5af751c0d` |
| `supervisor_host.py` | `9009c89b40b7a3ad492630beaa9652c41735677f5096d1d46282b002a01134eb` |
| Public bootstrap | `3bd90aef2e61cc3eea66eb6d36e280d1d2b15e86adceb562c06685e48ef21195` |

The candidate declares 57 project files and 14 Shore dependencies. The
packager re-read the source, host and bootstrap and ran `verify_release`; the
candidate record remains `REQUIRES_SHORE_INSTALLER_REVIEW`, with
`installed=false` and `activated=false`.

## Closed M3a contract

- Exactly one CSV row, with exact header
  `sample_id,simulation_mode,tree_json`.
- `simulation_mode` is exactly `antenna_tree`.
- Template filename is exactly `msa-bp.cst`; maximum 512 KiB; bytes and hash
  are part of the signed material.
- Material commits exactly five files: public bootstrap, CSV, template, Maid
  runtime and Bell config.
- Runtime fixes coordinate quantum to `0.01` mm, disconnected conductor to
  `false`, command timeout to 15 s, poll interval to 2 s and heartbeat to 5 s.
- Maid parent remains stdlib-only. One fixed child process owns and closes all
  CST objects for one case. It receives no executable or source path from the
  CSV row.
- A successful result is the exact six-file closure:
  `manifest.json`, `S11.csv`, `Rad_Eff.csv`, `Tot_Eff.csv`,
  `Farfield Source [1].ffs`, and `geometry_tree.json`.
- JSON/control traffic remains capped at 1 MiB. Artifact upload and expanded
  ZIP each have an independent 256 MiB cap.
- `max_attempts=1`; a returned solver error is a confirmed physical failure;
  process loss after STARTING remains unresolved and is never retried or
  relabelled not-started.
- Terminal provenance is bound to the selected workload mode. Synthetic and
  physical terminal records cannot be swapped.

## Verification

The focused CST/material/prepare/HTTP, runtime selector, five-file journal,
prepare-to-bind, worker-to-mock-child, and full worker-to-Princess collector
tests pass without opening CST:

- complete Shore selection: **1027 passed, 977 deselected**;
- explicit M3a plus release-closure selection: **172 passed**.

The full worker-to-Princess test uses a loopback collector and a mock CST child;
it verifies the physical six-file result, database settlement and both signed
evaluation receipts without contacting an external host or importing CST.

The prior one-shot SYSTEM/session-0 physical probe remains separate evidence:
request `50fffd9d668640f08b7b5cdce49580e9`, 77.863 s, five exported artifacts,
template hash unchanged, and no residual CST process. M3a does not reinterpret
that probe as a Supervisor deployment.

## Required v12 Shore work

1. Review this release closure and build a **new sibling v12 bridge package**.
   Do not reuse an old transaction, staging directory, release ID or rejected
   FirstInstall command.
2. Keep the existing synthetic Bell channel/schema/mode and signed domains.
   The installed Supervisor must choose the CST behavior only from the verified
   Maid material/runtime.
3. Bind or attest the worker-local solver baseline at
   `C:\Users\telecom\miniforge3\envs\maid\python.exe`. The project currently
   fixes the path and validates the Conda directory shape, but the v12
   installer/policy must decide how to pin the interpreter and required runtime
   files/packages. No executable path may come from job material.
4. Preserve the exact five-file material commit and the separate 1 MiB versus
   256 MiB transport limits.
5. Produce offline build/test/hash evidence first. Do not install or run the
   candidate until the user announces a real-control window and confirms the
   remote security software state.

## Explicit non-goals

- no multi-point DoE;
- no alternate CST model or simulation mode;
- no retry after unknown external start;
- no new Bell protocol;
- no automatic remote installation, task replacement or CST launch in this
  checkpoint.
