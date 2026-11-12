"""Prepare, bind or serve one explicitly authorized Shore synthetic job.

Defaults only print help. No SSH, service installation, CST or GPU is available.
Real bind/serve requires a separate approved firewall/deployment test window.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
BOOTSTRAP = r"D:\Shore_Protocal\artifacts\remote-public\authority-bootstrap.json"
BOOTSTRAP_SHA256 = "3bd90aef2e61cc3eea66eb6d36e280d1d2b15e86adceb562c06685e48ef21195"


def main(argv=None, *, supervisor=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shore-root", default=r"D:\Shore_Protocal")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Offline public input/source pinning only")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--job-id", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--points", type=int, choices=(1, 4), default=1)
    prepare.add_argument("--bootstrap", default=BOOTSTRAP)
    prepare.add_argument("--bootstrap-sha256", default=BOOTSTRAP_SHA256)
    if supervisor:
        prepare.add_argument("--release-profile", required=True,
                             help="Approved installed supervisor public profile; never uploads software")
    preflight = commands.add_parser("preflight", help="Offline: check every frozen source/input")
    preflight.add_argument("--preparation", required=True)
    for command in ("usb-check", "status"):
        sub = commands.add_parser(command, help=("Public-only USB check" if command == "usb-check"
                                  else "LIVE: signed read-only authority status; no job or session"))
        sub.add_argument("--bootstrap", default=BOOTSTRAP)
        sub.add_argument("--bootstrap-sha256", default=BOOTSTRAP_SHA256)
        sub.add_argument("--usb-manifest", default="auto")
        if command == "status":
            sub.add_argument("--live-approved", action="store_true")
    for command in ("bind", "serve", "close", "serve-receipts"):
        sub = commands.add_parser(command, help="LIVE: " + command + " the fixed synthetic job")
        sub.add_argument("--preparation", required=True)
        sub.add_argument("--live-approved", action="store_true",
                         help="Explicit post-review test window, including Huorong allowance")
        sub.add_argument("--usb-manifest", default="auto",
                         help="Discover enrolled removable USB, then lock path/volume; or specify its manifest")
        if command in {"bind", "serve", "serve-receipts"}:
            sub.add_argument("--collector-config", default=str(Path(os.environ.get("LOCALAPPDATA", ""))
                             / "ShorePrincess" / "collector" / "collector-local.json"))
        if command == "serve-receipts":
            sub.add_argument("--seconds", type=int, default=300,
                             help="Receipt-only window in [1,3600] seconds; no new work or renewal")
        if command == "close":
            sub.add_argument("--abandon-remaining-budget", action="store_true")
            sub.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    if args.command in {"bind", "serve", "close", "serve-receipts", "status"} and not args.live_approved:
        parser.error("Real control requires prior review + user-approved firewall window + --live-approved")
    if args.command == "close" and (not args.abandon_remaining_budget or not args.reason.strip()):
        parser.error("Close requires --abandon-remaining-budget and a nonempty --reason")
    if args.command == "serve-receipts" and not 1 <= args.seconds <= 3600:
        parser.error("Receipt window must be in [1,3600] seconds")
    from msabp_opt.simulation.distributed.control_remote_worker import MODE, checked_path, instance_lock

    previous_mode = os.environ.get("MSABP_CONTROL_MODE", "")
    if previous_mode and previous_mode != MODE:
        raise ValueError("Refusing to switch an existing controlled process into remote mode")
    os.environ["MSABP_CONTROL_MODE"] = MODE
    shore = checked_path(args.shore_root)
    sys.path.insert(0, str(shore))
    from msabp_opt.simulation.distributed.control_remote_prepare import prepare_synthetic_job, validate_preparation

    if args.command in {"usb-check", "status"}:
        from shore_control.crypto import fingerprint
        from msabp_opt.simulation.distributed.control_remote_prepare import _bootstrap
        from msabp_opt.simulation.distributed.control_remote_controller import CONTROLLER_ACTOR, HOST_NODE
        from msabp_opt.simulation.distributed.control_usb import LockedUsbIdentity, controller_client

        bootstrap = _bootstrap(checked_path(args.bootstrap), args.bootstrap_sha256, shore)
        signer = LockedUsbIdentity(actor=CONTROLLER_ACTOR,
            expected_fingerprint=fingerprint(bootstrap["controllers"][CONTROLLER_ACTOR]),
            selection=args.usb_manifest)
        public = signer.inspect()
        result = {"mode": MODE, "usb": public, "private_keys_read": False, "network": False}
        if args.command == "status":
            client = controller_client(bootstrap, actor=CONTROLLER_ACTOR,
                                       local_node_id=HOST_NODE, signer=signer)
            result.update(status=client.call("status"), private_keys_read=True, network=True)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "prepare":
        profile_options = {}
        if supervisor:
            from msabp_opt.simulation.distributed.control_remote_worker import read_json
            profile_options["release_profile"] = read_json(checked_path(args.release_profile))[0]
        result = prepare_synthetic_job(output_root=checked_path(args.output), job_id=args.job_id,
            run_id=args.run_id, source_root=ROOT, shore_source_root=shore,
            bootstrap_path=checked_path(args.bootstrap), bootstrap_sha256=args.bootstrap_sha256,
            points=args.points, **profile_options)
    else:
        result = validate_preparation(checked_path(args.preparation))
    if Path(result["source_root"]) != ROOT:
        raise ValueError("Execute the entrypoint from the pinned project source root")
    if ("release_profile" in result) != supervisor:
        raise ValueError("Use the matching legacy or supervisor Princess entrypoint for this preparation")
    if args.command in {"prepare", "preflight"}:
        print(json.dumps({"mode": MODE, "offline": True, "preparation": result["preparation_path"],
                          "points": result["points"], "network": False, "keys_read": False}, indent=2))
        return 0
    from msabp_opt.simulation.distributed.control_remote_controller import (
        bind_job, close_job, open_clients, serve_job, serve_receipts,
    )

    if args.command in {"serve", "close", "serve-receipts"} and any(not (Path(result["output_root"]) / name).is_file()
            for name in ("control.sqlite", "session.dpapi.json", "deployment/maid-runtime.json")):
        raise ValueError("Serve requires the original completed bind artifacts; it never creates a new job")
    with instance_lock(Path(result["output_root"]) / "controller-lock"):
        gate, verifier, session_id, public, bootstrap = open_clients(result, usb_manifest=args.usb_manifest)
        if args.command == "bind":
            bind_job(result, gate, bootstrap, session_id, public)
            print(json.dumps({"phase": gate.job["phase"], "job_id": result["job_id"],
                "deployment": str(Path(result["output_root"]) / "deployment"),
                "wake_issued": False, "remote_installed": False}, indent=2))
            # Never return an ACTIVE lease to an unattended deployment gap.
            # Keep this original process renewing while the operator installs
            # the already staged public files. Ctrl-C drains, never undrains.
            print(("Bind stays running: F authorizes work; installed supervisor discovers it."
                   if supervisor else "Bind stays running: renewing + serving during manual deployment."), flush=True)
        if args.command == "close":
            close_job(result, gate, reason=args.reason)
        elif args.command == "serve-receipts":
            serve_receipts(result, gate, bootstrap, verifier,
                           collector_config=args.collector_config, seconds=args.seconds)
        else:
            report = serve_job(result, gate, bootstrap, verifier, collector_config=args.collector_config)
            if supervisor and report["phase"] == "RELEASED":
                # The remote supervisor must collect the original signed release
                # after both actors exit. A later lost response can use the same
                # preparation's explicit serve-receipts entry, without new work.
                serve_receipts(result, gate, bootstrap, verifier,
                               collector_config=args.collector_config, seconds=30)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        # Never dump private material or signed envelopes to a console log.
        print(f"Remote synthetic error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
