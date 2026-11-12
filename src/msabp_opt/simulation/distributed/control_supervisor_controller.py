"""Opt-in Princess bridge to the installed supervisor; never installs software.

Only the controller loop may sign a new wake. HTTP reads below can replay
already-persisted assignments and original signed release evidence, including
during drain or receipt-only recovery, without consulting F or an authority.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket

from .control_remote_worker import LOCAL_NODE_ID, read_json


def knock(bootstrap, *, port=None):
    """Tell the installed supervisor to look for work now, instead of later.

    One empty datagram to the worker's Tailnet address. It is a hint and grants
    nothing: the supervisor still asks the authority for a fresh signed status
    and verifies everything as before. A datagram may be lost and the worker may
    not be listening, so a failure here is not an error; the supervisor's own
    fallback timer finds the job anyway, only later.
    """
    try:
        if port is None:
            from .control_supervisor_runtime import KNOCK_PORT as port
        address = bootstrap["network"]["nodes"][LOCAL_NODE_ID]["address"]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(b"\0", (address, port))
        return True
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        return False


def make_job_service(preparation, gate, queue, bootstrap):
    """Create read-only job data service from one original bound preparation."""
    from .control_job_http import JobMaterialService

    if "release_profile" not in preparation:
        return None
    root = Path(preparation["output_root"])

    def read_assignment():
        path = queue.root / "assignment.json"
        # A missing original assignment is not an instruction to issue one.
        return read_json(path)[0] if path.is_file() else None

    def read_release_proof():
        # No call to released_job()/rpc(): responses must already be durable.
        if gate.job["phase"] != "RELEASED":
            return None
        with gate.journal.transaction() as db:
            rows = db.execute(
                "SELECT key,params FROM rpc_journal WHERE method='release_job' AND identity=? ORDER BY rowid DESC",
                (gate.journal._identity(gate.controller),)).fetchall()
        for row in rows:
            params = json.loads(row["params"])
            if all(params.get(k) == v for k, v in gate.binding.items()):
                proof = gate.journal.proof(row["key"])
                if proof is not None:
                    return proof
        raise ValueError("Released job lacks its original signed release proof")

    return JobMaterialService(
        profile=preparation["release_profile"],
        bootstrap_data=Path(preparation["paths"]["bootstrap_path"]).read_bytes(),
        expected_binding=dict(queue.runtime["binding"]),
        read_assignment=read_assignment,
        csv_data=Path(preparation["paths"]["samples_csv"]).read_bytes(),
        nonce_db=root / "job-http-nonces.sqlite",
        read_release_proof=read_release_proof,
    )


def publish_original_wake(gate, queue):
    """Run in the F-backed control loop, never in the download HTTP handler.

    Missing F/lease/ACTIVE fails the existing require_new_work/validate_dispatch
    gate. Restarting uses the same operation and signed request, not a new wake.
    """
    if gate.job["phase"] == "ACTIVE":
        queue._assignment()
