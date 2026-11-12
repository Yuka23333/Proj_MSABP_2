"""Fixed, bounded synthetic artifacts. These are never electromagnetic results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

MAX_SYNTHETIC_ATTEMPT_BYTES = 1024 * 1024 - 1


class SyntheticSession:
    """No solver, process, accelerator, project or external resource exists."""

    def open(self):
        return None

    def close(self):
        pass


def synthetic_case(_row, **kwargs):
    """Generate only an allowlisted, constant-sized file set for one case."""
    from .case_runner import CaseRunResult
    from .control_remote_worker import checked_path

    case_id = kwargs["case_id"]
    if (type(case_id) is not str or len(case_id) > 128
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) is None):
        raise ValueError("Synthetic case requires a bounded safe case ID")
    directory = checked_path(Path(kwargs["output_root"])) / ("case_" + case_id)
    directory.mkdir(parents=True, exist_ok=False)
    checked_path(directory)
    artifacts = {}
    for name, filename in (("s11", "S11.csv"), ("rad_eff", "Rad_Eff.csv"),
                           ("tot_eff", "Tot_Eff.csv"),
                           ("farfield_source", "Farfield Source [1].ffs")):
        # Deliberately not parseable as physical numeric measurements.
        data = ("SYNTHETIC ONLY - NOT A PHYSICAL RESULT\n" + case_id + " " + name + "\n").encode()
        (directory / filename).write_bytes(data)
        artifacts[name] = {"path": filename, "size_bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "status": "completed", "dry_run": False,
        "case_id": case_id, "simulation_mode": "antenna_characterization",
        "synthetic": True, "physical_result": False,
        "generator": "msabp.shore-remote-synthetic-v1", "artifacts": artifacts,
    }, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    total = sum(path.stat().st_size for path in directory.iterdir())
    if total > MAX_SYNTHETIC_ATTEMPT_BYTES:
        raise ValueError("Synthetic attempt exceeds the strict 1 MiB bound")
    return CaseRunResult(case_id=case_id, case_directory=directory,
        manifest_path=manifest_path, s11_path=directory / "S11.csv",
        rad_eff_path=directory / "Rad_Eff.csv", tot_eff_path=directory / "Tot_Eff.csv",
        farfield_source_path=directory / "Farfield Source [1].ffs",
        dry_run=False, elapsed_seconds=0)
