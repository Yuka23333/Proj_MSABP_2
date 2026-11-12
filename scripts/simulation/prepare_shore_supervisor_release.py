"""Build an offline candidate package for Shore review, never install/activate.

Copies only the fixed project closure and standalone host, not identities,
dependencies, old task state or private keys. The installed interpreter trust
baseline and installation/activation transaction remain Shore-owned inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def build_package(output, *, source_root, shore_root, bootstrap_path, bootstrap_sha256):
    from msabp_opt.simulation.distributed.control_release import (
        _regular_hash, build_release_manifest, public_profile, verify_release,
    )
    from msabp_opt.simulation.distributed.control_remote_worker import _json_bytes, checked_path

    output, source, shore, bootstrap = map(checked_path,
        (output, source_root, shore_root, bootstrap_path))
    if output.exists():
        raise FileExistsError("Candidate package must use a new directory")
    if _regular_hash(bootstrap) != bootstrap_sha256:
        raise ValueError("Public bootstrap pin mismatch")
    manifest = build_release_manifest(source, shore)
    profile = public_profile(manifest, bootstrap_sha256)
    host = source / "scripts/simulation/supervisor_host.py"
    host_hash = _regular_hash(host)
    # No installation paths are writable through this command. A caller must
    # choose an explicit workspace/scratch output, not the fixed service root.
    forbidden = Path(r"C:\ProgramData\ShorePrincess")
    if output == forbidden or forbidden in output.parents:
        raise ValueError("Packaging cannot write to the installed service tree")
    output.mkdir(parents=True, exist_ok=False)
    release = output / "releases" / profile["release_id"]
    app = release / "app"
    for relative in manifest["source_hashes"]:
        target = app / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    (output / "supervisor").mkdir()
    shutil.copyfile(host, output / "supervisor/supervisor_host.py")
    (release / "release-manifest.json").write_bytes(_json_bytes(manifest))
    (output / "public-release-profile.json").write_bytes(_json_bytes(profile))
    verify_release(profile, app, shore)
    if (build_release_manifest(source, shore) != manifest or _regular_hash(host) != host_hash
            or _regular_hash(output / "supervisor/supervisor_host.py") != host_hash
            or _regular_hash(bootstrap) != bootstrap_sha256):
        raise ValueError("Source/host/bootstrap changed during candidate staging")
    record = {"schema": "msabp.supervisor-candidate-package", "schema_version": 1,
        "status": "REQUIRES_SHORE_INSTALLER_REVIEW", "installed": False, "activated": False,
        "release_id": profile["release_id"], "host_sha256": host_hash,
        "public_profile_sha256": hashlib.sha256(_json_bytes(profile)).hexdigest(),
        "bootstrap_sha256": bootstrap_sha256,
        "project_files": len(manifest["source_hashes"]),
        "shore_dependencies": len(manifest["shore_source_hashes"]),
        "missing_deployment_inputs": ["reviewed installed interpreter/runtime baseline",
            "protected policy and committed active-release transaction",
            "explicit first-install ledger initialization and SYSTEM task installer",
            "real SYSTEM/nested-job and physical E/F acceptance window"]}
    (output / "candidate-package.json").write_bytes(_json_bytes(record))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shore-root", type=Path, default=Path(r"D:\Shore_Protocal"))
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--bootstrap-sha256", required=True)
    args = parser.parse_args(argv)
    result = build_package(args.output.absolute(), source_root=ROOT,
        shore_root=args.shore_root.absolute(), bootstrap_path=args.bootstrap.absolute(),
        bootstrap_sha256=args.bootstrap_sha256)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
