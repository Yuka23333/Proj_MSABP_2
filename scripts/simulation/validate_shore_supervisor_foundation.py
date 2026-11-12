"""Run local foundation regressions and freeze review evidence, never deploy.

Uses only generated test identities and short-lived test Python processes.
No authority, remote host, USB signing, installation, CST or GPU is contacted.
The public bootstrap is read only to produce a candidate release profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def historical_supplement():
    path = ROOT / "results/reports/shore_remote_live_2026-10-16/acceptance.json"
    before = sha(path)
    original = json.loads(path.read_bytes())
    rows = []
    for old in original["runs"]:
        report = ROOT / "simulations/runs" / old["run_id"] / "completion-report.json"
        if sha(report) != old["completion_report_sha256"]:
            raise ValueError("Historical completion report hash changed")
        value = json.loads(report.read_bytes())
        tasks = [task for snapshot in value["checkpoint"]["snapshots"] for task in snapshot["tasks"]]
        counts = [task["attempt_count"] for task in tasks]
        if not counts or any(type(n) is not int or n != 1 for n in counts):
            raise ValueError("Unexpected historical attempt counts; manual review required")
        rows.append({"run_id": old["run_id"], "original_task_attempt_counts": old["task_attempt_counts"],
            "corrected_task_attempt_counts": counts,
            "completion_report": report.relative_to(ROOT).as_posix(), "completion_report_sha256": sha(report),
            "source_pointer": "/checkpoint/snapshots/*/tasks/*/attempt_count"})
    if sha(path) != before:
        raise ValueError("Historical acceptance changed during review")
    return {"schema": "msabp.historical-attempt-count-supplement", "schema_version": 1,
        "date": "2026-10-17", "original_acceptance": path.relative_to(ROOT).as_posix(),
        "original_acceptance_sha256": before, "original_evidence_overwritten": False,
        "live_state_rechecked": False, "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("foundation", "integration"), default="foundation")
    parser.add_argument("--shore-root", type=Path, default=Path(r"D:\Shore_Protocal"))
    parser.add_argument("--pytest-root", type=Path,
                        default=Path(r"C:\Program Files\Python312\Lib\site-packages"))
    parser.add_argument("--ruff-python", type=Path, default=Path(r"C:\Program Files\Python312\python.exe"))
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    from msabp_opt.simulation.distributed.control_release import build_release_manifest, public_profile

    public_bootstrap = args.shore_root / "artifacts/remote-public/authority-bootstrap.json"
    manifest = build_release_manifest(ROOT, args.shore_root)
    profile = public_profile(manifest, sha(public_bootstrap))
    write_json(output / "candidate-release-manifest.json", manifest)
    write_json(output / "candidate-release-profile.json", profile)
    write_json(output / "historical-attempt-count-supplement.json", historical_supplement())
    tests = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("test_shore*.py"))
    tests.append("tests/test_control_authority.py")
    tests += [f"tests/test_distributed_{name}.py" for name in ("runtime", "state", "transport")]
    new_names = ("winjob", "job_auth", "release", "job_material", "supervisor")
    scoped = [f"src/msabp_opt/simulation/distributed/control_{name}.py" for name in new_names]
    scoped += [f"tests/test_shore_{name}.py" for name in ("winjob", "job_auth", "release", "job_material")]
    scoped += ["tests/test_shore_supervisor_material_journal.py",
               "tests/test_shore_remote_controller.py",
               "tests/test_shore_remote_receipt_lifecycle.py",
               "src/msabp_opt/simulation/distributed/control_remote_http.py",
               "scripts/simulation/validate_shore_supervisor_foundation.py"]
    if args.stage == "integration":
        scoped += [f"src/msabp_opt/simulation/distributed/control_{name}.py"
                   for name in ("job_http", "supervisor_runtime", "supervisor_controller", "remote_prepare",
                                "remote_controller", "remote_worker")]
        scoped += [f"scripts/simulation/{name}.py" for name in (
            "supervisor_host", "shore_supervisor", "shore_supervisor_princess",
            "prepare_shore_supervisor_release")]
        scoped += [f"tests/test_shore_{name}.py" for name in (
            "job_http", "supervisor_runtime", "supervisor_host", "supervisor_controller",
            "supervisor_package", "supervisor_maintenance")]
    old_usb = json.loads((ROOT / "results/reports/shore_usb_2026-10-17/review.json").read_bytes())
    scoped = sorted(set(scoped) | set(old_usb["source_hashes"]))
    freeze = sorted(set(manifest["source_hashes"]) | set(tests) | set(scoped) | {"tests/conftest.py"})
    hashes = {name: sha(ROOT / name) for name in freeze}
    write_json(output / "source-test-hashes.json", hashes)
    test_argv = [sys.executable, "-B", "-c",
        # platform.node() can invoke Windows `ver` on first use. Warm only that
        # local OS metadata before old tests deliberately forbid all processes.
        "import sys,platform; platform.uname(); sys.path.insert(0,sys.argv[1]); sys.path.append(sys.argv[2]); "
        "import pytest; raise SystemExit(pytest.main(sys.argv[3:]))",
        str(args.shore_root.absolute()), str(args.pytest_root.absolute()), *tests,
        "-q", "--disable-warnings", "--junitxml=" + str(output / "tests.xml")]
    lint_argv = [str(args.ruff_python.absolute()), "-m", "ruff", "check", *scoped]
    write_json(output / "commands.json", {"pytest": test_argv, "ruff": lint_argv})
    with (output / "pytest.log").open("x", encoding="utf-8") as log:
        test_result = subprocess.run(test_argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    with (output / "ruff.log").open("x", encoding="utf-8") as log:
        lint_result = subprocess.run(lint_argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=60)
    suites = ET.parse(output / "tests.xml").getroot().findall("testsuite")
    counts = {key: sum(int(s.get(key, "0")) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    drift = [name for name, expected in hashes.items() if sha(ROOT / name) != expected]
    shore_drift = [name for name, expected in manifest["shore_source_hashes"].items()
                   if sha(args.shore_root / name) != expected]
    bootstrap_unchanged = sha(public_bootstrap) == profile["bootstrap_sha256"]
    unchanged_usb = all(sha(ROOT / name) == expected for name, expected in old_usb["source_hashes"].items())
    report = {"schema": "msabp.supervisor-foundation-review", "schema_version": 1,
        "date": "2026-10-17", "stage": args.stage,
        "status": ("LOCAL_FOUNDATION_ONLY_NOT_DEPLOYABLE" if args.stage == "foundation"
                   else "LOCAL_INTEGRATION_REQUIRES_SHORE_INSTALLER_REVIEW"),
        "tests": counts, "pytest_exit_code": test_result.returncode,
        "ruff_exit_code": lint_result.returncode, "ruff_files": len(scoped),
        "source_drift": drift, "previous_usb_patch_unchanged": unchanged_usb,
        "previous_usb_files_changed": [name for name, expected in old_usb["source_hashes"].items()
                                       if sha(ROOT / name) != expected],
        "shore_source_drift": shore_drift, "bootstrap_unchanged": bootstrap_unchanged,
        "candidate_release_id": profile["release_id"],
        "project_release_files": len(manifest["source_hashes"]),
        "shore_pinned_dependencies": len(manifest["shore_source_hashes"]),
        "live_network": False, "real_keys_read": False, "remote_installation": False,
        "loopback_fixture_network": True, "tailnet_or_live_authority_access": False,
        "cst_or_gpu": False, "host_and_runtime_entrypoints_present": args.stage == "integration",
        "system_scheduler_validated": False,
        "next": (["collector job endpoints and pinned-source client integration",
                 "protected policy and host bootstrap plus lifetime instance lock",
                 "fresh discovery and original-job terminal recovery adapter",
                 "durable activation/rollback and missing-ledger/release gates",
                 "full release and test closure review by Shore before installation"]
                 if args.stage == "foundation" else [
                     "Shore review of full source/test/package closure",
                     "Shore-supplied pinned installed interpreter/runtime baseline",
                     "Shore installation/activation/rollback transaction and protected policy",
                     "approved SYSTEM and physical E/F acceptance window"])}
    write_json(output / "review.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(test_result.returncode != 0 or lint_result.returncode != 0 or bool(drift) or bool(shore_drift)
               or not bootstrap_unchanged
               or counts["failures"] + counts["errors"] + counts["skipped"] != 0
               or (args.stage == "foundation" and not unchanged_usb))


if __name__ == "__main__":
    raise SystemExit(main())
