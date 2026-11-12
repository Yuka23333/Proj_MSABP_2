"""Offline, immutable preparation of one or four synthetic-only job inputs.

This freezes public input and reviewed source hashes. It does not acquire a job,
create a binding/session, read device keys, contact a daemon or install anything.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path, PureWindowsPath
import re
import sys

from .control_remote_worker import (
    MODE, PROJECT_SOURCE_FILES, SHORE_SOURCE_FILES, _hash, _json_bytes,
    checked_path, read_json, sha256_file,
)

PREPARATION_SCHEMA = "msabp.shore-remote-synthetic-preparation"
PREPARATION_SOURCE_FILES = tuple(dict.fromkeys((*PROJECT_SOURCE_FILES,
    "scripts/simulation/shore_synthetic_princess.py",
    *("src/msabp_opt/simulation/distributed/" + name + ".py" for name in (
        "control_remote_prepare", "control_remote_controller", "control_remote_session", "control_usb", "control_collector",
        "runtime", "princess", "config", "transport", "bell")))))
_ARTIFACTS = {"samples_csv": "samples.csv", "job_manifest": "job-manifest.json",
              "accounting_policy": "accounting-policy.json", "bootstrap_path": "authority-bootstrap.json"}
_FIELDS = {"schema", "schema_version", "mode", "synthetic", "physical_result", "created_utc",
           "job_id", "run_id", "points", "source_root", "shore_source_root", "source_hashes",
           "shore_source_hashes", "artifact_sha256", "job_spec", "remote_paths"}


def _identifier(value, label):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) is None:
        raise ValueError("Invalid " + label)
    return value


def remote_paths(job_id: str) -> dict[str, str]:
    """Deployment destinations are metadata only, never touched by prepare."""
    job_id = _identifier(job_id, "job ID")
    root = PureWindowsPath(r"C:\ProgramData\ShorePrincess")
    state = root / "state" / "project-msabp" / job_id
    config = root / "project-msabp" / "config" / job_id
    return {
        "source_root": str(root / "project-msabp" / "app"),
        "shore_source_root": str(root / "app"),
        "python_executable": str(root / "venv" / "Scripts" / "python.exe"),
        "worker_config_path": str(root / "state" / "worker" / "worker-config.json"),
        "config_root": str(config),
        "bootstrap_path": str(config / "authority-bootstrap.json"),
        "csv_path": str(config / "samples.csv"),
        "runtime_config_path": str(config / "maid-runtime.json"),
        "bell_config_path": str(config / "bell-config.json"),
        "maid_state_root": str(state / "maid-state"),
        "maid_output_root": str(state / "maid-output"),
        "bell_state_root": str(state / "bell-state"),
    }


def _source_hashes(root, required):
    source = checked_path(root)
    return {relative: sha256_file(source / relative) for relative in required}


def _bootstrap(path, expected, shore_root):
    if sha256_file(path) != _hash(expected, "bootstrap_sha256"):
        raise ValueError("Public bootstrap SHA256 mismatch")
    # Import one fixed API only after inspecting the explicit source manifest.
    if str(shore_root) not in sys.path:
        sys.path.insert(0, str(shore_root))
    from shore_control import remote

    if Path(remote.__file__).resolve().parent.parent != Path(shore_root).resolve():
        raise ValueError("Shore bootstrap reader must come from the pinned source root")
    return remote.load_bootstrap(path)


def _return_record(root, record):
    return {**record, "preparation_path": str(root / "preparation.json"),
            "output_root": str(root), "bootstrap_sha256": record["artifact_sha256"]["authority-bootstrap.json"],
            "paths": {name: str(root / filename) for name, filename in _ARTIFACTS.items()}}


def _write_new(path, data):
    import os

    with checked_path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def prepare_synthetic_job(*, output_root: str | Path, job_id: str, run_id: str,
                          source_root: str | Path, shore_source_root: str | Path,
                          bootstrap_path: str | Path, bootstrap_sha256: str,
                          points: int, release_profile: dict | None = None) -> dict:
    """Create a fresh public-only preparation; any existing target is refused."""
    if type(points) is not int or points not in (1, 4):
        raise ValueError("The synthetic preparation supports explicitly 1 or 4 points")
    job_id, run_id = _identifier(job_id, "job ID"), _identifier(run_id, "run ID")
    root, source, shore, public = map(checked_path,
        (output_root, source_root, shore_source_root, bootstrap_path))
    if root.exists():
        raise FileExistsError("Preparation is immutable; use a new output directory")
    if release_profile is not None:
        from .control_release import build_release_manifest, safe_job_id, validate_profile
        release_profile = validate_profile(release_profile)
        safe_job_id(job_id)
        safe_job_id(run_id)
        if (release_profile["bootstrap_sha256"] != bootstrap_sha256
                or build_release_manifest(source, shore) != release_profile["release_manifest"]):
            raise ValueError("Approved supervisor profile differs from current source/bootstrap")
    required = (PREPARATION_SOURCE_FILES if release_profile is None
                else tuple(release_profile["release_manifest"]["source_hashes"]))
    source_hashes = _source_hashes(source, required)
    shore_hashes = _source_hashes(shore, SHORE_SOURCE_FILES)
    bootstrap = _bootstrap(public, bootstrap_sha256, shore)
    csv_data = ("sample_id,value\n" + "".join(f"{index},{index}\n" for index in range(points))).encode()
    manifest = {"schema": "msabp.shore-remote-synthetic-job", "schema_version": 1,
        "mode": MODE, "synthetic": True, "physical_result": False,
        "job_id": job_id, "run_id": run_id, "approved_units": points,
        "initial_csv_sha256": hashlib.sha256(csv_data).hexdigest(),
        "fleet_id": bootstrap["fleet_id"], "authority_id": bootstrap["authority_id"]}
    if release_profile is not None:
        manifest["supervisor_release_id"] = release_profile["release_id"]
    policy = {"schema": "msabp.budget-policy", "schema_version": 1,
        "unit_key": "evaluation_id", "settled_outcomes": ["completed"]}
    data = {"samples.csv": csv_data, "job-manifest.json": _json_bytes(manifest),
        "accounting-policy.json": _json_bytes(policy), "authority-bootstrap.json": public.read_bytes()}
    hashes = {filename: hashlib.sha256(content).hexdigest() for filename, content in data.items()}
    if hashes["authority-bootstrap.json"] != bootstrap_sha256:
        raise ValueError("Bootstrap changed during preparation")
    spec = {"schema": "shore.job-spec", "schema_version": 1,
        "manifest_sha256": hashes["job-manifest.json"], "approved_units": points,
        "budget_unit": "objective_evaluation", "accounting_policy_sha256": hashes["accounting-policy.json"]}
    record = {"schema": PREPARATION_SCHEMA, "schema_version": 1, "mode": MODE,
        "synthetic": True, "physical_result": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id, "run_id": run_id, "points": points,
        "source_root": str(source), "shore_source_root": str(shore),
        "source_hashes": source_hashes, "shore_source_hashes": shore_hashes,
        "artifact_sha256": hashes, "job_spec": spec, "remote_paths": remote_paths(job_id)}
    if release_profile is not None:
        from .control_release import runtime_paths
        record["release_profile"] = release_profile
        record["remote_paths"] = runtime_paths(release_profile, job_id)
    # Complete all validation before creating anything, and retain failures.
    root.mkdir(parents=True, exist_ok=False)
    for filename, content in data.items():
        _write_new(root / filename, content)
    _write_new(root / ".gitignore", b"*\n")
    if release_profile is not None:
        from .control_job_auth import initialize_job_nonce_db
        initialize_job_nonce_db(root / "job-http-nonces.sqlite")
    _write_new(root / "preparation.json", _json_bytes(record))
    return validate_preparation(root / "preparation.json")


def validate_preparation(path: str | Path) -> dict:
    """Recheck every fixed source/artifact pin; never repair drift or drop pins."""
    path = checked_path(path)
    if path.is_dir():
        path = path / "preparation.json"
    if path.name != "preparation.json":
        raise ValueError("Expected preparation.json")
    record, _ = read_json(path)
    has_profile = "release_profile" in record
    if (set(record) != _FIELDS | ({"release_profile"} if has_profile else set())
            or record["schema"] != PREPARATION_SCHEMA
            or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["mode"] != MODE or record["synthetic"] is not True
            or record["physical_result"] is not False
            or type(record["points"]) is not int or record["points"] not in (1, 4)):
        raise ValueError("Malformed immutable synthetic preparation")
    _identifier(record["job_id"], "job ID")
    _identifier(record["run_id"], "run ID")
    expected_paths = remote_paths(record["job_id"])
    required = PREPARATION_SOURCE_FILES
    if has_profile:
        from .control_release import build_release_manifest, runtime_paths, safe_job_id, validate_profile
        profile = validate_profile(record["release_profile"])
        safe_job_id(record["job_id"])
        safe_job_id(record["run_id"])
        expected_paths = runtime_paths(profile, record["job_id"])
        required = tuple(profile["release_manifest"]["source_hashes"])
        if (build_release_manifest(record["source_root"], record["shore_source_root"])
                != profile["release_manifest"]
                or profile["bootstrap_sha256"] != record["artifact_sha256"]["authority-bootstrap.json"]):
            raise ValueError("Supervisor preparation profile/source/bootstrap changed")
        nonce = checked_path(path.parent / "job-http-nonces.sqlite")
        if not nonce.is_file():
            raise ValueError("Original job HTTP nonce ledger is missing; do not reinitialize")
    if record["remote_paths"] != expected_paths:
        raise ValueError("Preparation redirects fixed deployment paths")
    if type(record["source_hashes"]) is not dict or set(record["source_hashes"]) != set(required):
        raise ValueError("Project source pins differ from the fixed allowlist")
    if type(record["shore_source_hashes"]) is not dict or set(record["shore_source_hashes"]) != set(SHORE_SOURCE_FILES):
        raise ValueError("Shore source pins differ from the fixed allowlist")
    if _source_hashes(record["source_root"], required) != record["source_hashes"]:
        raise ValueError("Frozen project source has drifted")
    if _source_hashes(record["shore_source_root"], SHORE_SOURCE_FILES) != record["shore_source_hashes"]:
        raise ValueError("Frozen Shore source has drifted")
    hashes = record["artifact_sha256"]
    if type(hashes) is not dict or set(hashes) != set(_ARTIFACTS.values()):
        raise ValueError("Public artifacts differ from the fixed allowlist")
    for filename, expected in hashes.items():
        if sha256_file(path.parent / filename) != _hash(expected, "artifact SHA256"):
            raise ValueError("Frozen public input has drifted: " + filename)
    bootstrap = _bootstrap(path.parent / "authority-bootstrap.json", hashes["authority-bootstrap.json"],
                           record["shore_source_root"])
    manifest, _ = read_json(path.parent / "job-manifest.json")
    policy, _ = read_json(path.parent / "accounting-policy.json")
    expected_manifest = {"schema": "msabp.shore-remote-synthetic-job", "schema_version": 1,
        "mode": MODE, "synthetic": True, "physical_result": False,
        "job_id": record["job_id"], "run_id": record["run_id"], "approved_units": record["points"],
        "initial_csv_sha256": hashes["samples.csv"], "fleet_id": bootstrap["fleet_id"],
        "authority_id": bootstrap["authority_id"]}
    if has_profile:
        expected_manifest["supervisor_release_id"] = profile["release_id"]
    expected_policy = {"schema": "msabp.budget-policy", "schema_version": 1,
        "unit_key": "evaluation_id", "settled_outcomes": ["completed"]}
    expected_spec = {"schema": "shore.job-spec", "schema_version": 1,
        "manifest_sha256": hashes["job-manifest.json"], "approved_units": record["points"],
        "budget_unit": "objective_evaluation", "accounting_policy_sha256": hashes["accounting-policy.json"]}
    expected_csv = ("sample_id,value\n" + "".join(f"{index},{index}\n" for index in range(record["points"]))).encode()
    if (_json_bytes(manifest) != _json_bytes(expected_manifest)
            or _json_bytes(policy) != _json_bytes(expected_policy)
            or _json_bytes(record["job_spec"]) != _json_bytes(expected_spec)
            or (path.parent / "samples.csv").read_bytes() != expected_csv):
        raise ValueError("Preparation no longer describes the fixed synthetic workload")
    return _return_record(path.parent, record)
