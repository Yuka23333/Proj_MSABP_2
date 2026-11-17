"""Offline immutable preparation for one Shore-controlled real CST case.

This is deliberately parallel to, rather than an extension of, the synthetic
preparation contract.  It freezes public source/bootstrap bytes, one strict
antenna-tree CSV row and one fixed ``msa-bp.cst`` template.  It never reads a
private key, contacts a service, imports CST/Shapely, or starts a process.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .control_cst_material import (
    CST_MODE,
    TEMPLATE_NAME,
    validate_cst_csv,
    validate_cst_template,
)
from .control_release import (
    RELEASE_SOURCE_FILES,
    build_release_manifest,
    cst_runtime_paths,
    safe_job_id,
    validate_profile,
)
from .control_remote_prepare import _bootstrap, _source_hashes, _write_new
from .control_remote_worker import (
    SHORE_SOURCE_FILES,
    _hash,
    _json_bytes,
    checked_path,
    read_json,
    sha256_file,
)


PREPARATION_SCHEMA = "msabp.shore-remote-cst-preparation"
JOB_SCHEMA = "msabp.shore-remote-cst-job"
_ARTIFACTS = {
    "samples_csv": "samples.csv",
    "job_manifest": "job-manifest.json",
    "accounting_policy": "accounting-policy.json",
    "bootstrap_path": "authority-bootstrap.json",
    "project_template": TEMPLATE_NAME,
}
_FIELDS = {
    "schema", "schema_version", "mode", "synthetic", "physical_result",
    "created_utc", "job_id", "run_id", "points", "source_root",
    "shore_source_root", "source_hashes", "shore_source_hashes",
    "artifact_sha256", "job_spec", "remote_paths", "release_profile",
}

def _return_record(root: Path, record: dict) -> dict:
    return {
        **record,
        "preparation_path": str(root / "preparation.json"),
        "output_root": str(root),
        "bootstrap_sha256": record["artifact_sha256"]["authority-bootstrap.json"],
        "paths": {name: str(root / filename) for name, filename in _ARTIFACTS.items()},
    }


def prepare_cst_job(
    *,
    output_root: str | Path,
    job_id: str,
    run_id: str,
    source_root: str | Path,
    shore_source_root: str | Path,
    bootstrap_path: str | Path,
    bootstrap_sha256: str,
    csv_path: str | Path,
    project_path: str | Path,
    release_profile: dict,
) -> dict:
    """Freeze one real case; any existing output directory is refused."""
    job_id, run_id = safe_job_id(job_id), safe_job_id(run_id)
    root, source, shore, public, csv_source, project_source = map(
        checked_path,
        (
            output_root,
            source_root,
            shore_source_root,
            bootstrap_path,
            csv_path,
            project_path,
        ),
    )
    if root.exists():
        raise FileExistsError("CST preparation is immutable; use a new output directory")
    if not csv_source.is_file():
        raise FileNotFoundError("CST samples CSV is missing")
    if not project_source.is_file() or project_source.suffix.lower() != ".cst":
        raise FileNotFoundError("CST project template is missing")
    profile = validate_profile(release_profile)
    if (
        profile["bootstrap_sha256"] != _hash(bootstrap_sha256, "bootstrap_sha256")
        or build_release_manifest(source, shore) != profile["release_manifest"]
    ):
        raise ValueError("Approved supervisor profile differs from current source/bootstrap")

    csv_data = csv_source.read_bytes()
    project_data = project_source.read_bytes()
    validate_cst_csv(csv_data)
    project_sha256 = validate_cst_template(project_data)
    source_hashes = _source_hashes(source, RELEASE_SOURCE_FILES)
    shore_hashes = _source_hashes(shore, SHORE_SOURCE_FILES)
    bootstrap = _bootstrap(public, bootstrap_sha256, shore)
    csv_sha256 = hashlib.sha256(csv_data).hexdigest()
    manifest = {
        "schema": JOB_SCHEMA,
        "schema_version": 1,
        "mode": CST_MODE,
        "synthetic": False,
        "physical_result": True,
        "job_id": job_id,
        "run_id": run_id,
        "approved_units": 1,
        "initial_csv_sha256": csv_sha256,
        "project_template_sha256": project_sha256,
        "fleet_id": bootstrap["fleet_id"],
        "authority_id": bootstrap["authority_id"],
        "supervisor_release_id": profile["release_id"],
    }
    policy = {
        "schema": "msabp.budget-policy",
        "schema_version": 1,
        "unit_key": "evaluation_id",
        "settled_outcomes": ["completed"],
    }
    data = {
        "samples.csv": csv_data,
        "job-manifest.json": _json_bytes(manifest),
        "accounting-policy.json": _json_bytes(policy),
        "authority-bootstrap.json": public.read_bytes(),
        TEMPLATE_NAME: project_data,
    }
    hashes = {
        filename: hashlib.sha256(content).hexdigest()
        for filename, content in data.items()
    }
    if hashes["authority-bootstrap.json"] != bootstrap_sha256:
        raise ValueError("Bootstrap changed during CST preparation")
    spec = {
        "schema": "shore.job-spec",
        "schema_version": 1,
        "manifest_sha256": hashes["job-manifest.json"],
        "approved_units": 1,
        "budget_unit": "objective_evaluation",
        "accounting_policy_sha256": hashes["accounting-policy.json"],
    }
    record = {
        "schema": PREPARATION_SCHEMA,
        "schema_version": 1,
        "mode": CST_MODE,
        "synthetic": False,
        "physical_result": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "run_id": run_id,
        "points": 1,
        "source_root": str(source),
        "shore_source_root": str(shore),
        "source_hashes": source_hashes,
        "shore_source_hashes": shore_hashes,
        "artifact_sha256": hashes,
        "job_spec": spec,
        "remote_paths": cst_runtime_paths(profile, job_id),
        "release_profile": profile,
    }

    root.mkdir(parents=True, exist_ok=False)
    for filename, content in data.items():
        _write_new(root / filename, content)
    _write_new(root / ".gitignore", b"*\n")
    from .control_job_auth import initialize_job_nonce_db

    initialize_job_nonce_db(root / "job-http-nonces.sqlite")
    _write_new(root / "preparation.json", _json_bytes(record))
    return validate_cst_preparation(root / "preparation.json")


def validate_cst_preparation(path: str | Path) -> dict:
    """Recheck every source and input byte; never repair or regenerate state."""
    path = checked_path(path)
    if path.is_dir():
        path = path / "preparation.json"
    if path.name != "preparation.json":
        raise ValueError("Expected preparation.json")
    record, _ = read_json(path)
    if (
        set(record) != _FIELDS
        or record["schema"] != PREPARATION_SCHEMA
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["mode"] != CST_MODE
        or record["synthetic"] is not False
        or record["physical_result"] is not True
        or record["points"] != 1
        or type(record["created_utc"]) is not str
        or not record["created_utc"]
    ):
        raise ValueError("Malformed immutable CST preparation")
    job_id, run_id = safe_job_id(record["job_id"]), safe_job_id(record["run_id"])
    profile = validate_profile(record["release_profile"])
    expected_paths = cst_runtime_paths(profile, job_id)
    if record["remote_paths"] != expected_paths:
        raise ValueError("CST preparation redirects fixed deployment paths")
    if (
        build_release_manifest(record["source_root"], record["shore_source_root"])
        != profile["release_manifest"]
        or profile["bootstrap_sha256"]
        != record["artifact_sha256"].get("authority-bootstrap.json")
    ):
        raise ValueError("Supervisor CST preparation profile/source/bootstrap changed")
    if (
        type(record["source_hashes"]) is not dict
        or set(record["source_hashes"]) != set(RELEASE_SOURCE_FILES)
        or _source_hashes(record["source_root"], RELEASE_SOURCE_FILES)
        != record["source_hashes"]
    ):
        raise ValueError("Frozen CST project source has drifted")
    if (
        type(record["shore_source_hashes"]) is not dict
        or set(record["shore_source_hashes"]) != set(SHORE_SOURCE_FILES)
        or _source_hashes(record["shore_source_root"], SHORE_SOURCE_FILES)
        != record["shore_source_hashes"]
    ):
        raise ValueError("Frozen Shore source has drifted")
    hashes = record["artifact_sha256"]
    if type(hashes) is not dict or set(hashes) != set(_ARTIFACTS.values()):
        raise ValueError("CST public artifacts differ from the fixed allowlist")
    for filename, expected in hashes.items():
        if sha256_file(path.parent / filename) != _hash(expected, "artifact SHA256"):
            raise ValueError("Frozen CST public input has drifted: " + filename)
    nonce = checked_path(path.parent / "job-http-nonces.sqlite")
    if not nonce.is_file():
        raise ValueError("Original CST job HTTP nonce ledger is missing")
    bootstrap = _bootstrap(
        path.parent / "authority-bootstrap.json",
        hashes["authority-bootstrap.json"],
        record["shore_source_root"],
    )
    manifest, _ = read_json(path.parent / "job-manifest.json")
    policy, _ = read_json(path.parent / "accounting-policy.json")
    expected_manifest = {
        "schema": JOB_SCHEMA,
        "schema_version": 1,
        "mode": CST_MODE,
        "synthetic": False,
        "physical_result": True,
        "job_id": job_id,
        "run_id": run_id,
        "approved_units": 1,
        "initial_csv_sha256": hashes["samples.csv"],
        "project_template_sha256": hashes[TEMPLATE_NAME],
        "fleet_id": bootstrap["fleet_id"],
        "authority_id": bootstrap["authority_id"],
        "supervisor_release_id": profile["release_id"],
    }
    expected_policy = {
        "schema": "msabp.budget-policy",
        "schema_version": 1,
        "unit_key": "evaluation_id",
        "settled_outcomes": ["completed"],
    }
    expected_spec = {
        "schema": "shore.job-spec",
        "schema_version": 1,
        "manifest_sha256": hashes["job-manifest.json"],
        "approved_units": 1,
        "budget_unit": "objective_evaluation",
        "accounting_policy_sha256": hashes["accounting-policy.json"],
    }
    csv_data = (path.parent / "samples.csv").read_bytes()
    project_data = (path.parent / TEMPLATE_NAME).read_bytes()
    validate_cst_csv(csv_data)
    if validate_cst_template(project_data) != hashes[TEMPLATE_NAME]:
        raise ValueError("CST template digest changed")
    if (
        _json_bytes(manifest) != _json_bytes(expected_manifest)
        or _json_bytes(policy) != _json_bytes(expected_policy)
        or _json_bytes(record["job_spec"]) != _json_bytes(expected_spec)
    ):
        raise ValueError("Preparation no longer describes the fixed real CST workload")
    return _return_record(path.parent, record)
