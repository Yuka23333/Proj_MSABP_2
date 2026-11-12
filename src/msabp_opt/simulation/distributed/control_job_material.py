"""Bounded synthetic job data, separate from software installation and launch.

The caller supplies the approved LOCAL profile/bootstrap and an independently
verified job binding. Validation never accepts a replacement trust root from
the wire, issues a permit, accepts work, or starts a process. In particular, an
old valid wake is useful during drain but is not fresh authority to execute it.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from .control_release import runtime_paths, safe_job_id, validate_profile
from .control_remote_worker import (
    BELL_ACTOR, BINDING_FIELDS, DEVICE_ID, LOCAL_NODE_ID, MAID_ACTOR, MODE,
    PROJECT_SOURCE_FILES, SCHEMA, RemoteSyntheticMaidConfig, _json_bytes, _unique,
)

MATERIAL_SCHEMA = "msabp.supervisor-job-material"
MAX_RUNTIME_BYTES = 64 * 1024
MAX_CSV_BYTES = 256 * 1024
MAX_MATERIAL_BYTES = 1024 * 1024


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_object(raw: bytes, limit: int) -> dict:
    if type(raw) is not bytes or not raw or len(raw) > limit:
        raise ValueError("Missing or oversized canonical material")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
    if type(value) is not dict or _json_bytes(value) != raw:
        raise ValueError("Exact canonical JSON object required")
    return value


def synthetic_csv(points: int) -> bytes:
    if type(points) is not int or points not in (1, 4):
        raise ValueError("Only 1 or 4 synthetic points are supported")
    return ("sample_id,value\n" + "".join(f"{i},{i}\n" for i in range(points))).encode()


def _binding(value: Mapping, bootstrap: dict) -> dict:
    from shore_control.job_contract import validate_job_spec

    if type(value) is not dict or set(value) != BINDING_FIELDS:
        raise ValueError("Complete independently verified job binding required")
    value = json.loads(_json_bytes(value))
    safe_job_id(value["job_id"])
    if type(value["epoch"]) is not int or value["epoch"] < 1:
        raise ValueError("Invalid job epoch")
    if any(type(value[k]) is not str or not value[k]
           for k in BINDING_FIELDS - {"epoch", "job_spec"}):
        raise ValueError("Invalid job scope")
    if (value["fleet_id"] != bootstrap["fleet_id"] or value["project_id"] != "MSABP_2"
            or value["host_id"] != bootstrap["network"]["local_node_id"]
            or value["collector_url"] != bootstrap["collector"]["endpoint"]):
        raise ValueError("Job is outside the locally pinned fleet/project/collector")
    validate_job_spec(value["job_spec"])
    synthetic_csv(value["job_spec"]["approved_units"])
    return value


def _local_bootstrap(profile: dict, bootstrap_data: bytes) -> dict:
    from shore_control.remote import _validate_bootstrap

    if _digest(bootstrap_data) != profile["bootstrap_sha256"]:
        raise ValueError("Local bootstrap differs from the approved profile")
    # Existing bootstrap files may have whitespace. They are pinned byte-wise,
    # not rewritten; downloaded runtime and material use canonical JSON.
    if len(bootstrap_data) > MAX_RUNTIME_BYTES:
        raise ValueError("Oversized local bootstrap")
    value = json.loads(bootstrap_data.decode("utf-8"), object_pairs_hook=_unique)
    bootstrap = _validate_bootstrap(value)
    if set(bootstrap["workers"]) != {BELL_ACTOR, MAID_ACTOR}:
        raise ValueError("Bootstrap requires the fixed Bell and Maid registration pair")
    for actor, role, action in ((BELL_ACTOR, "bell", "wake"), (MAID_ACTOR, "maid", "attempt")):
        row = bootstrap["workers"][actor]
        if (row["device_id"] != DEVICE_ID or row["tailnet_node_id"] != LOCAL_NODE_ID
                or row["role"] != role or row["allowed_actions"] != [action]):
            raise ValueError("Bootstrap differs from the fixed device-local actor registration")
    return bootstrap


def build_runtime(profile: dict, bootstrap_data: bytes, binding: dict, *,
                  run_id: str, api_token: str) -> dict:
    """Princess constructs final remote paths BEFORE signing the wake digest.

    No filesystem, process, key or network operation. The API token is retained
    only for legacy schema compatibility and is not authentication.
    """
    profile = validate_profile(profile)
    bootstrap = _local_bootstrap(profile, bootstrap_data)
    binding = _binding(binding, bootstrap)
    safe_job_id(run_id)
    paths = runtime_paths(profile, binding["job_id"])
    manifest = profile["release_manifest"]
    value = {
        "schema": SCHEMA, "schema_version": 1, "mode": MODE, "run_id": run_id,
        "worker_id": MAID_ACTOR, "local_node_id": LOCAL_NODE_ID,
        **{k: paths[k] for k in ("bootstrap_path", "source_root", "shore_source_root",
                                "worker_config_path", "csv_path")},
        "bootstrap_sha256": profile["bootstrap_sha256"],
        # Preserve the existing strict runner schema. The supervisor separately
        # verifies the FULL release, including new adapters, before importing it.
        "source_hashes": {k: manifest["source_hashes"][k] for k in PROJECT_SOURCE_FILES},
        "shore_source_hashes": manifest["shore_source_hashes"],
        "csv_sha256": _digest(synthetic_csv(binding["job_spec"]["approved_units"])),
        "api_token": api_token, "binding": binding,
        "state_root": paths["maid_state_root"], "output_root": paths["maid_output_root"],
        "poll_seconds": 2, "heartbeat_seconds": 5,
    }
    return dict(RemoteSyntheticMaidConfig.from_mapping(value).raw)


def encode_material(profile: dict, assignment: dict, csv_data: bytes) -> bytes:
    """Serialize existing issued material; never creates/substitutes a permit."""
    profile = validate_profile(profile)
    if type(csv_data) is not bytes or len(csv_data) > MAX_CSV_BYTES:
        raise ValueError("Oversized CSV material")
    raw = _json_bytes({"schema": MATERIAL_SCHEMA, "schema_version": 1, "mode": MODE,
        "release_id": profile["release_id"], "assignment": assignment,
        "csv_b64": base64.b64encode(csv_data).decode("ascii")})
    if len(raw) > MAX_MATERIAL_BYTES:
        raise ValueError("Oversized material response")
    return raw


@dataclass(frozen=True)
class VerifiedMaterial:
    """In-memory validation output, not an execution capability."""

    job_id: str
    release_id: str
    material_sha256: str
    files: Mapping[str, bytes]
    record: Mapping


def validate_material(raw: bytes, *, profile: dict, bootstrap_data: bytes,
                      expected_binding: dict) -> VerifiedMaterial:
    """Verify fixed data and original wake against local policy and job binding.

    expected_binding is the ORIGINAL durable binding (including its session),
    not a replacement session from a later resume. Caller must separately fence
    fresh job state and verify installed release bytes/ACLs. This function does
    not infer ACTIVE from the existence of a signed permit.
    """
    from shore_control.permits import verify_permit
    from shore_control.registry import registration_sha256

    profile = validate_profile(profile)
    bootstrap = _local_bootstrap(profile, bootstrap_data)
    binding = _binding(expected_binding, bootstrap)
    value = _canonical_object(raw, MAX_MATERIAL_BYTES)
    if (set(value) != {"schema", "schema_version", "mode", "release_id", "assignment", "csv_b64"}
            or value["schema"] != MATERIAL_SCHEMA or type(value["schema_version"]) is not int
            or value["schema_version"] != 1 or value["mode"] != MODE
            or value["release_id"] != profile["release_id"]):
        raise ValueError("Material schema/release differs from local policy")
    assignment = value["assignment"]
    if type(assignment) is not dict or set(assignment) != {"runtime", "control_authority"}:
        raise ValueError("Expected original wake assignment")
    runtime = assignment["runtime"]
    runtime_raw = _json_bytes(runtime)
    if len(runtime_raw) > MAX_RUNTIME_BYTES:
        raise ValueError("Oversized runtime")
    config = RemoteSyntheticMaidConfig.from_mapping(runtime)
    expected = build_runtime(profile, bootstrap_data, binding, run_id=config["run_id"],
                             api_token=config["api_token"])
    if runtime_raw != _json_bytes(expected):
        raise ValueError("Runtime differs from local release/paths/binding")
    if type(value["csv_b64"]) is not str:
        raise ValueError("Invalid CSV encoding")
    csv_data = base64.b64decode(value["csv_b64"], validate=True)
    if (len(csv_data) > MAX_CSV_BYTES or base64.b64encode(csv_data).decode() != value["csv_b64"]
            or _digest(csv_data) != config["csv_sha256"]
            or csv_data != synthetic_csv(binding["job_spec"]["approved_units"])):
        raise ValueError("CSV differs from the fixed bounded synthetic workload")
    control = assignment["control_authority"]
    if type(control) is not dict or set(control) != {"permit", "descriptor"}:
        raise ValueError("Original signed wake permit/descriptor required")
    permit = verify_permit(bootstrap["authority_public_key"], control["permit"], control["descriptor"])
    registration = bootstrap["workers"][BELL_ACTOR]
    operation_id = config["run_id"] + ".wake-001"
    expected_descriptor = {**binding, "operation_id": operation_id, "attempt_id": operation_id,
        "run_id": config["run_id"], "worker_id": BELL_ACTOR, "action": "wake",
        "input_sha256": config.runtime_sha256, "device_id": DEVICE_ID, "worker_role": "bell",
        "worker_registration_sha256": registration_sha256(registration),
        "action_context": {"target_actor_id": MAID_ACTOR, "runtime_sha256": config.runtime_sha256}}
    if _json_bytes(control["descriptor"]) != _json_bytes(expected_descriptor):
        raise ValueError("Wake permit differs from the fixed actor/job/runtime")
    paths = runtime_paths(profile, binding["job_id"])
    bell_config = {"schema": "msabp.shore-remote-synthetic-bell", "schema_version": 1, "mode": MODE,
        "runtime_config_path": paths["runtime_config_path"], "runtime_sha256": config.runtime_sha256,
        "state_root": paths["bell_state_root"], "poll_seconds": 2}
    files = {"authority-bootstrap.json": bootstrap_data, "samples.csv": csv_data,
        "maid-runtime.json": runtime_raw, "bell-config.json": _json_bytes(bell_config)}
    record = {"schema": "msabp.supervisor-material-record", "schema_version": 1,
        "job_id": binding["job_id"], "binding": binding, "release_id": profile["release_id"],
        "material_sha256": _digest(raw), "runtime_sha256": config.runtime_sha256,
        "file_sha256": {name: _digest(data) for name, data in files.items()},
        "control_authority": control, "permit_id": permit["permit_id"]}
    return VerifiedMaterial(binding["job_id"], profile["release_id"], _digest(raw), files, record)
