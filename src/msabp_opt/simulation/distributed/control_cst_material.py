"""Strict, public-only material contract for one real Shore-controlled CST case.

This module deliberately does not extend the synthetic-v1 envelope.  The two
profiles can ship in one reviewed release, but their material bytes, schemas and
validators remain independent.  Validation is offline: it neither opens CST nor
reads a private key, contacts the authority, or grants permission to launch.
"""
from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
import math
from pathlib import PureWindowsPath
import re
from typing import Any, Mapping

from .control_job_material import (
    MAX_CSV_BYTES,
    MAX_MATERIAL_BYTES,
    MAX_RUNTIME_BYTES,
    _binding,
    _canonical_object,
    _local_bootstrap,
)
from .control_release import cst_runtime_paths, safe_job_id, validate_profile
from .control_remote_worker import (
    BELL_ACTOR,
    CST_PROJECT_SOURCE_FILES,
    DEVICE_ID,
    LOCAL_NODE_ID,
    MAID_ACTOR,
    MODE,
    SHORE_SOURCE_FILES,
    _absolute_path,
    _hash,
    _json_bytes,
)


CST_MODE = "shore-remote-cst-v1"
CST_MAID_SCHEMA = "msabp.shore-remote-cst-maid"
CST_MATERIAL_SCHEMA = "msabp.supervisor-cst-job-material"
TEMPLATE_NAME = "msa-bp.cst"
MAX_TEMPLATE_BYTES = 512 * 1024
CSV_FIELDS = ("sample_id", "simulation_mode", "tree_json")
_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", flags=re.ASCII)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate tree_json field")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite tree_json number: " + value)


def _finite_tree(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Non-finite tree_json number")
        return
    if type(value) is list:
        for child in value:
            _finite_tree(child)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            _finite_tree(child)
        return
    raise ValueError("tree_json contains an unsupported value")


def validate_cst_csv(raw: bytes) -> dict[str, Any]:
    """Validate the exact one-row M3a antenna-tree table without canonicalizing it."""
    if type(raw) is not bytes or not raw or len(raw) > MAX_CSV_BYTES or b"\0" in raw:
        raise ValueError("Missing, oversized or NUL-containing CST CSV")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("CST CSV must be strict UTF-8 without a BOM") from exc
    if text.startswith("\ufeff"):
        raise ValueError("CST CSV must not contain a BOM")
    try:
        rows = list(csv.reader(StringIO(text, newline=""), strict=True))
    except (csv.Error, TypeError) as exc:
        raise ValueError("Malformed CST CSV") from exc
    if len(rows) != 2 or tuple(rows[0]) != CSV_FIELDS or len(rows[1]) != len(CSV_FIELDS):
        raise ValueError("CST CSV requires the exact header and exactly one data row")
    sample_id, simulation_mode, tree_raw = rows[1]
    if _SAMPLE_ID.fullmatch(sample_id) is None:
        raise ValueError("Invalid CST sample_id")
    if simulation_mode != "antenna_tree":
        raise ValueError("CST material supports antenna_tree only")
    try:
        tree = json.loads(
            tree_raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise ValueError("tree_json must be one strict JSON object") from exc
    if type(tree) is not dict:
        raise ValueError("tree_json must be one strict JSON object")
    _finite_tree(tree)
    return {"sample_id": sample_id, "simulation_mode": simulation_mode, "tree": tree}


def validate_cst_template(raw: bytes) -> str:
    if type(raw) is not bytes or not raw or len(raw) > MAX_TEMPLATE_BYTES:
        raise ValueError("Missing or oversized CST template")
    return _digest(raw)


def _canonical_b64(value: Any, label: str, maximum: int) -> bytes:
    if type(value) is not str:
        raise ValueError("Invalid " + label + " encoding")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid " + label + " encoding") from exc
    if len(raw) > maximum or base64.b64encode(raw).decode("ascii") != value:
        raise ValueError("Non-canonical or oversized " + label)
    return raw


@dataclass(frozen=True)
class RemoteCstMaidConfig:
    """Exact signed runtime shape for one real CST child-process execution."""

    raw: Mapping[str, Any]
    runtime_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RemoteCstMaidConfig":
        value = json.loads(_json_bytes(dict(raw)))
        required = {
            "schema", "schema_version", "mode", "run_id", "worker_id", "local_node_id",
            "bootstrap_path", "bootstrap_sha256", "source_root", "source_hashes",
            "shore_source_root", "shore_source_hashes", "worker_config_path", "csv_path",
            "csv_sha256", "api_token", "binding", "state_root", "output_root",
            "project_template_path", "project_template_sha256", "coordinate_quantum_mm",
            "allow_disconnected_conductor", "command_timeout_seconds", "poll_seconds",
            "heartbeat_seconds",
        }
        if set(value) != required:
            raise ValueError("Unexpected or missing CST Maid runtime fields")
        if (
            value["schema"] != CST_MAID_SCHEMA
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["mode"] != CST_MODE
            or value["worker_id"] != MAID_ACTOR
            or value["local_node_id"] != LOCAL_NODE_ID
        ):
            raise ValueError("Explicit fixed real-CST Maid mode and actor required")
        safe_job_id(value["run_id"])
        if type(value["api_token"]) is not str or not 0 < len(value["api_token"]) <= 256:
            raise ValueError("Invalid api_token")
        for name in ("bootstrap_sha256", "csv_sha256", "project_template_sha256"):
            _hash(value[name], name)
        if (
            value["coordinate_quantum_mm"] != "0.01"
            or value["allow_disconnected_conductor"] is not False
            or type(value["command_timeout_seconds"]) is not int
            or value["command_timeout_seconds"] != 15
            or type(value["poll_seconds"]) is not int
            or value["poll_seconds"] != 2
            or type(value["heartbeat_seconds"]) is not int
            or value["heartbeat_seconds"] != 5
        ):
            raise ValueError("CST solver parameters differ from the reviewed M3a profile")
        for name in (
            "bootstrap_path", "source_root", "shore_source_root", "worker_config_path",
            "csv_path", "state_root", "output_root", "project_template_path",
        ):
            if type(value[name]) is not str:
                raise ValueError("Invalid runtime path: " + name)
            _absolute_path(value[name])
        if PureWindowsPath(value["project_template_path"]).name != TEMPLATE_NAME:
            raise ValueError("CST runtime requires the fixed template filename")
        if type(value["binding"]) is not dict:
            raise ValueError("Missing CST job binding")
        for name, required_names in (
            ("source_hashes", CST_PROJECT_SOURCE_FILES),
            ("shore_source_hashes", SHORE_SOURCE_FILES),
        ):
            pins = value[name]
            if type(pins) is not dict or set(pins) != set(required_names):
                raise ValueError("CST runtime source pins differ from the reviewed release")
            for digest in pins.values():
                _hash(digest, name)
        return cls(value, _digest(_json_bytes(value)))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def build_runtime_cst(
    profile: dict,
    bootstrap_data: bytes,
    binding: dict,
    *,
    run_id: str,
    api_token: str,
    csv_data: bytes,
    project_template_sha256: str,
) -> dict:
    """Construct all paths locally; no executable or command comes from material."""
    profile = validate_profile(profile)
    bootstrap = _local_bootstrap(profile, bootstrap_data)
    binding = _binding(binding, bootstrap)
    if binding["job_spec"]["approved_units"] != 1:
        raise ValueError("M3a real CST material requires exactly one approved unit")
    validate_cst_csv(csv_data)
    safe_job_id(run_id)
    template_hash = _hash(project_template_sha256, "project_template_sha256")
    paths = cst_runtime_paths(profile, binding["job_id"])
    manifest = profile["release_manifest"]
    value = {
        "schema": CST_MAID_SCHEMA,
        "schema_version": 1,
        "mode": CST_MODE,
        "run_id": run_id,
        "worker_id": MAID_ACTOR,
        "local_node_id": LOCAL_NODE_ID,
        **{
            key: paths[key]
            for key in (
                "bootstrap_path", "source_root", "shore_source_root", "worker_config_path",
                "csv_path",
            )
        },
        "bootstrap_sha256": profile["bootstrap_sha256"],
        "source_hashes": {
            name: manifest["source_hashes"][name] for name in CST_PROJECT_SOURCE_FILES
        },
        "shore_source_hashes": {name: manifest["shore_source_hashes"][name] for name in SHORE_SOURCE_FILES},
        "csv_sha256": _digest(csv_data),
        "api_token": api_token,
        "binding": binding,
        "state_root": paths["maid_state_root"],
        "output_root": paths["maid_output_root"],
        "project_template_path": paths["project_template_path"],
        "project_template_sha256": template_hash,
        "coordinate_quantum_mm": "0.01",
        "allow_disconnected_conductor": False,
        "command_timeout_seconds": 15,
        "poll_seconds": 2,
        "heartbeat_seconds": 5,
    }
    return dict(RemoteCstMaidConfig.from_mapping(value).raw)


def encode_cst_material(
    profile: dict,
    assignment: dict,
    csv_data: bytes,
    project_template: bytes,
) -> bytes:
    profile = validate_profile(profile)
    validate_cst_csv(csv_data)
    validate_cst_template(project_template)
    raw = _json_bytes({
        "schema": CST_MATERIAL_SCHEMA,
        "schema_version": 1,
        "mode": CST_MODE,
        "release_id": profile["release_id"],
        "assignment": assignment,
        "csv_b64": base64.b64encode(csv_data).decode("ascii"),
        "project_template_b64": base64.b64encode(project_template).decode("ascii"),
    })
    if len(raw) > MAX_MATERIAL_BYTES:
        raise ValueError("Oversized CST material response")
    return raw


@dataclass(frozen=True)
class VerifiedCstMaterial:
    job_id: str
    release_id: str
    material_sha256: str
    files: Mapping[str, bytes]
    record: Mapping[str, Any]


def validate_cst_material(
    raw: bytes,
    *,
    profile: dict,
    bootstrap_data: bytes,
    expected_binding: dict,
) -> VerifiedCstMaterial:
    """Validate one original issued real-CST material envelope offline."""
    from shore_control.permits import verify_permit
    from shore_control.registry import registration_sha256

    profile = validate_profile(profile)
    bootstrap = _local_bootstrap(profile, bootstrap_data)
    binding = _binding(expected_binding, bootstrap)
    if binding["job_spec"]["approved_units"] != 1:
        raise ValueError("M3a real CST material requires exactly one approved unit")
    value = _canonical_object(raw, MAX_MATERIAL_BYTES)
    expected_fields = {
        "schema", "schema_version", "mode", "release_id", "assignment", "csv_b64",
        "project_template_b64",
    }
    if (
        set(value) != expected_fields
        or value["schema"] != CST_MATERIAL_SCHEMA
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["mode"] != CST_MODE
        or value["release_id"] != profile["release_id"]
    ):
        raise ValueError("CST material schema/release differs from local policy")
    assignment = value["assignment"]
    if type(assignment) is not dict or set(assignment) != {"runtime", "control_authority"}:
        raise ValueError("Expected original CST wake assignment")
    runtime = assignment["runtime"]
    runtime_raw = _json_bytes(runtime)
    if len(runtime_raw) > MAX_RUNTIME_BYTES:
        raise ValueError("Oversized CST runtime")
    config = RemoteCstMaidConfig.from_mapping(runtime)
    csv_data = _canonical_b64(value["csv_b64"], "CST CSV", MAX_CSV_BYTES)
    template = _canonical_b64(value["project_template_b64"], "CST template", MAX_TEMPLATE_BYTES)
    validate_cst_csv(csv_data)
    template_hash = validate_cst_template(template)
    expected_runtime = build_runtime_cst(
        profile,
        bootstrap_data,
        binding,
        run_id=config["run_id"],
        api_token=config["api_token"],
        csv_data=csv_data,
        project_template_sha256=template_hash,
    )
    if runtime_raw != _json_bytes(expected_runtime):
        raise ValueError("CST runtime differs from local release/paths/binding/material")
    control = assignment["control_authority"]
    if type(control) is not dict or set(control) != {"permit", "descriptor"}:
        raise ValueError("Original signed CST wake permit/descriptor required")
    permit = verify_permit(bootstrap["authority_public_key"], control["permit"], control["descriptor"])
    registration = bootstrap["workers"][BELL_ACTOR]
    operation_id = config["run_id"] + ".wake-001"
    expected_descriptor = {
        **binding,
        "operation_id": operation_id,
        "attempt_id": operation_id,
        "run_id": config["run_id"],
        "worker_id": BELL_ACTOR,
        "action": "wake",
        "input_sha256": config.runtime_sha256,
        "device_id": DEVICE_ID,
        "worker_role": "bell",
        "worker_registration_sha256": registration_sha256(registration),
        "action_context": {"target_actor_id": MAID_ACTOR, "runtime_sha256": config.runtime_sha256},
    }
    if _json_bytes(control["descriptor"]) != _json_bytes(expected_descriptor):
        raise ValueError("CST wake permit differs from the fixed actor/job/runtime")
    paths = cst_runtime_paths(profile, binding["job_id"])
    bell_config = {
        # Bell is the stable transport/lifecycle channel.  The hash-bound Maid
        # runtime selects CST; keeping this wrapper schema unchanged preserves
        # the installed Bell protocol and signed domains.
        "schema": "msabp.shore-remote-synthetic-bell",
        "schema_version": 1,
        "mode": MODE,
        "runtime_config_path": paths["runtime_config_path"],
        "runtime_sha256": config.runtime_sha256,
        "state_root": paths["bell_state_root"],
        "poll_seconds": 2,
    }
    files = {
        "authority-bootstrap.json": bootstrap_data,
        "samples.csv": csv_data,
        TEMPLATE_NAME: template,
        "maid-runtime.json": runtime_raw,
        "bell-config.json": _json_bytes(bell_config),
    }
    record = {
        "schema": "msabp.supervisor-material-record",
        "schema_version": 1,
        "mode": CST_MODE,
        "job_id": binding["job_id"],
        "binding": binding,
        "release_id": profile["release_id"],
        "material_sha256": _digest(raw),
        "runtime_sha256": config.runtime_sha256,
        "file_sha256": {name: _digest(data) for name, data in files.items()},
        "control_authority": control,
        "permit_id": permit["permit_id"],
    }
    return VerifiedCstMaterial(binding["job_id"], profile["release_id"], _digest(raw), files, record)
