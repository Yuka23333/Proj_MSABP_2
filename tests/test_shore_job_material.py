"""Material data-plane tests with generated in-memory authority/worker keys."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from test_shore_remote_worker import remote_worker  # noqa: F401
from shore_control.authority import Authority, digest
from shore_control.client import Client, MemorySigner
from shore_control.crypto import canonical_json, public_key_text
from shore_control.permits import DESCRIPTOR_FIELDS

from msabp_opt.simulation.distributed.control_job_material import (
    MAX_MATERIAL_BYTES, build_runtime, encode_material, synthetic_csv, validate_material,
)
from msabp_opt.simulation.distributed.control_release import (
    RELEASE_SOURCE_FILES, public_profile, runtime_paths,
)
from msabp_opt.simulation.distributed.control_remote_worker import (
    BELL_ACTOR, MAID_ACTOR, MODE, SHORE_SOURCE_FILES, _json_bytes,
)


@pytest.fixture
def materials(remote_worker, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr("socket.socket", lambda *_a, **_k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("Process forbidden"))
    bootstrap = remote_worker.bootstrap
    bootstrap_data = canonical_json(bootstrap)
    manifest = {"schema": "msabp.supervisor-release", "schema_version": 1, "mode": MODE,
        "project_id": "MSABP_2",
        "source_hashes": {n: digest(n) for n in RELEASE_SOURCE_FILES},
        "shore_source_hashes": {n: digest(n) for n in SHORE_SOURCE_FILES}}
    profile = public_profile(manifest, hashlib.sha256(bootstrap_data).hexdigest())

    def make(points=4):
        keys = remote_worker.keys
        authority = Authority(tmp_path / f"material-{points}.sqlite", keys["authority"],
            fleet_id=bootstrap["fleet_id"], controllers=bootstrap["controllers"], workers=bootstrap["workers"])
        actor = next(iter(bootstrap["controllers"]))
        controller = Client(authority, actor, MemorySigner(keys["controller"]),
                            keys["authority"].public_key(), bootstrap["fleet_id"])
        spec = {"schema": "shore.job-spec", "schema_version": 1, "manifest_sha256": digest("manifest"),
            "approved_units": points, "budget_unit": "objective_evaluation",
            "accounting_policy_sha256": digest("policy")}
        job = controller.call("acquire", job_id=f"job-{points}",
            host_id=bootstrap["network"]["local_node_id"], session_id="original-session",
            session_public_key=public_key_text(keys["session"].public_key()), project_id="MSABP_2",
            collector_url=bootstrap["collector"]["endpoint"], data_root=str(tmp_path), job_spec=spec)
        binding = {**{k: job[k] for k in ("job_id", "epoch", "host_id", "session_id")},
                   **job["scope"], "fleet_id": bootstrap["fleet_id"]}
        runtime = build_runtime(profile, bootstrap_data, binding, run_id=f"run-{points}",
                                 api_token="fixture-not-an-auth-credential")
        bound = {k: binding[k] for k in ("job_id", "epoch", "host_id", "session_id")}
        controller.call("register_run", **bound, run_id=runtime["run_id"],
                        original_plan_sha256=digest(runtime), input_manifest_sha256=runtime["csv_sha256"])
        op_id = runtime["run_id"] + ".wake-001"
        permit = controller.call("validate_dispatch", **bound, operation={
            "operation_id": op_id, "attempt_id": op_id, "run_id": runtime["run_id"],
            "worker_id": BELL_ACTOR, "action": "wake", "input_sha256": digest(runtime),
            "action_context": {"target_actor_id": MAID_ACTOR, "runtime_sha256": digest(runtime)}})
        control = {"permit": permit, "descriptor": {k: permit["payload"][k] for k in DESCRIPTOR_FIELDS}}
        assignment = {"runtime": runtime, "control_authority": control}
        csv = synthetic_csv(points)
        raw = encode_material(profile, assignment, csv)
        kwargs = {"profile": profile, "bootstrap_data": bootstrap_data, "expected_binding": binding}
        return SimpleNamespace(**locals())
    return SimpleNamespace(make=make, profile=profile, bootstrap_data=bootstrap_data)


@pytest.mark.parametrize("points", [1, 4])
def test_fixed_paths_final_digest_four_files_and_original_permit(materials, points):
    case = materials.make(points)
    result = validate_material(case.raw, **case.kwargs)
    assert set(result.files) == {"authority-bootstrap.json", "samples.csv", "maid-runtime.json", "bell-config.json"}
    assert result.files["authority-bootstrap.json"] == materials.bootstrap_data
    assert result.files["maid-runtime.json"] == canonical_json(case.runtime)
    assert result.record["runtime_sha256"] == case.permit["payload"]["input_sha256"]
    assert result.record["control_authority"] == case.control
    bell = json.loads(result.files["bell-config.json"])
    assert bell["runtime_config_path"] == runtime_paths(materials.profile, case.binding["job_id"])["runtime_config_path"]
    assert len(result.files["samples.csv"].splitlines()) == points + 1


@pytest.mark.parametrize("field,value", [
    ("source_root", r"C:\untrusted\app"), ("csv_path", r"C:\other.csv"),
    ("poll_seconds", 3), ("command", "evil.exe"), ("bootstrap_sha256", "a" * 64),
    ("run_id", "CON"), ("worker_id", BELL_ACTOR), ("heartbeat_seconds", True),
])
def test_tampered_runtime_never_rewritten_or_accepted(materials, field, value):
    case = materials.make()
    obj = json.loads(case.raw)
    obj["assignment"]["runtime"][field] = value
    with pytest.raises((ValueError, KeyError)):
        validate_material(_json_bytes(obj), **case.kwargs)


@pytest.mark.parametrize("field,value", [
    ("release_id", "a" * 64), ("schema_version", True), ("code", "print('no')"),
    ("csv_b64", base64.b64encode(b"sample_id,value\nevil,999\n").decode()),
    ("csv_b64", "***"), ("mode", "cst"),
])
def test_tampered_material_refused(materials, field, value):
    case = materials.make()
    obj = json.loads(case.raw)
    obj[field] = value
    with pytest.raises(ValueError):
        validate_material(_json_bytes(obj), **case.kwargs)


@pytest.mark.parametrize("kind", ["duplicate", "whitespace", "array", "oversize", "bad-signature"])
def test_strict_bounded_envelope(materials, kind):
    case = materials.make()
    raw = case.raw
    if kind == "duplicate":
        raw = b'{"mode":"x",' + raw[1:]
    elif kind == "whitespace":
        raw = b" " + raw
    elif kind == "array":
        raw = b"[]"
    elif kind == "oversize":
        raw = b" " * (MAX_MATERIAL_BYTES + 1)
    else:
        obj = json.loads(raw)
        obj["assignment"]["control_authority"]["permit"]["signature"] = "a" * 88
        raw = _json_bytes(obj)
    with pytest.raises(ValueError):
        validate_material(raw, **case.kwargs)


def test_old_permit_is_data_not_permission_to_launch_while_draining(materials):
    case = materials.make()
    bound = {k: case.binding[k] for k in ("job_id", "epoch", "host_id", "session_id")}
    case.controller.call("begin_drain", **bound, reason="offline material drain test")
    assert validate_material(case.raw, **case.kwargs).job_id == case.binding["job_id"]


def test_original_binding_not_rewritten_to_new_session(materials):
    case = materials.make()
    new = {**case.binding, "session_id": "new-session"}
    with pytest.raises(ValueError, match="binding"):
        validate_material(case.raw, **{**case.kwargs, "expected_binding": new})
    assert validate_material(case.raw, **case.kwargs).record["binding"]["session_id"] == "original-session"


@pytest.mark.parametrize("change", ["missing", "role", "device"])
def test_locally_pinned_bootstrap_must_still_match_fixed_runner_roles(materials, change):
    case = materials.make()
    bootstrap = json.loads(materials.bootstrap_data)
    if change == "missing":
        del bootstrap["workers"][MAID_ACTOR]
    elif change == "role":
        bootstrap["workers"][MAID_ACTOR].update(role="gpu", allowed_actions=["gpu"])
    else:
        bootstrap["workers"][MAID_ACTOR]["device_id"] = "other"
    raw = canonical_json(bootstrap)
    profile = public_profile(materials.profile["release_manifest"], hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="registration|mapping"):
        build_runtime(profile, raw, case.binding, run_id="run-4", api_token="x")


def test_profile_and_binding_input_not_mutated(materials):
    case = materials.make()
    before = copy.deepcopy(case.kwargs)
    validate_material(case.raw, **case.kwargs)
    assert case.kwargs == before
