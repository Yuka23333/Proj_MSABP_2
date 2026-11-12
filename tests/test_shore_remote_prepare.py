"""Offline public synthetic preparation; no device key or network fixture."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_shore_remote_worker import remote_worker  # noqa: F401
from msabp_opt.simulation.distributed.control_remote_prepare import (
    PREPARATION_SOURCE_FILES, prepare_synthetic_job, validate_preparation,
)


@pytest.fixture
def prepare_args(remote_worker, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr("socket.socket", lambda *_a, **_k: pytest.fail("No network"))
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("No process"))
    # Fixture code names are the exact allowlist; no arbitrary manifest paths.
    source = tmp_path / "source-fixture"
    for name in PREPARATION_SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Offline preparation test source\n", encoding="utf-8")
    return {"output_root": tmp_path / "prepared", "job_id": "synthetic-offline", "run_id": "run-offline",
        "source_root": source, "shore_source_root": remote_worker.config["shore_source_root"],
        "bootstrap_path": remote_worker.config["bootstrap_path"],
        "bootstrap_sha256": remote_worker.config["bootstrap_sha256"], "points": 1}


@pytest.mark.parametrize("points", [1, 4])
def test_public_preparation_exact_budget_paths_and_no_key_material(prepare_args, points):
    result = prepare_synthetic_job(**{**prepare_args, "points": points})
    assert result["points"] == result["job_spec"]["approved_units"] == points
    assert result["synthetic"] is True and result["physical_result"] is False
    assert set(result["paths"]) == {"samples_csv", "job_manifest", "accounting_policy", "bootstrap_path"}
    assert len(Path(result["paths"]["samples_csv"]).read_text().splitlines()) == points + 1
    assert set(path.name for path in prepare_args["output_root"].iterdir()) == {
        "samples.csv", "job-manifest.json", "accounting-policy.json", "authority-bootstrap.json",
        ".gitignore", "preparation.json"}
    assert result == validate_preparation(prepare_args["output_root"])
    assert not any("PRIVATE KEY" in path.read_text() for path in prepare_args["output_root"].iterdir())
    assert "binding" not in result


@pytest.mark.parametrize("points", [0, 2, 128, True])
def test_only_explicit_one_or_four_points(prepare_args, points):
    with pytest.raises(ValueError):
        prepare_synthetic_job(**{**prepare_args, "points": points})
    assert not prepare_args["output_root"].exists()


def test_preparation_never_overwrites(prepare_args):
    first = prepare_synthetic_job(**prepare_args)
    before = Path(first["preparation_path"]).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_synthetic_job(**prepare_args)
    assert Path(first["preparation_path"]).read_bytes() == before


def test_drift_is_reported_and_original_bytes_retained(prepare_args):
    result = prepare_synthetic_job(**prepare_args)
    frozen = Path(result["preparation_path"]).read_bytes()
    source = prepare_args["source_root"] / PREPARATION_SOURCE_FILES[0]
    source.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source has drifted"):
        validate_preparation(result["preparation_path"])
    assert Path(result["preparation_path"]).read_bytes() == frozen


def test_manifest_cannot_drop_required_source_pin(prepare_args):
    result = prepare_synthetic_job(**prepare_args)
    path = Path(result["preparation_path"])
    record = json.loads(path.read_bytes())
    record["source_hashes"].pop(PREPARATION_SOURCE_FILES[0])
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed allowlist"):
        validate_preparation(path)


def test_modified_csv_even_with_updated_hash_is_not_a_synthetic_plan(prepare_args):
    result = prepare_synthetic_job(**prepare_args)
    path = Path(result["preparation_path"])
    record = json.loads(path.read_bytes())
    csv = Path(result["paths"]["samples_csv"])
    csv.write_text("sample_id,value\nphysical,999\n", encoding="utf-8")
    record["artifact_sha256"]["samples.csv"] = hashlib.sha256(csv.read_bytes()).hexdigest()
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed synthetic workload"):
        validate_preparation(path)


def test_bootstrap_pin_mismatch_creates_nothing(prepare_args):
    args = copy.copy(prepare_args)
    args["bootstrap_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bootstrap SHA256"):
        prepare_synthetic_job(**args)
    assert not prepare_args["output_root"].exists()


def test_prepared_public_inputs_bind_to_strict_remote_runtime_without_network(prepare_args, remote_worker):  # noqa: F811
    from msabp_opt.simulation.distributed.control_remote_controller import bind_job, CollectorWakeQueue
    from msabp_opt.simulation.distributed.control_remote_worker import RemoteSyntheticMaidConfig, _json_bytes

    preparation = prepare_synthetic_job(**prepare_args)
    calls = []
    class FakeGate:
        controller = SimpleNamespace(fleet_id=remote_worker.bootstrap["fleet_id"])
        registrations = remote_worker.bootstrap["workers"]
        def acquire(self, **kwargs):
            calls.append("acquire")
            self.binding = {key: kwargs[key] for key in ("job_id", "host_id", "session_id")}
            self.binding["epoch"] = 1
            self.job = {"phase": "ACTIVE", "scope": {key: kwargs[key] for key in (
                "project_id", "collector_url", "data_root", "job_spec")}}
        def bind_job_documents(self, manifest_path, policy_path):
            calls.append("documents")
            assert json.loads(Path(manifest_path).read_bytes())["synthetic"] is True
            assert set(json.loads(Path(policy_path).read_bytes())) == {
                "schema", "schema_version", "unit_key", "settled_outcomes"}
        def register_run(self, run_id, plan_hash, csv_hash):
            calls.append("register")
            assert run_id == preparation["run_id"]
            assert len(plan_hash) == len(csv_hash) == 64
        def bind_run_plan(self, run_id, runtime_path):
            calls.append("plan")
            assert Path(runtime_path).is_file()
        def descriptor_for_attempt(self, intent):
            from shore_control.registry import registration_sha256
            registration = self.registrations[intent["worker_id"]]
            return {**intent, **self.binding, **self.job["scope"],
                "fleet_id": self.controller.fleet_id,
                "device_id": registration["device_id"], "worker_role": registration["role"],
                "worker_registration_sha256": registration_sha256(registration)}
    gate = FakeGate()
    prepared, runtime = bind_job(preparation, gate, remote_worker.bootstrap, "offline-session", "offline-public-key")
    assert calls == ["acquire", "documents", "register", "plan"]
    config = RemoteSyntheticMaidConfig.from_mapping(runtime)
    deployment = Path(preparation["output_root"]) / "deployment"
    assert (deployment / "maid-runtime.json").read_bytes() == _json_bytes(runtime)
    bell = json.loads((deployment / "bell-config.json").read_bytes())
    assert bell["runtime_sha256"] == config.runtime_sha256
    assert bell["runtime_config_path"] == preparation["remote_paths"]["runtime_config_path"]
    assert runtime["csv_sha256"] == prepared.worklist.worklist.sha256
    assert runtime["binding"]["job_spec"]["approved_units"] == 1
    queue = CollectorWakeQueue(Path(preparation["output_root"]) / "offline-wake", gate, runtime)
    assert queue.intent["input_sha256"] == config.runtime_sha256
    assert queue.intent["action_context"]["runtime_sha256"] == config.runtime_sha256
