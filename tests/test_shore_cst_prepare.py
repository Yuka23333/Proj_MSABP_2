"""Offline one-point CST preparation; no private key, network or process."""
from __future__ import annotations

import copy
import csv
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_shore_remote_prepare import prepare_args, remote_worker  # noqa: F401
from msabp_opt.simulation.distributed.control_cst_prepare import (
    cst_runtime_paths,
    prepare_cst_job,
    validate_cst_preparation,
)
from msabp_opt.simulation.distributed.control_release import (
    RELEASE_SOURCE_FILES,
    build_release_manifest,
    public_profile,
)


def one_row(sample_id="cst-001", tree=None):
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("sample_id", "simulation_mode", "tree_json"))
    writer.writerow((sample_id, "antenna_tree", json.dumps(tree or {}, separators=(",", ":"))))
    return stream.getvalue().encode()


@pytest.fixture
def cst_args(prepare_args, tmp_path):
    source = prepare_args["source_root"]
    for name in RELEASE_SOURCE_FILES:
        path = source / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixed CST preparation fixture\n", encoding="utf-8")
    profile = public_profile(
        build_release_manifest(source, prepare_args["shore_source_root"]),
        prepare_args["bootstrap_sha256"],
    )
    csv_path = tmp_path / "input.csv"
    csv_path.write_bytes(one_row())
    project = tmp_path / "input.cst"
    project.write_bytes(b"fixed standalone cst template")
    return {
        "output_root": tmp_path / "prepared-cst",
        "job_id": "cst-offline",
        "run_id": "cst-run-001",
        "source_root": source,
        "shore_source_root": prepare_args["shore_source_root"],
        "bootstrap_path": prepare_args["bootstrap_path"],
        "bootstrap_sha256": prepare_args["bootstrap_sha256"],
        "csv_path": csv_path,
        "project_path": project,
        "release_profile": profile,
    }


def test_cst_preparation_freezes_exact_one_point_and_template(cst_args):
    result = prepare_cst_job(**cst_args)
    assert result["points"] == result["job_spec"]["approved_units"] == 1
    assert result["synthetic"] is False and result["physical_result"] is True
    assert set(result["paths"]) == {
        "samples_csv", "job_manifest", "accounting_policy", "bootstrap_path",
        "project_template",
    }
    assert Path(result["paths"]["project_template"]).name == "msa-bp.cst"
    assert result["remote_paths"] == cst_runtime_paths(result["release_profile"], result["job_id"])
    assert result == validate_cst_preparation(result["preparation_path"])
    assert set(p.name for p in Path(result["output_root"]).iterdir()) == {
        "samples.csv", "job-manifest.json", "accounting-policy.json",
        "authority-bootstrap.json", "msa-bp.cst", ".gitignore",
        "job-http-nonces.sqlite", "preparation.json",
    }


def test_cst_preparation_binds_exact_runtime_template_and_synthetic_bell_channel(
    cst_args, remote_worker,
):  # noqa: F811
    from msabp_opt.simulation.distributed.control_remote_controller import bind_job
    from msabp_opt.simulation.distributed.control_remote_worker import (
        CST_MODE,
        CST_PROJECT_SOURCE_FILES,
        MODE,
        RemoteSyntheticMaidConfig,
        _json_bytes,
    )

    preparation = prepare_cst_job(**cst_args)
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
            manifest = json.loads(Path(manifest_path).read_bytes())
            assert manifest["mode"] == CST_MODE
            assert manifest["synthetic"] is False and manifest["physical_result"] is True
            assert set(json.loads(Path(policy_path).read_bytes())) == {
                "schema", "schema_version", "unit_key", "settled_outcomes"}

        def register_run(self, run_id, plan_hash, csv_hash):
            calls.append("register")
            assert run_id == preparation["run_id"]
            assert len(plan_hash) == len(csv_hash) == 64

        def bind_run_plan(self, run_id, runtime_path):
            calls.append("plan")
            assert run_id == preparation["run_id"] and Path(runtime_path).is_file()

        def descriptor_for_attempt(self, intent):
            from shore_control.registry import registration_sha256

            registration = self.registrations[intent["worker_id"]]
            return {**intent, **self.binding, **self.job["scope"],
                "fleet_id": self.controller.fleet_id,
                "device_id": registration["device_id"], "worker_role": registration["role"],
                "worker_registration_sha256": registration_sha256(registration)}

    gate = FakeGate()
    prepared, runtime = bind_job(
        preparation, gate, remote_worker.bootstrap, "offline-session", "offline-public-key")
    assert calls == ["acquire", "documents", "register", "plan"]
    config = RemoteSyntheticMaidConfig.from_mapping(runtime)
    assert config["mode"] == CST_MODE
    assert set(config["source_hashes"]) == set(CST_PROJECT_SOURCE_FILES)
    assert config["csv_sha256"] == prepared.worklist.worklist.sha256
    assert config["project_template_sha256"] == preparation["artifact_sha256"]["msa-bp.cst"]

    deployment = Path(preparation["output_root"]) / "deployment"
    assert (deployment / "maid-runtime.json").read_bytes() == _json_bytes(runtime)
    assert (deployment / "msa-bp.cst").read_bytes() == Path(
        preparation["paths"]["project_template"]).read_bytes()
    bell = json.loads((deployment / "bell-config.json").read_bytes())
    assert bell["schema"] == "msabp.shore-remote-synthetic-bell"
    assert bell["mode"] == MODE
    assert bell["runtime_sha256"] == config.runtime_sha256
    assert bell["runtime_config_path"] == preparation["remote_paths"]["runtime_config_path"]


@pytest.mark.parametrize("csv_data", [
    b"sample_id,simulation_mode,tree_json\na,antenna_tree,{}\nb,antenna_tree,{}\n",
    b"sample_id,simulation_mode,tree_json\na,wrong,{}\n",
    b"simulation_mode,sample_id,tree_json\nantenna_tree,a,{}\n",
])
def test_invalid_cst_csv_creates_nothing(cst_args, csv_data):
    Path(cst_args["csv_path"]).write_bytes(csv_data)
    with pytest.raises(ValueError):
        prepare_cst_job(**cst_args)
    assert not Path(cst_args["output_root"]).exists()


def test_cst_preparation_never_overwrites(cst_args):
    first = prepare_cst_job(**cst_args)
    before = Path(first["preparation_path"]).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_cst_job(**cst_args)
    assert Path(first["preparation_path"]).read_bytes() == before


def test_template_drift_is_rejected_without_repair(cst_args):
    result = prepare_cst_job(**cst_args)
    frozen = Path(result["preparation_path"]).read_bytes()
    Path(result["paths"]["project_template"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="drifted"):
        validate_cst_preparation(result["preparation_path"])
    assert Path(result["preparation_path"]).read_bytes() == frozen


def test_profile_mismatch_creates_nothing(cst_args):
    args = copy.deepcopy(cst_args)
    args["release_profile"]["bootstrap_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        prepare_cst_job(**args)
    assert not Path(args["output_root"]).exists()
