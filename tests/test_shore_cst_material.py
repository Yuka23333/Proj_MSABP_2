"""Offline contract tests for the independent one-point real-CST material."""
from __future__ import annotations

import base64
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_shore_job_material import materials  # noqa: F401
from test_shore_remote_worker import remote_worker  # noqa: F401
from test_control_cst import SuccessfulRunner, call_executor
from shore_control.authority import digest
from shore_control.permits import DESCRIPTOR_FIELDS

from msabp_opt.simulation.distributed.control_cst_material import (
    CST_MODE,
    MAX_TEMPLATE_BYTES,
    TEMPLATE_NAME,
    build_runtime_cst,
    encode_cst_material,
    validate_cst_csv,
    validate_cst_material,
)
from msabp_opt.simulation.distributed.control_release import runtime_paths
from msabp_opt.simulation.distributed.control_remote_worker import BELL_ACTOR, MAID_ACTOR, _json_bytes
from msabp_opt.simulation.distributed import control_cst, control_remote_worker
from msabp_opt.simulation.distributed.control_job_auth import initialize_job_nonce_db
from msabp_opt.simulation.distributed.control_job_http import JobMaterialService
from msabp_opt.simulation.distributed.control_supervisor import MaterialJournal, initialize_state


def one_row_csv(*, sample_id="case-001", mode="antenna_tree", tree=None) -> bytes:
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("sample_id", "simulation_mode", "tree_json"))
    writer.writerow((sample_id, mode, json.dumps({"W": 67.0} if tree is None else tree,
                                                 sort_keys=True, separators=(",", ":"))))
    return stream.getvalue().encode()


@pytest.fixture
def cst_material(materials):  # noqa: F811
    base = materials.make(1)
    csv_data = one_row_csv()
    template = b"CST-template-fixture\x00\x01"
    runtime = build_runtime_cst(
        materials.profile,
        materials.bootstrap_data,
        base.binding,
        run_id="cst-run-001",
        api_token="fixture-not-an-auth-credential",
        csv_data=csv_data,
        project_template_sha256=hashlib.sha256(template).hexdigest(),
    )
    bound = {key: base.binding[key] for key in ("job_id", "epoch", "host_id", "session_id")}
    base.controller.call(
        "register_run",
        **bound,
        run_id=runtime["run_id"],
        original_plan_sha256=digest(runtime),
        input_manifest_sha256=runtime["csv_sha256"],
    )
    operation_id = runtime["run_id"] + ".wake-001"
    permit = base.controller.call(
        "validate_dispatch",
        **bound,
        operation={
            "operation_id": operation_id,
            "attempt_id": operation_id,
            "run_id": runtime["run_id"],
            "worker_id": BELL_ACTOR,
            "action": "wake",
            "input_sha256": digest(runtime),
            "action_context": {"target_actor_id": MAID_ACTOR, "runtime_sha256": digest(runtime)},
        },
    )
    control = {"permit": permit, "descriptor": {key: permit["payload"][key] for key in DESCRIPTOR_FIELDS}}
    assignment = {"runtime": runtime, "control_authority": control}
    raw = encode_cst_material(materials.profile, assignment, csv_data, template)
    kwargs = {
        "profile": materials.profile,
        "bootstrap_data": materials.bootstrap_data,
        "expected_binding": base.binding,
    }
    return SimpleNamespace(**locals())


def test_real_cst_roundtrip_has_exact_five_staged_files(cst_material):
    verified = validate_cst_material(cst_material.raw, **cst_material.kwargs)
    assert set(verified.files) == {
        "authority-bootstrap.json", "samples.csv", TEMPLATE_NAME,
        "maid-runtime.json", "bell-config.json",
    }
    assert verified.files[TEMPLATE_NAME] == cst_material.template
    assert verified.files["samples.csv"] == cst_material.csv_data
    assert verified.record["mode"] == CST_MODE
    assert verified.record["runtime_sha256"] == cst_material.permit["payload"]["input_sha256"]
    paths = runtime_paths(cst_material.materials.profile, cst_material.base.binding["job_id"])
    runtime = json.loads(verified.files["maid-runtime.json"])
    assert runtime["project_template_path"].lower() == (
        paths["config_root"] + "\\" + TEMPLATE_NAME
    ).lower()


def test_cst_job_service_and_supervisor_journal_preserve_exact_five_file_closure(
    cst_material, tmp_path,
):
    nonce_db = tmp_path / "job-http-nonces.sqlite"
    initialize_job_nonce_db(nonce_db)
    service = JobMaterialService(
        profile=cst_material.materials.profile,
        bootstrap_data=cst_material.materials.bootstrap_data,
        expected_binding=cst_material.base.binding,
        read_assignment=lambda: cst_material.assignment,
        csv_data=cst_material.csv_data,
        project_data=cst_material.template,
        workload_mode=CST_MODE,
        nonce_db=nonce_db,
    )
    raw, offer = service._original()
    assert raw == cst_material.raw
    assert offer["material_sha256"] == hashlib.sha256(raw).hexdigest()

    config = tmp_path / "installed-config"
    config.mkdir()
    supervisor = tmp_path / "supervisor"
    initialize_state(supervisor, config, cst_material.materials.profile)
    journal = MaterialJournal(
        supervisor,
        profile=cst_material.materials.profile,
        bootstrap_data=cst_material.materials.bootstrap_data,
    )
    job_id = cst_material.base.binding["job_id"]
    journal.download(job_id, raw, expected_binding=cst_material.base.binding)
    journal.verify(job_id)
    target = journal.commit(job_id)
    assert {path.name for path in target.iterdir()} == {
        "authority-bootstrap.json",
        "samples.csv",
        TEMPLATE_NAME,
        "maid-runtime.json",
        "bell-config.json",
    }
    assert (target / TEMPLATE_NAME).read_bytes() == cst_material.template


def test_cst_runtime_selects_isolated_executor_and_one_mock_child(
    remote_worker, tmp_path, monkeypatch,
):  # noqa: F811
    source_root = Path(remote_worker.raw["source_root"])
    csv_path = tmp_path / "cst-row.csv"
    csv_path.write_bytes(one_row_csv())
    template = tmp_path / TEMPLATE_NAME
    template.write_bytes(b"fixed worker-local CST template")
    solver_root = tmp_path / "solver-env"
    (solver_root / "Library" / "bin").mkdir(parents=True)
    (solver_root / "Scripts").mkdir()
    solver_python = solver_root / "python.exe"
    solver_python.write_bytes(b"test executable placeholder")

    raw = json.loads(json.dumps(remote_worker.raw))
    raw.update({
        "schema": control_remote_worker.CST_SCHEMA,
        "mode": CST_MODE,
        "csv_path": str(csv_path),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source_hashes": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in control_remote_worker.CST_PROJECT_SOURCE_FILES
        },
        "project_template_path": str(template),
        "project_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "coordinate_quantum_mm": "0.01",
        "allow_disconnected_conductor": False,
        "command_timeout_seconds": 15,
        "poll_seconds": 2,
        "heartbeat_seconds": 5,
    })
    raw["binding"]["job_spec"]["approved_units"] = 1
    config = control_remote_worker.RemoteSyntheticMaidConfig.from_mapping(raw)
    monkeypatch.setattr(control_remote_worker, "CST_PYTHON_EXECUTABLE", solver_python)

    session, executor = control_remote_worker._execution_dependencies(config)
    assert isinstance(session, control_cst.CstProcessSession)
    assert isinstance(executor, control_cst.CstCaseExecutor)
    assert executor.policy.project_template == template.resolve()
    runner = SuccessfulRunner()
    executor._process_runner = runner
    result = call_executor(executor, tmp_path)
    assert result.case_id == "case-001"
    assert result.simulation_mode == "antenna_tree"
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "raw,match",
    [
        (b"sample_id,simulation_mode,tree_json\n", "exactly one"),
        (b"sample_id,simulation_mode,tree_json\na,antenna_tree,{}\nb,antenna_tree,{}\n", "exactly one"),
        (b"sample_id,tree_json,simulation_mode\na,{},antenna_tree\n", "exact header"),
        (b"sample_id,simulation_mode,tree_json\na,legacy,{}\n", "antenna_tree"),
        (b"sample_id,simulation_mode,tree_json\na,antenna_tree,[]\n", "JSON object"),
        (b"sample_id,simulation_mode,tree_json\na,antenna_tree,{\"x\":1\x00}\n", "NUL"),
        (b"\xef\xbb\xbfsample_id,simulation_mode,tree_json\na,antenna_tree,{}\n", "BOM"),
        (b"sample_id,simulation_mode,tree_json\n../a,antenna_tree,{}\n", "sample_id"),
        (b"sample_id,simulation_mode,tree_json\na,antenna_tree,{\"x\":NaN}\n", "Non-finite"),
        (b'sample_id,simulation_mode,tree_json\na,antenna_tree,"{""x"":1,""x"":2}"\n', "Duplicate"),
    ],
)
def test_strict_single_row_csv(raw, match):
    with pytest.raises(ValueError, match=match):
        validate_cst_csv(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("project_template_path", r"C:\untrusted\other.cst"),
        ("project_template_sha256", "a" * 64),
        ("coordinate_quantum_mm", "0.1"),
        ("allow_disconnected_conductor", True),
        ("command_timeout_seconds", 60),
        ("python_executable", r"C:\evil.exe"),
        ("mode", "shore-remote-synthetic-v1"),
    ],
)
def test_runtime_tampering_is_not_rewritten(cst_material, field, value):
    material = json.loads(cst_material.raw)
    material["assignment"]["runtime"][field] = value
    with pytest.raises((ValueError, KeyError)):
        validate_cst_material(_json_bytes(material), **cst_material.kwargs)


@pytest.mark.parametrize("target", ["csv", "template"])
def test_material_payload_and_signed_hash_must_agree(cst_material, target):
    material = json.loads(cst_material.raw)
    key = "csv_b64" if target == "csv" else "project_template_b64"
    original = base64.b64decode(material[key])
    changed = original.replace(b"67.0", b"68.0") if target == "csv" else original + b"x"
    material[key] = base64.b64encode(changed).decode()
    with pytest.raises(ValueError, match="runtime differs"):
        validate_cst_material(_json_bytes(material), **cst_material.kwargs)


def test_noncanonical_base64_and_unknown_material_fields_fail(cst_material):
    material = json.loads(cst_material.raw)
    material["project_template_b64"] += "="
    with pytest.raises(ValueError, match="canonical|encoding"):
        validate_cst_material(_json_bytes(material), **cst_material.kwargs)
    material = json.loads(cst_material.raw)
    material["command"] = "evil.exe"
    with pytest.raises(ValueError, match="schema"):
        validate_cst_material(_json_bytes(material), **cst_material.kwargs)


def test_template_and_total_material_limits(cst_material):
    with pytest.raises(ValueError, match="template"):
        encode_cst_material(
            cst_material.materials.profile,
            cst_material.assignment,
            cst_material.csv_data,
            b"x" * (MAX_TEMPLATE_BYTES + 1),
        )
    assert len(cst_material.raw) < 1024 * 1024


def test_cst_profile_refuses_more_than_one_approved_unit(materials):  # noqa: F811
    base = materials.make(4)
    with pytest.raises(ValueError, match="exactly one"):
        build_runtime_cst(
            materials.profile,
            materials.bootstrap_data,
            base.binding,
            run_id="cst-run-004",
            api_token="x",
            csv_data=one_row_csv(),
            project_template_sha256="a" * 64,
        )
