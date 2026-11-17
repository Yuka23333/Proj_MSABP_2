from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from msabp_opt.simulation.distributed import control_cst


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "msabp_opt" / "simulation" / "distributed" / "control_cst.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy(tmp_path: Path) -> control_cst.CstProcessPolicy:
    env = tmp_path / "solver-env"
    (env / "Library" / "bin").mkdir(parents=True)
    (env / "Scripts").mkdir()
    python = env / "python.exe"
    python.write_bytes(b"fixed solver executable")
    app = tmp_path / "app"
    (app / "src").mkdir(parents=True)
    (app / "scripts").mkdir()
    project = tmp_path / "material" / "msa-bp.cst"
    project.parent.mkdir()
    project.write_bytes(b"approved standalone template")
    return control_cst.CstProcessPolicy(python, app, project, sha256(project))


def write_artifacts(output_root: Path, case_id: str) -> tuple[Path, Path]:
    case = output_root / ("case_" + case_id)
    case.mkdir()
    records = {}
    for key, name in control_cst._REQUIRED_ARTIFACTS.items():
        data = (case_id + " " + key + "\n").encode()
        path = case / name
        path.write_bytes(data)
        records[key] = {"path": name, "size_bytes": len(data), "sha256": sha256(path)}
    manifest = case / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "status": "completed",
        "dry_run": False,
        "case_id": case_id,
        "simulation_mode": "antenna_tree",
        "elapsed_seconds": 12.5,
        "artifacts": records,
    }), encoding="utf-8")
    return case, manifest


class SuccessfulRunner:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        case, manifest = write_artifacts(Path(request["output_root"]), request["case_id"])
        response_path.write_text(json.dumps({
            "schema": control_cst.RESPONSE_SCHEMA,
            "schema_version": 1,
            "success": True,
            "case_id": request["case_id"],
            "case_directory": str(case),
            "manifest_path": str(manifest),
            "project_copy_sha256_before": request["project_copy_sha256"],
            "project_copy_sha256_after": sha256(Path(request["project_copy"])),
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)


def call_executor(executor: control_cst.CstCaseExecutor, tmp_path: Path, **overrides):
    values = dict(
        project_path=executor.policy.project_template,
        output_root=tmp_path / "attempt",
        project=None,
        case_id="case-001",
        coordinate_quantum_mm=0.01,
        allow_disconnected_conductor=False,
        command_timeout=60,
        overwrite=False,
        save_project_after_case=False,
        dry_run=False,
        stage_callback=None,
        local_artifact_root=tmp_path / "local-only",
    )
    values.update(overrides)
    return executor({"sample_id": "case-001", "tree_json": "{}"}, **values)


def test_parent_module_top_level_is_stdlib_only() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__", "argparse", "dataclasses", "hashlib", "json", "math", "os",
        "pathlib", "re", "shutil", "subprocess", "sys", "traceback", "typing", "uuid",
    }
    source = MODULE.read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "import shapely" not in source
    assert control_cst.CstProcessSession().open() is None


def test_executor_uses_fixed_policy_conda_environment_and_validates_manifest(tmp_path, monkeypatch) -> None:
    runner = SuccessfulRunner()
    executor = control_cst.CstCaseExecutor(policy(tmp_path), process_runner=runner)
    monkeypatch.setenv(control_cst.CONTROL_MODE_ENV, "shore-remote-cst-v1")
    stages = []
    result = call_executor(executor, tmp_path, stage_callback=stages.append)

    assert result.case_id == "case-001"
    assert result.simulation_mode == "antenna_tree"
    assert result.s11_path.read_text().startswith("case-001")
    assert stages == ["starting_cst_subprocess", "cst_subprocess_finished"]
    assert len(runner.calls) == 1
    command, invocation = runner.calls[0]
    assert command[:3] == [str(executor.policy.python_executable), "-I", "-B"]
    assert command[3] == str(MODULE)
    environment = invocation["env"]
    assert control_cst.CONTROL_MODE_ENV not in environment
    assert environment["CONDA_PREFIX"] == str(executor.policy.python_executable.parent)
    prefix = environment["PATH"].split(os.pathsep)[:3]
    assert prefix == [
        str(executor.policy.python_executable.parent),
        str(executor.policy.python_executable.parent / "Library" / "bin"),
        str(executor.policy.python_executable.parent / "Scripts"),
    ]
    request = next((tmp_path / "attempt" / ".cst-child").glob("*/request.json"))
    payload = json.loads(request.read_text(encoding="utf-8"))
    assert "python_executable" not in payload
    assert Path(payload["project_copy"]).parent == request.parent / "model"
    assert sha256(executor.policy.project_template) == executor.policy.project_template_sha256


@pytest.mark.parametrize("reserved", ["python_path", "executable", "project_path", "COMMAND"])
def test_task_row_cannot_select_executable_or_path(tmp_path, reserved) -> None:
    runner = SuccessfulRunner()
    executor = control_cst.CstCaseExecutor(policy(tmp_path), process_runner=runner)
    with pytest.raises(ValueError, match="may not select"):
        executor({"sample_id": "case-001", reserved: "attacker-controlled"},
            project_path=executor.policy.project_template, output_root=tmp_path / "attempt",
            project=None, case_id="case-001", coordinate_quantum_mm=.01,
            allow_disconnected_conductor=False, command_timeout=60, overwrite=False,
            save_project_after_case=False, dry_run=False, stage_callback=None)
    assert runner.calls == []


def test_native_child_failure_is_terminal_and_never_retried(tmp_path) -> None:
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, -1066598273)
    executor = control_cst.CstCaseExecutor(policy(tmp_path), process_runner=fail)
    with pytest.raises(control_cst.CstProcessError, match="-1066598273"):
        call_executor(executor, tmp_path)
    assert len(calls) == 1


def test_manifest_hash_mismatch_is_rejected(tmp_path) -> None:
    runner = SuccessfulRunner()
    executor = control_cst.CstCaseExecutor(policy(tmp_path), process_runner=runner)
    original = runner.__call__
    def corrupt(command, **kwargs):
        result = original(command, **kwargs)
        request = json.loads(Path(command[command.index("--request") + 1]).read_text())
        (Path(request["output_root"]) / "case_case-001" / "S11.csv").write_text("tampered")
        return result
    executor._process_runner = corrupt
    with pytest.raises(control_cst.CstProcessError, match="Artifact size mismatch"):
        call_executor(executor, tmp_path)


def test_manifest_extra_artifact_is_rejected(tmp_path) -> None:
    runner = SuccessfulRunner()
    executor = control_cst.CstCaseExecutor(policy(tmp_path), process_runner=runner)
    original = runner.__call__
    def add_extra(command, **kwargs):
        result = original(command, **kwargs)
        request = json.loads(Path(command[command.index("--request") + 1]).read_text())
        manifest_path = Path(request["output_root"]) / "case_case-001" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["unexpected"] = manifest["artifacts"]["s11"]
        manifest_path.write_text(json.dumps(manifest))
        return result
    executor._process_runner = add_extra
    with pytest.raises(control_cst.CstProcessError, match="exact tree result closure"):
        call_executor(executor, tmp_path)


@pytest.mark.parametrize("suffix", [',\"schema\":\"duplicate\"', ',\"bad\":NaN'])
def test_strict_json_rejects_duplicate_keys_and_nan(tmp_path, suffix) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"schema":"x"' + suffix + "}", encoding="utf-8")
    with pytest.raises(ValueError):
        control_cst._strict_json(path.read_text(encoding="utf-8"))


def test_child_clears_control_mode_closes_cst_and_writes_response(tmp_path, monkeypatch) -> None:
    request_root = tmp_path / "attempt" / ".cst-child" / "case-001-fixed"
    model = request_root / "model"
    model.mkdir(parents=True)
    project_copy = model / "msa-bp.cst"
    project_copy.write_bytes(b"copy")
    app = tmp_path / "app"
    (app / "src").mkdir(parents=True)
    (app / "scripts").mkdir()
    response = request_root / "response.json"
    request = request_root / "request.json"
    payload = {
        "schema": control_cst.REQUEST_SCHEMA,
        "schema_version": 1,
        "case_id": "case-001",
        "row": {"sample_id": "case-001", "tree_json": "{}"},
        "application_root": str(app),
        "project_copy": str(project_copy),
        "project_copy_sha256": sha256(project_copy),
        "output_root": str(tmp_path / "attempt"),
        "coordinate_quantum_mm": .01,
        "allow_disconnected_conductor": False,
        "command_timeout_seconds": 60.0,
    }
    request.write_text(json.dumps(payload), encoding="utf-8")

    events = []
    class Project:
        def close(self): events.append("project_closed")
    class Environment:
        def open_project(self, path):
            events.append(("opened", path))
            return Project()
        def close(self): events.append("environment_closed")
    def case_runner(row, **kwargs):
        events.append("ran")
        case, manifest = write_artifacts(Path(kwargs["output_root"]), kwargs["case_id"])
        return SimpleNamespace(case_directory=case, manifest_path=manifest)
    bindings = control_cst._ChildBindings(Environment, case_runner)
    monkeypatch.setenv(control_cst.CONTROL_MODE_ENV, "shore-remote-cst-v1")
    assert control_cst._execute_child_request(request, response, bindings_loader=lambda _root: bindings)
    assert control_cst.CONTROL_MODE_ENV not in os.environ
    assert events[-2:] == ["project_closed", "environment_closed"]
    assert json.loads(response.read_text())["success"] is True


def test_policy_rejects_template_hash_mismatch(tmp_path) -> None:
    value = policy(tmp_path)
    broken = control_cst.CstProcessPolicy(
        value.python_executable, value.application_root, value.project_template, "0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        broken.validated()
