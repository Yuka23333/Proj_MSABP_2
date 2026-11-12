from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from scripts.simulation import probe_cst_system_solve as probe


class FakeProject:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEnvironment:
    def __init__(self) -> None:
        self.closed = False
        self.project = FakeProject()
        self.opened_path: Path | None = None

    def open_project(self, value: str) -> FakeProject:
        self.opened_path = Path(value)
        return self.project

    def close(self) -> None:
        self.closed = True


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_archive(path: Path, *, unsafe: bool = False) -> Path:
    payload_name = "../escape.py" if unsafe else "src/package.py"
    payload = b"VALUE = 1\n"
    manifest = {
        "schema_version": 1,
        "files": [{
            "path": payload_name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(payload_name, payload)
        archive.writestr("source_manifest.json", json.dumps(manifest))
    return path


def write_curve(path: Path, offset: float = 0.0) -> None:
    path.write_text(f"2.0 {offset + 0.1}\n3.0 {offset + 0.2}\n", encoding="utf-8")


def successful_executor(row, **kwargs):
    assert row["sample_id"] == probe.CASE_ID
    assert kwargs["project"] is not None and kwargs["dry_run"] is False
    callback = kwargs["stage_callback"]
    for stage in (
        "clearing_results", "building_tree_geometry", "checking_simulation_setup",
        "solving", "exporting_1d_results", "copying_farfield_source",
    ):
        callback(stage)
    case = Path(kwargs["output_root"]) / f"case_{probe.CASE_ID}"
    case.mkdir(parents=True)
    write_curve(case / "S11.csv")
    write_curve(case / "Rad_Eff.csv", 0.1)
    write_curve(case / "Tot_Eff.csv", 0.2)
    (case / "Farfield Source [1].ffs").write_bytes(b"fresh-ffs")
    artifacts = {}
    for key, filename in probe.REQUIRED_ARTIFACTS.items():
        item = case / filename
        artifacts[key] = {"path": filename, "size_bytes": item.stat().st_size, "sha256": digest(item)}
    manifest_path = case / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "simulation_mode": "antenna_tree",
        "case_id": probe.CASE_ID, "status": "completed", "dry_run": False,
        "artifacts": artifacts,
    }), encoding="utf-8")
    return SimpleNamespace(case_directory=case, manifest_path=manifest_path)


def test_solve_probe_validates_complete_artifacts_and_closes_everything(tmp_path, monkeypatch):
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-template")
    archive = source_archive(tmp_path / "source.zip")
    environment = FakeEnvironment()
    monkeypatch.setenv("MSABP_CONTROL_MODE", "shore-remote-cst-v1")

    def loader(source_root):
        assert (source_root / "src" / "package.py").is_file()
        return probe.RuntimeBindings(
            environment_factory=lambda: environment,
            case_executor=successful_executor,
            default_request=lambda: {"tree": {}, "params": {}},
            cst_interface_origin="synthetic/cst/interface.py",
        )

    report = probe.run_probe(
        project, archive, tmp_path / "workspace",
        expected_project_sha256=digest(project), expected_archive_sha256=digest(archive),
        runtime_loader=loader, identity_fn=lambda: "nt authority\\system",
        session_id_fn=lambda: 0, process_snapshot_fn=lambda: [],
    )
    assert report["success"] is True
    assert report["identity"] == "nt authority\\system" and report["session_id"] == 0
    assert report["solver_started"] and report["solver_completed"]
    assert set(report["artifacts"]) == {*probe.REQUIRED_ARTIFACTS, "manifest"}
    assert report["curves"]["s11"]["point_count"] == 2
    assert report["source_sha256_before"] == report["source_sha256_after"]
    assert report["control_mode_was_present"] is True
    assert "MSABP_CONTROL_MODE" not in probe.os.environ
    assert environment.project.closed and environment.closed
    assert environment.opened_path == tmp_path / "workspace" / "model" / project.name


def test_solve_failure_is_reported_and_lifecycle_still_closes(tmp_path):
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-template")
    archive = source_archive(tmp_path / "source.zip")
    environment = FakeEnvironment()

    def failed(*_args, **kwargs):
        kwargs["stage_callback"]("solving")
        raise RuntimeError("synthetic solver failure")

    bindings = probe.RuntimeBindings(
        environment_factory=lambda: environment, case_executor=failed,
        default_request=lambda: {}, cst_interface_origin=None,
    )
    report = probe.run_probe(
        project, archive, tmp_path / "workspace",
        expected_project_sha256=None, expected_archive_sha256=None,
        runtime_loader=lambda _root: bindings, process_snapshot_fn=lambda: [],
    )
    assert report["success"] is False
    assert report["solver_started"] is True and report["solver_completed"] is False
    assert report["error_type"] == "RuntimeError"
    assert "synthetic solver failure" in report["error_message"]
    assert environment.project.closed and environment.closed


def test_probe_rejects_stale_workspace_hash_mismatch_and_unsafe_zip(tmp_path):
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-template")
    archive = source_archive(tmp_path / "source.zip")
    stale = tmp_path / "stale"
    stale.mkdir()
    with pytest.raises(FileExistsError):
        probe.run_probe(project, archive, stale, expected_project_sha256=None, expected_archive_sha256=None)
    with pytest.raises(ValueError, match="archive hash"):
        probe.run_probe(
            project, archive, tmp_path / "hash", expected_project_sha256=None,
            expected_archive_sha256="0" * 64,
        )
    unsafe = source_archive(tmp_path / "unsafe.zip", unsafe=True)
    with pytest.raises(ValueError, match="unsafe source archive"):
        probe.run_probe(
            project, unsafe, tmp_path / "unsafe-workspace",
            expected_project_sha256=None, expected_archive_sha256=None,
        )


def test_main_removes_control_mode_before_loading_runtime() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    main = source.index("def main(")
    pop = source.index('os.environ.pop("MSABP_CONTROL_MODE", None)', main)
    call = source.index("run_probe(", main)
    assert pop < call
