from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
for root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from msabp_opt.simulation.distributed import case_runner  # noqa: E402
from msabp_opt.simulation.distributed.http_api import ApiError  # noqa: E402
from msabp_opt.simulation.distributed.princess import (  # noqa: E402
    PrincessCoordinator,
)
from msabp_opt.simulation.distributed.propagation_case_runner import (  # noqa: E402
    SIMULATION_MODE,
    _transform_second_antenna_vba,
    inspect_propagation_infrastructure,
)
from msabp_opt.simulation.distributed import propagation_case_runner  # noqa: E402
from scripts.automation import cst_build_msabp_geometry  # noqa: E402
from scripts.simulation import prepare_propagation_13  # noqa: E402


def test_propagation_worklist_is_12_medoids_plus_rank_1() -> None:
    rows = prepare_propagation_13.build_rows()

    assert len(rows) == 13
    assert tuple(int(row["candidate_rank"]) for row in rows) == (
        1,
        4,
        8,
        10,
        18,
        19,
        20,
        23,
        30,
        31,
        33,
        34,
        36,
    )
    assert all(row["simulation_mode"] == SIMULATION_MODE for row in rows)
    assert all(int(row["candidate_rank"]) != 35 for row in rows)


def test_propagation_dry_run_dispatches_without_cst(tmp_path: Path) -> None:
    row = prepare_propagation_13.build_rows()[0]
    result = case_runner.run_csv_row(
        row,
        project_path=tmp_path / "not-required.cst",
        output_root=tmp_path / "results",
        dry_run=True,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.simulation_mode == SIMULATION_MODE
    assert result.s21_path is None
    assert manifest["simulation_mode"] == SIMULATION_MODE
    assert manifest["status"] == "dry_run"
    assert manifest["artifacts"] == {}


def test_second_antenna_transform_does_not_touch_infrastructure() -> None:
    vba = _transform_second_antenna_vba()

    assert '.Name "component1_1"' in vba
    assert '.Vector "0", "300", "0"' in vba
    assert vba.count('.MultipleObjects "False"') == 2
    for forbidden in ("Connector", "Port", "Muscle", "Field Monitor"):
        assert forbidden not in vba


def test_propagation_infrastructure_requires_two_ports_and_three_e_fields() -> None:
    class Model3D:
        def get_tree_items(self, timeout=None):
            return (
                r"Ports\port1",
                r"Ports\port2",
                r"Field Monitors\e-field (f=3.1)",
                r"Field Monitors\e-field (f=4)",
                r"Field Monitors\e-field (f=4.8)",
            )

        def get_active_solver_name(self, timeout=None):
            return "HF Time Domain"

        def is_solver_running(self, timeout=None):
            return False

    class Project:
        model3d = Model3D()

    report = inspect_propagation_infrastructure(Project(), 15.0)
    assert report["ports"] == (r"Ports\port1", r"Ports\port2")
    assert len(report["e_field_monitors"]) == 3


def test_princess_accepts_s21_only_for_propagation(tmp_path: Path) -> None:
    s21 = tmp_path / "S21.csv"
    s21.write_bytes(b"frequency_ghz,s21_real\n2,0.1\n")
    manifest = {
        "simulation_mode": SIMULATION_MODE,
        "status": "completed",
        "artifacts": {
            "s21": {
                "path": s21.name,
                "size_bytes": s21.stat().st_size,
                "sha256": case_runner.sha256_file(s21),
            }
        },
    }

    PrincessCoordinator._verify_manifest_artifacts(tmp_path, manifest)

    manifest["simulation_mode"] = case_runner.DEFAULT_SIMULATION_MODE
    with pytest.raises(ApiError, match="missing required artifacts"):
        PrincessCoordinator._verify_manifest_artifacts(tmp_path, manifest)


def test_propagation_case_returns_only_s21_and_keeps_fields_local(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "worker.cst"
    project_path.write_bytes(b"device-local-template-copy")
    row = prepare_propagation_13.build_rows()[0]
    events: list[str] = []

    class Model3D:
        def get_tree_items(self, timeout=None):
            return (
                r"Ports\port1",
                r"Ports\port2",
                r"Field Monitors\e-field (f=3.1)",
                r"Field Monitors\e-field (f=4)",
                r"Field Monitors\e-field (f=4.8)",
            )

        def get_active_solver_name(self, timeout=None):
            return "HF Time Domain"

        def is_solver_running(self, timeout=None):
            return False

        def run_solver(self, timeout=None):
            assert timeout is None
            events.append("solve")

    class Project:
        model3d = Model3D()

    project = Project()

    def fake_build(**kwargs):
        events.append(f"build:{kwargs['component_name']}")
        assert kwargs["project"] is project
        assert kwargs["save_project"] is False
        if kwargs["component_name"] == "component1":
            assert kwargs["define_substrate_material"] is True
        else:
            assert kwargs["define_substrate_material"] is False
        parameters = kwargs["parameters"]
        return cst_build_msabp_geometry.build_sampled_polygon_specs(parameters)[1]

    def fake_export(_project_path, output_directory, **kwargs):
        events.append("export")
        assert kwargs["project"] is project
        output_directory = Path(output_directory)
        (output_directory / "e_field_native").mkdir(parents=True)
        (output_directory / "S21_complex.csv").write_bytes(
            b"frequency_ghz,s21_real,s21_imag\n2,0.1,0.2\n"
        )
        (output_directory / "e_field_native" / "field.m3d").write_bytes(b"field")
        return SimpleNamespace(e_field_monitor_count=3)

    monkeypatch.setattr(
        propagation_case_runner.cst_run_and_export_s11,
        "clear_results_on_project",
        lambda *_args, **_kwargs: events.append("clear"),
    )
    monkeypatch.setattr(
        propagation_case_runner.cst_build_msabp_geometry,
        "build_msabp_in_cst",
        fake_build,
    )
    monkeypatch.setattr(
        propagation_case_runner,
        "execute_project_vba",
        lambda *_args, **_kwargs: events.append("position"),
    )
    monkeypatch.setattr(
        propagation_case_runner.export_propagation_results,
        "export_propagation_results",
        fake_export,
    )

    result = case_runner.run_csv_row(
        row,
        project_path=project_path,
        output_root=tmp_path / "attempt",
        local_artifact_root=tmp_path / "maid-local",
        project=project,
    )

    assert events == [
        "clear",
        "build:component1",
        "build:component1_1",
        "position",
        "solve",
        "export",
    ]
    assert result.s21_path is not None and result.s21_path.is_file()
    assert result.local_e_field_directory is not None
    assert (result.local_e_field_directory / "field.m3d").is_file()
    assert sorted(path.name for path in result.case_directory.iterdir()) == [
        "S21.csv",
        "manifest.json",
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == {"s21"}
    assert manifest["local_only"]["transferred_to_princess"] is False
