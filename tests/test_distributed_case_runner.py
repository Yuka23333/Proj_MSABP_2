from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed import case_runner  # noqa: E402
from scripts.automation import antenna_sampler  # noqa: E402
from scripts.automation import cst_build_msabp_geometry  # noqa: E402
from scripts.automation import cst_run_and_export_s11  # noqa: E402


def _default_csv_row(sample_id: int = 0) -> dict[str, str]:
    row = {
        name: str(spec.code_default)
        for name, spec in antenna_sampler.PARAMETER_REGISTRY.items()
    }
    row.update(
        sample_id=str(sample_id),
        geometry_valid="True",
        geometry_error="",
        final_conductor_components="1",
    )
    return row


def test_parameters_from_csv_row_is_strongly_typed() -> None:
    row = _default_csv_row()
    row["rectangle_length_mm"] = "68.25"
    row["inner_slot_order2_reserved_up_enabled"] = "False"
    row["inner_slot_order2_reserved_down1_enabled"] = "1"

    parameters = antenna_sampler.parameters_from_csv_row(row)

    assert parameters.rectangle_length_mm == pytest.approx(68.25)
    assert parameters.inner_slot_order2_reserved_up_enabled is False
    assert parameters.inner_slot_order2_reserved_down1_enabled is True
    assert isinstance(parameters.rectangle_length_mm, float)

    row["inner_slot_order2_reserved_up_enabled"] = "not-a-boolean"
    with pytest.raises(ValueError, match="must be a boolean"):
        antenna_sampler.parameters_from_csv_row(row)


def test_builder_reuses_existing_project_without_opening_or_saving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = object()
    labels: list[str] = []

    def fail_open(_: str) -> object:
        raise AssertionError("existing-project build must not open CST again")

    monkeypatch.setattr(cst_build_msabp_geometry, "open_cst_project", fail_open)
    monkeypatch.setattr(
        cst_build_msabp_geometry,
        "execute_project_vba",
        lambda received, label, _vba, timeout=None: labels.append(label),
    )
    monkeypatch.setattr(
        cst_build_msabp_geometry,
        "execute_save_project",
        lambda *_args, **_kwargs: pytest.fail("save must be disabled"),
    )

    report = cst_build_msabp_geometry.build_msabp_in_cst(
        project_path=tmp_path / "not-opened.cst",
        project=project,
        save_project=False,
    )

    assert labels[0] == "prepare project"
    assert labels[-1] == "verify geometry"
    assert report.final_conductor_component_count == 1


def test_existing_project_solver_order_and_s11_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    output_path = tmp_path / "S11.csv"

    class FakeModel3D:
        def run_solver(self, timeout=None) -> None:
            assert timeout is None
            events.append("solver")

        def get_tree_items(self, timeout=None) -> tuple[str, ...]:
            return (cst_run_and_export_s11.S11_TREE_PATH,)

    class FakeProject:
        model3d = FakeModel3D()

    def fake_execute(_project, label, vba, timeout=None) -> None:
        if label == "clear previous simulation results":
            events.append("clear")
            return
        if label == "export case S11":
            events.append("s11")
            match = re.search(r'\.FileName "([^"]+)"', vba)
            assert match is not None
            Path(match.group(1)).write_text("1.0 -12.0\n2.0 -8.0\n", encoding="utf-8")

    monkeypatch.setattr(cst_run_and_export_s11, "execute_project_vba", fake_execute)
    monkeypatch.setattr(
        cst_run_and_export_s11,
        "execute_save_project",
        lambda *_args, **_kwargs: pytest.fail("save must be disabled"),
    )

    returned = cst_run_and_export_s11.solve_and_export_s11_on_project(
        FakeProject(),
        output_path,
    )

    assert returned == output_path.resolve()
    assert events == ["clear", "solver", "s11"]
    assert output_path.stat().st_size > 0


def test_recorded_setup_vba_preserves_port_and_monitor_history() -> None:
    cleanup = cst_run_and_export_s11.build_delete_recorded_setup_vba()
    port = cst_run_and_export_s11.build_recorded_port_vba()
    monitor = cst_run_and_export_s11.build_recorded_farfield_monitors_vba()
    solver = cst_run_and_export_s11.build_recorded_solver_parameters_vba()

    assert "Port.Delete 1" in cleanup
    assert cleanup.count("Monitor.Delete") == 51
    assert 'Monitor.Delete "farfield (f=2)"' in cleanup
    assert 'Monitor.Delete "farfield (f=7)"' in cleanup
    assert "Pick.ClearAllPicks" in cleanup
    assert 'Pick.PickEdgeFromId "Connector:ConFace", "30", "22"' in port
    assert '.Coordinates "Picks"' in port
    assert '.Yrange "-8.89", "-8.89"' in port
    assert '.CreateUsingLinearStep "2", "7", "0.1"' in monitor
    assert '.ExportFarfieldSource "True"' in monitor
    assert '.CalculationType "TD-S"' in solver
    assert '.StimulationPort "All"' in solver


def test_restore_recorded_setup_executes_in_history_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []

    class FakeModel3D:
        def get_tree_items(self, timeout=None) -> tuple[str, ...]:
            return (
                cst_run_and_export_s11.RECORDED_PORT_TREE_ITEM,
                *cst_run_and_export_s11.RECORDED_FARFIELD_MONITOR_TREE_ITEMS,
            )

        def get_active_solver_name(self, timeout=None) -> str:
            return "Time Domain Solver"

        def is_solver_running(self, timeout=None) -> bool:
            return False

    class FakeProject:
        model3d = FakeModel3D()

    monkeypatch.setattr(
        cst_run_and_export_s11,
        "execute_project_vba",
        lambda _project, label, _vba, timeout=None: labels.append(label),
    )

    prerequisites = cst_run_and_export_s11.restore_recorded_simulation_setup(
        FakeProject()
    )

    assert labels == [
        "delete recorded port and farfield monitors",
        "recreate connector pick and Port 1",
        "restore solver frequency range",
        "restore Hexahedral FIT mesh",
        "restore time-domain solver acceleration",
        "restore time-domain solver parameters",
        "recreate 2 to 7 GHz farfield monitors",
    ]
    assert prerequisites.ports == (
        cst_run_and_export_s11.RECORDED_PORT_TREE_ITEM,
    )
    assert len(prerequisites.farfield_monitors) == 51


def test_case_runner_dry_run_never_connects_to_cst(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        case_runner.cst_run_and_export_s11,
        "open_cst_project",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not connect to CST"),
    )
    stages: list[str] = []

    result = case_runner.run_csv_row(
        _default_csv_row(sample_id=3),
        project_path=tmp_path / "project-does-not-need-to-exist.cst",
        output_root=tmp_path / "results",
        dry_run=True,
        stage_callback=stages.append,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.case_directory.name == "case_0003"
    assert result.s11_path is None
    assert result.farfield_source_path is None
    assert manifest["status"] == "dry_run"
    assert manifest["artifacts"] == {}
    assert manifest["parameters"]["inner_slot_order2_reserved_up_enabled"] is False
    assert stages == ["prechecking_geometry", "writing_manifest", "completed"]


def test_case_runner_builds_solves_copies_ffs_and_hashes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "worker.cst"
    project_path.write_bytes(b"standalone project")
    fake_project = object()
    events: list[str] = []
    stages: list[str] = []
    farfield_source = case_runner.project_farfield_source_path(project_path)

    def fake_build(**kwargs):
        events.append("build")
        assert kwargs["project"] is fake_project
        assert kwargs["save_project"] is False
        _, report = cst_build_msabp_geometry.build_polygon_specs(
            parameters=kwargs["parameters"],
            coordinate_quantum_mm=kwargs["coordinate_quantum_mm"],
            allow_disconnected_conductor=kwargs["allow_disconnected_conductor"],
        )
        return report

    def fake_clear(project, *, timeout):
        assert project is fake_project
        events.append("clear")

    def fake_restore(project, *, timeout):
        assert project is fake_project
        events.append("restore")

    def fake_solve(
        project,
        output_path,
        *,
        overwrite,
        command_timeout,
        save_project,
        clear_results,
        stage_callback,
    ):
        assert project is fake_project
        assert clear_results is False
        for stage, event in (
            ("solving", "solver"),
            ("exporting_s11", "s11"),
        ):
            stage_callback(stage)
            events.append(event)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"1.0 -12.0\n2.0 -8.0\n")
        farfield_source.parent.mkdir(parents=True, exist_ok=True)
        farfield_source.write_bytes(b"farfield-source-data")
        return output_path

    monkeypatch.setattr(
        case_runner.cst_build_msabp_geometry,
        "build_msabp_in_cst",
        fake_build,
    )
    monkeypatch.setattr(
        case_runner.cst_run_and_export_s11,
        "clear_results_on_project",
        fake_clear,
    )
    monkeypatch.setattr(
        case_runner.cst_run_and_export_s11,
        "restore_recorded_simulation_setup",
        fake_restore,
    )
    monkeypatch.setattr(
        case_runner.cst_run_and_export_s11,
        "solve_and_export_s11_on_project",
        fake_solve,
    )

    result = case_runner.run_csv_row(
        _default_csv_row(sample_id=7),
        project_path=project_path,
        output_root=tmp_path / "results",
        project=fake_project,
        stage_callback=stages.append,
    )

    assert events == ["clear", "build", "restore", "solver", "s11"]
    assert result.farfield_source_path is not None
    assert result.farfield_source_path.read_bytes() == b"farfield-source-data"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["artifacts"]["s11"]["sha256"] == case_runner.sha256_file(
        result.s11_path
    )
    assert manifest["artifacts"]["farfield_source"][
        "sha256"
    ] == case_runner.sha256_file(result.farfield_source_path)
    assert stages == [
        "prechecking_geometry",
        "clearing_results",
        "building_geometry",
        "restoring_simulation_setup",
        "solving",
        "exporting_s11",
        "copying_farfield_source",
        "writing_manifest",
        "completed",
    ]


def test_case_runner_rejects_unchanged_stale_farfield_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "worker.cst"
    project_path.write_bytes(b"standalone project")
    farfield_source = case_runner.project_farfield_source_path(project_path)
    farfield_source.parent.mkdir(parents=True)
    farfield_source.write_bytes(b"stale-farfield")

    monkeypatch.setattr(
        case_runner.cst_build_msabp_geometry,
        "build_msabp_in_cst",
        lambda **_kwargs: cst_build_msabp_geometry.build_polygon_specs()[1],
    )
    monkeypatch.setattr(
        case_runner.cst_run_and_export_s11,
        "clear_results_on_project",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        case_runner.cst_run_and_export_s11,
        "restore_recorded_simulation_setup",
        lambda *_args, **_kwargs: None,
    )

    def fake_solve(_project, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"1.0 -12.0\n2.0 -8.0\n")
        return output_path

    monkeypatch.setattr(
        case_runner.cst_run_and_export_s11,
        "solve_and_export_s11_on_project",
        fake_solve,
    )

    with pytest.raises(case_runner.CaseRunError, match="not regenerated") as exc_info:
        case_runner.run_csv_row(
            _default_csv_row(),
            project_path=project_path,
            output_root=tmp_path / "results",
            project=object(),
        )

    assert exc_info.value.stage == "copy_farfield_source"


def test_sampler_invalid_geometry_is_rejected_before_cst(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _default_csv_row(sample_id=5)
    row["geometry_valid"] = "False"
    row["geometry_error"] = "MultiPolygon"
    monkeypatch.setattr(
        case_runner.cst_run_and_export_s11,
        "open_cst_project",
        lambda *_args, **_kwargs: pytest.fail("invalid geometry must not reach CST"),
    )

    with pytest.raises(case_runner.CaseRunError, match="MultiPolygon") as exc_info:
        case_runner.run_csv_row(
            row,
            project_path=tmp_path / "missing.cst",
            output_root=tmp_path / "results",
        )

    assert exc_info.value.stage == "precheck"
