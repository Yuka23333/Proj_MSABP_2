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
    row["SLOT_MAIN_LENGTH"] = "48.25"
    row["BRANCH_DOWN_1_K2"] = "0.75"

    parameters = antenna_sampler.parameters_from_csv_row(row)

    assert parameters.SLOT_MAIN_LENGTH == pytest.approx(48.25)
    assert parameters.BRANCH_DOWN_1_K2 == pytest.approx(0.75)
    assert isinstance(parameters.SLOT_MAIN_LENGTH, float)

    row["BRANCH_DOWN_1_K2"] = "1.1"
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
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


def test_builder_uses_builtin_vacuum_without_redefining_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cst_build_msabp_geometry,
        "execute_project_vba",
        lambda _project, label, vba, timeout=None: commands.append((label, vba)),
    )
    monkeypatch.setattr(
        cst_build_msabp_geometry,
        "execute_save_project",
        lambda *_args, **_kwargs: pytest.fail("save must be disabled"),
    )

    report = cst_build_msabp_geometry.build_msabp_in_cst(
        project_path=tmp_path / "not-opened.cst",
        project=object(),
        substrate_material_name="Vacuum",
        save_project=False,
    )

    labels = [label for label, _vba in commands]
    substrate_vba = next(vba for label, vba in commands if label == "extrude substrate")
    assert "define substrate material" not in labels
    assert '.Material "Vacuum"' in substrate_vba
    assert report.substrate_material_name == "Vacuum"
    assert report.substrate_relative_permittivity == pytest.approx(1.0)


def test_case_runner_dry_run_records_vacuum_substrate(tmp_path: Path) -> None:
    row = _default_csv_row(sample_id=2)
    row[case_runner.SUBSTRATE_MATERIAL_COLUMN] = "Vacuum"

    result = case_runner.run_csv_row(
        row,
        project_path=tmp_path / "not-opened.cst",
        output_root=tmp_path / "results",
        dry_run=True,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["geometry"]["substrate_material_name"] == "Vacuum"
    assert manifest["geometry"]["substrate_relative_permittivity"] == pytest.approx(1.0)


def test_existing_project_solver_order_and_standard_1d_exports(
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
            return (
                cst_run_and_export_s11.S11_TREE_PATH,
                cst_run_and_export_s11.RAD_EFF_TREE_PATH,
                cst_run_and_export_s11.TOT_EFF_TREE_PATH,
            )

    class FakeProject:
        model3d = FakeModel3D()

    def fake_execute(_project, label, vba, timeout=None) -> None:
        if label == "clear previous simulation results":
            events.append("clear")
            return
        if label == "export case 1D results":
            events.append("1d-results")
            for tree_path in (
                cst_run_and_export_s11.S11_TREE_PATH,
                cst_run_and_export_s11.RAD_EFF_TREE_PATH,
                cst_run_and_export_s11.TOT_EFF_TREE_PATH,
            ):
                assert f'SelectTreeItem("{tree_path}")' in vba
            matches = re.findall(r'\.FileName "([^"]+)"', vba)
            assert len(matches) == 3
            for exported_path in matches:
                Path(exported_path).write_text(
                    "1.0 -12.0\n2.0 -8.0\n",
                    encoding="utf-8",
                )

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
    assert events == ["clear", "solver", "1d-results"]
    assert output_path.stat().st_size > 0
    assert output_path.with_name("Rad_Eff.csv").stat().st_size > 0
    assert output_path.with_name("Tot_Eff.csv").stat().st_size > 0


def test_recorded_setup_inspection_does_not_mutate_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel3D:
        def get_tree_items(self, timeout=None) -> tuple[str, ...]:
            return (
                cst_run_and_export_s11.RECORDED_PORT_TREE_ITEM,
                *cst_run_and_export_s11.RECORDED_FARFIELD_MONITOR_TREE_ITEMS,
            )

        def get_active_solver_name(self, timeout=None) -> str:
            return cst_run_and_export_s11.EXPECTED_SOLVER_NAME

        def is_solver_running(self, timeout=None) -> bool:
            return False

    class FakeProject:
        model3d = FakeModel3D()

    monkeypatch.setattr(
        cst_run_and_export_s11,
        "execute_project_vba",
        lambda *_args, **_kwargs: pytest.fail("setup inspection must be read-only"),
    )

    prerequisites = cst_run_and_export_s11.inspect_recorded_simulation_setup(
        FakeProject()
    )

    assert prerequisites.ports == (cst_run_and_export_s11.RECORDED_PORT_TREE_ITEM,)
    assert len(prerequisites.farfield_monitors) == 61


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
    assert manifest["parameters"]["BRANCH_DOWN_1_K3"] == pytest.approx(0.0)
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
        _, report = cst_build_msabp_geometry.build_sampled_polygon_specs(
            kwargs["parameters"],
            coordinate_quantum_mm=kwargs["coordinate_quantum_mm"],
        )
        return report

    def fake_clear(project, *, timeout):
        assert project is fake_project
        events.append("clear")

    def fake_inspect(project, timeout):
        assert project is fake_project
        events.append("inspect")

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
            ("exporting_1d_results", "1d-results"),
        ):
            stage_callback(stage)
            events.append(event)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"1.0 -12.0\n2.0 -8.0\n")
        output_path.with_name(case_runner.RAD_EFF_FILENAME).write_bytes(
            b"1.0 -2.0\n2.0 -1.5\n"
        )
        output_path.with_name(case_runner.TOT_EFF_FILENAME).write_bytes(
            b"1.0 -3.0\n2.0 -2.5\n"
        )
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
        "inspect_recorded_simulation_setup",
        fake_inspect,
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

    assert events == ["clear", "build", "inspect", "solver", "1d-results"]
    assert result.farfield_source_path is not None
    assert result.farfield_source_path.read_bytes() == b"farfield-source-data"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["artifacts"]["s11"]["sha256"] == case_runner.sha256_file(
        result.s11_path
    )
    assert result.rad_eff_path is not None
    assert result.tot_eff_path is not None
    assert manifest["artifacts"]["rad_eff"]["sha256"] == case_runner.sha256_file(
        result.rad_eff_path
    )
    assert manifest["artifacts"]["tot_eff"]["sha256"] == case_runner.sha256_file(
        result.tot_eff_path
    )
    assert manifest["artifacts"]["farfield_source"][
        "sha256"
    ] == case_runner.sha256_file(result.farfield_source_path)
    assert stages == [
        "prechecking_geometry",
        "clearing_results",
        "building_geometry",
        "checking_simulation_setup",
        "solving",
        "exporting_1d_results",
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
        "inspect_recorded_simulation_setup",
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
