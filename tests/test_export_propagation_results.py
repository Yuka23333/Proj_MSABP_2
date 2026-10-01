from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.postprocessing import export_propagation_results as exporter


@pytest.mark.parametrize('fields_enabled', [False, True])
def test_optional_field_export(tmp_path, monkeypatch, fields_enabled):
    project = tmp_path / 'solved.cst'
    project.write_bytes(b'cst')
    output = tmp_path / 'out'
    def export_s21(_project, destination, **kwargs):
        _write_valid_s21(destination)
        return 2
    def field_items(*args, **kwargs):
        assert fields_enabled, 'S21-only mode must not query E-field results'
        return [(4.0, 'field at 4 GHz')]
    field = tmp_path / 'field.m3d'
    metadata = tmp_path / 'field.rex'
    field.write_bytes(b'field')
    metadata.write_bytes(b'metadata')
    monkeypatch.setattr(exporter, '_export_complex_s21_in_child', export_s21)
    monkeypatch.setattr(exporter, '_field_tree_items', field_items)
    monkeypatch.setattr(exporter, '_native_field_sources', lambda *args: (field, metadata))
    # Omission keeps the historical E-field-enabled default.
    kwargs = {} if fields_enabled else {'export_e_fields': False}
    report = exporter.export_propagation_results(project, output, **kwargs)
    assert report.s21_sample_count == 2
    assert report.e_field_monitor_count == int(fields_enabled)
    assert len(report.files) == (3 if fields_enabled else 1)
    assert (output / 'e_field_native').exists() == fields_enabled
    assert json.loads((output / 'export_manifest.json').read_text())['export_e_fields'] == fields_enabled


def test_s21_only_rejects_explicit_field_frequencies(tmp_path):
    project = tmp_path / 'solved.cst'
    project.write_bytes(b'cst')
    with pytest.raises(ValueError, match='requires export_e_fields'):
        exporter.export_propagation_results(project, tmp_path / 'out',
                                            export_e_fields=False, field_frequencies_ghz=[4])
    assert exporter.parse_args(['--no-e-fields']).no_e_fields


def _write_valid_s21(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(exporter.S21_CSV_COLUMNS)
        writer.writerow((2.0, 0.1, 0.2, 0.22360679775, -13.0102999566, 63.4349488229))
        writer.writerow((8.0, -0.1, 0.3, 0.31622776602, -10.0, 108.434948823))


def test_s21_export_runs_in_an_isolated_child_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_path = tmp_path / "solved.cst"
    project_path.write_bytes(b"cst")
    output_path = tmp_path / "S21_complex.csv"
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        destination = Path(command[command.index("--output") + 1])
        _write_valid_s21(destination)
        return subprocess.CompletedProcess(command, 0, "worker ok\n", "")

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)

    count = exporter._export_complex_s21_in_child(
        project_path,
        output_path,
        overwrite=True,
        timeout=12.5,
    )

    assert count == 2
    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == [sys.executable, "-I", str(exporter.S21_WORKER_PATH)]
    assert command[command.index("--project") + 1] == str(project_path)
    assert command[command.index("--tree-path") + 1] == exporter.S21_TREE_PATH
    assert command[-1] == "--overwrite"
    assert captured["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 12.5,
    }


def test_s21_child_failure_is_reported_without_importing_results_in_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_path = tmp_path / "solved.cst"
    project_path.write_bytes(b"cst")

    monkeypatch.setattr(
        exporter.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            17,
            "",
            "native worker failed",
        ),
    )

    with pytest.raises(RuntimeError, match=r"exit code 17: native worker failed"):
        exporter._export_complex_s21_in_child(
            project_path,
            tmp_path / "S21_complex.csv",
            overwrite=False,
            timeout=60.0,
        )
