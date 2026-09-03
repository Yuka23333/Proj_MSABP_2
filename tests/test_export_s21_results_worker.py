from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.postprocessing import export_s21_results_worker as worker


def test_module_loads_only_standard_library_before_lazy_cst_import() -> None:
    source_path = Path(worker.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots <= {
        "__future__",
        "argparse",
        "collections",
        "csv",
        "importlib",
        "math",
        "os",
        "pathlib",
        "sys",
        "typing",
        "uuid",
    }
    assert "cst" not in imported_roots


def test_load_complex_samples_uses_results_only_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_path = tmp_path / "solved.cst"
    project_path.write_bytes(b"cst")
    calls: dict[str, object] = {}

    class FakeResultItem:
        def get_data(self) -> list[tuple[float, complex, float]]:
            return [(2.0, 0.5 + 0.25j, 50.0), (8.0, -0.25j, 50.0)]

    class FakeResultModule:
        def get_result_item(self, tree_path: str) -> FakeResultItem:
            calls["tree_path"] = tree_path
            return FakeResultItem()

    class FakeProjectFile:
        def __init__(self, path: str, *, allow_interactive: bool) -> None:
            calls["project_path"] = path
            calls["allow_interactive"] = allow_interactive

        def get_3d(self) -> FakeResultModule:
            return FakeResultModule()

    def fake_import_module(name: str) -> SimpleNamespace:
        calls["import_name"] = name
        return SimpleNamespace(ProjectFile=FakeProjectFile)

    monkeypatch.setattr(worker.importlib, "import_module", fake_import_module)
    samples = worker.load_complex_samples(project_path)

    assert samples == [(2.0, 0.5 + 0.25j), (8.0, -0.25j)]
    assert calls == {
        "import_name": "cst.results",
        "project_path": str(project_path.resolve()),
        "allow_interactive": True,
        "tree_path": worker.DEFAULT_TREE_PATH,
    }


def test_write_complex_csv_atomic_uses_existing_schema(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "S21_complex.csv"
    samples = [(2.0, 0.3 + 0.4j), (8.0, -0.5j)]

    result = worker.write_complex_csv_atomic(samples, output_path)

    assert result == output_path.resolve()
    with output_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == worker.CSV_HEADER
    assert len(rows) == 2
    assert float(rows[0]["frequency_ghz"]) == pytest.approx(2.0)
    assert float(rows[0]["s21_real"]) == pytest.approx(0.3)
    assert float(rows[0]["s21_imag"]) == pytest.approx(0.4)
    assert float(rows[0]["s21_magnitude"]) == pytest.approx(0.5)
    assert float(rows[0]["s21_magnitude_db"]) == pytest.approx(20.0 * math.log10(0.5))
    assert float(rows[0]["s21_phase_deg"]) == pytest.approx(math.degrees(math.atan2(0.4, 0.3)))
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))


def test_write_complex_csv_atomic_requires_overwrite_for_existing_output(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "S21_complex.csv"
    output_path.write_text("keep-me\n", encoding="utf-8")
    samples = [(2.0, 0.5 + 0.0j), (8.0, 0.25 + 0.0j)]

    with pytest.raises(FileExistsError, match="--overwrite"):
        worker.write_complex_csv_atomic(samples, output_path)
    assert output_path.read_text(encoding="utf-8") == "keep-me\n"

    worker.write_complex_csv_atomic(samples, output_path, overwrite=True)
    assert output_path.read_text(encoding="utf-8").startswith("frequency_ghz,")


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ([(2.0, 0.5 + 0.0j)], "fewer than two samples"),
        ([(2.0, 0.5 + 0.0j), (2.0, 0.4 + 0.0j)], "not strictly increasing"),
        ([(2.0, 0.5 + 0.0j), (math.nan, 0.4 + 0.0j)], "non-finite"),
        ([(2.0, 0.5 + 0.0j), (8.0, complex(math.inf, 0.0))], "non-finite"),
    ],
)
def test_validate_complex_samples_rejects_invalid_results(
    samples: list[tuple[float, complex]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        worker.validate_complex_samples(samples)


def test_main_returns_nonzero_and_does_not_create_output_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_path = tmp_path / "solved.cst"
    project_path.write_bytes(b"cst")
    output_path = tmp_path / "S21_complex.csv"

    def fail_load(project: str | Path, tree_path: str) -> list[tuple[float, complex]]:
        raise RuntimeError("native result reader failed")

    monkeypatch.setattr(worker, "load_complex_samples", fail_load)
    exit_code = worker.main(
        [
            "--project",
            str(project_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ERROR RuntimeError: native result reader failed" in captured.err
    assert not output_path.exists()


def test_main_exports_requested_tree_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_path = tmp_path / "solved.cst"
    project_path.write_bytes(b"cst")
    output_path = tmp_path / "S21_complex.csv"
    requested_tree_path = r"1D Results\S-Parameters\S2,1 custom"
    calls: dict[str, object] = {}

    def fake_load(project: str | Path, tree_path: str) -> list[tuple[float, complex]]:
        calls["project"] = project
        calls["tree_path"] = tree_path
        return [(2.0, 0.5 + 0.0j), (8.0, 0.25 + 0.0j)]

    monkeypatch.setattr(worker, "load_complex_samples", fake_load)
    exit_code = worker.main(
        [
            "--project",
            str(project_path),
            "--output",
            str(output_path),
            "--tree-path",
            requested_tree_path,
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    assert calls == {"project": project_path, "tree_path": requested_tree_path}
    assert "exported 2 samples" in capsys.readouterr().out
