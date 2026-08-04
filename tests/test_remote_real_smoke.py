from __future__ import annotations

import csv
from pathlib import Path

from scripts.automation import antenna_sampler, cst_build_msabp_geometry
from scripts.simulation import run_remote_real_smoke


def test_prepare_input_revalidates_four_persisted_rows(tmp_path: Path) -> None:
    prepared = run_remote_real_smoke.prepare_input(
        tmp_path / "real-smoke.csv",
        candidate_count=4,
    )

    with prepared.csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert prepared.row_count == 4
    assert len(rows) == 4
    assert all(row["geometry_valid"].lower() == "true" for row in rows)
    for row in rows:
        parameters = antenna_sampler.parameters_from_csv_row(row)
        cst_build_msabp_geometry.build_polygon_specs(
            parameters=parameters,
            coordinate_quantum_mm=prepared.coordinate_quantum_mm,
            allow_disconnected_conductor=prepared.allow_disconnected_conductor,
        )


def test_princess_command_is_real_and_selects_both_remote_devices(
    tmp_path: Path,
) -> None:
    prepared = run_remote_real_smoke.PreparedSmokeInput(
        csv_path=tmp_path / "cases.csv",
        resolved_path=tmp_path / "cases.resolved.json",
        row_count=4,
        coordinate_quantum_mm=0.01,
        allow_disconnected_conductor=False,
    )
    command = run_remote_real_smoke.build_princess_command(
        prepared,
        run_id="test-real-smoke",
        device_ids=("convallariag5", "coconutg2"),
        project_template=tmp_path / "msa-bp.cst",
        device_config=tmp_path / "devices.json",
        python_executable="python.exe",
    )

    assert command[2] == "start"
    assert "--dry-run" not in command
    assert command.count("--device") == 2
    assert "convallariag5" in command
    assert "coconutg2" in command
    assert "--project" in command
