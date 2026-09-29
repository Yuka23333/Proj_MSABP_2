from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from scripts.optimization import morris_tree_common as common
from scripts.postprocessing import analyze_morris_tree as morris


def plan():
    return {
        "schema_version": 1, "campaign_id": "test-morris",
        "variables": [{"name": "x", "lower": 10, "upper": 20},
                      {"name": "y", "lower": -5, "upper": 5}],
        "num_levels": 4, "delta": 2 / 3,
        "source_sha256": {key: hashlib.sha256(key.encode()).hexdigest() for key in morris.SOURCE_KEYS},
        "metrics": {"band_ghz": [3.1, 4.8], "reference_area_mm2": 100.0},
    }


def points():
    coordinates = [[[0, 1], [2 / 3, 1], [2 / 3, 1 / 3]],
                   [[1, 0], [1, 2 / 3], [1 / 3, 2 / 3]]]
    return [{"sample_id": f"morris_t{trajectory:06d}_p{step:02d}",
             "trajectory_id": trajectory, "step_index": step, "unit": unit,
             "request": {"params": {"x": unit[0], "y": unit[1]}, "tree": {}, "build_options": {}},
             "manufactured_copper_sha256": "c" * 64, "substrate_area_mm2": 100.0}
            for trajectory, units in enumerate(coordinates, 1) for step, unit in enumerate(units)]


def linear_values(rows):
    # Four outputs, with opposite derivative signs. Effects must not be
    # multiplied by variable ranges or flipped on downward Morris steps.
    slopes = np.asarray([[2, -3, .4, 0], [-5, 4, .1, 1]])
    return {row["sample_id"]: dict(zip(morris.METRICS, np.asarray(row["unit"]) @ slopes, strict=True))
            for row in rows}


def write_curve(path, values=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if values is None:
        values = [(3.0, -10), (4.0, -20), (5.0, -10)]
    path.write_text('"Frequency / GHz" "S1,1 / dB"\n--------------------\n'
                    + "".join(f"{frequency:.15g}\t{value:.15g}\n" for frequency, value in values), encoding="utf-8")


def write_result(root, item, frozen, *, offset_db=0):
    folder = root / ("case_" + item["sample_id"])
    folder.mkdir(parents=True)
    artifacts = {}
    for key, filename in [("s11", "S11.csv"), ("rad_eff", "Rad_Eff.csv"), ("tot_eff", "Tot_Eff.csv")]:
        file = folder / filename
        write_curve(file, [(3, -10 + offset_db), (4, -20 + offset_db), (5, -10 + offset_db)])
        artifacts[key] = {"path": filename, "sha256": morris.sha256(file), "size_bytes": file.stat().st_size}
    geometry = {"request": item["request"], "manufactured_copper_sha256": item["manufactured_copper_sha256"],
                "substrate_bounds": [0, 0, 10, 10], "reflector": {"exterior": [], "holes": []}, "quantum_mm": .01}
    file = folder / "geometry_tree.json"
    file.write_text(json.dumps(geometry), encoding="utf-8")
    artifacts["geometry_tree"] = {"path": file.name, "sha256": morris.sha256(file), "size_bytes": file.stat().st_size}
    manifest = {"schema_version": 1, "case_id": item["sample_id"], "simulation_mode": "antenna_tree",
                "status": "completed", "dry_run": False, "parameters": item["request"],
                "source_sha256": frozen["source_sha256"], "manufactured_copper_sha256": item["manufactured_copper_sha256"],
                "tree_request_sha256": hashlib.sha256(json.dumps(item["request"], sort_keys=True).encode()).hexdigest(),
                "geometry_sha256": artifacts["geometry_tree"]["sha256"], "baseline_history_sha256": "b" * 64,
                "artifacts": artifacts}
    file = folder / "manifest.json"
    file.write_text(json.dumps(manifest), encoding="utf-8")
    return file


def freeze_campaign(folder, frozen, rows):
    folder.mkdir(parents=True)
    (folder / "template.cst").write_bytes(b"not a real CST model; no solver is called")
    frozen["template_sha256"] = common.file_hash(folder / "template.cst")
    frozen["plan_sha256"] = common.digest(frozen)
    common.write_json(folder / "plan.json", frozen)
    batch_folder = folder / "batches" / "batch_0001"
    batch_folder.mkdir(parents=True)
    (batch_folder / "sample.csv").write_text("sample_id\n", encoding="utf-8")
    (batch_folder / "candidates.jsonl").write_text("{}\n", encoding="utf-8")
    batch = {"plan_sha256": frozen["plan_sha256"], "points": rows,
             "csv_sha256": common.file_hash(batch_folder / "sample.csv"),
             "candidates_sha256": common.file_hash(batch_folder / "candidates.jsonl")}
    batch["batch_sha256"] = common.digest(batch)
    common.write_json(batch_folder / "batch.json", batch)


def test_analytic_linear_model_signed_actual_steps_and_statistics():
    rows = points()
    values = linear_values(rows)
    expected = [[2, -3, .4, 0], [-5, 4, .1, 1]]
    for trajectory in (rows[:3], rows[3:]):
        np.testing.assert_allclose(morris.elementary_effects(plan(), trajectory, values), expected, atol=1e-14)
    report = morris.analyze(plan(), rows, values, bootstrap_resamples=40)
    assert report["status"] == "complete"
    for row in report["indices"]:
        slope = expected[["x", "y"].index(row["variable"])][morris.METRICS.index(row["metric"])]
        assert row["mu"] == pytest.approx(slope)
        assert row["mu_star"] == pytest.approx(abs(slope))
        assert row["sigma"] == pytest.approx(0, abs=1e-14)
        assert row["mu_ci95_low"] == pytest.approx(slope)
        assert row["mu_ci95_high"] == pytest.approx(slope)
        assert row["n_effective"] == 2
    assert report["indices"][0]["variable"] == "y"


def test_incomplete_never_concatenates_points_and_default_refuses_ranking():
    rows = points()
    values = linear_values(rows)
    del values[rows[1]["sample_id"]]
    with pytest.raises(ValueError, match="Incomplete trajectory"):
        morris.elementary_effects(plan(), rows[:3], values)
    report = morris.analyze(plan(), rows, values, bootstrap_resamples=10)
    assert not report["ranking_available"]
    assert report["indices"] == []
    assert report["valid_point_count"] == 5
    assert report["effective_trajectory_count"] == 1
    assert report["excluded_trajectories"][0]["trajectory_id"] == 1
    conditional = morris.analyze(plan(), rows, values, allow_partial=True, bootstrap_resamples=10)
    assert conditional["status"] == "conditional"
    assert conditional["conditional_on_complete_trajectories"]
    assert conditional["complete_trajectory_ids"] == [2]
    assert all(row["sigma"] is None and row["mu_star_ci95_low"] is None for row in conditional["indices"])


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(1), "D\\+1"),
    (lambda rows: rows.reverse(), "step order"),
    (lambda rows: rows[1].update(step_index=2), "step order"),
    (lambda rows: rows[1].update(trajectory_id=2), "Mixed trajectory"),
    (lambda rows: rows[1].update(unit=[2 / 3, 1 / 3]), "exactly one"),
    (lambda rows: rows[2].update(unit=[0, 1]), "exactly once"),
    (lambda rows: (rows[1].update(unit=[1 / 3, 1]), rows[2].update(unit=[1 / 3, 1 / 3])), "frozen delta"),
    (lambda rows: rows[1].update(unit=[.5, 1]), "off the frozen"),
    (lambda rows: rows[1].update(unit=[2, 1]), "outside"),
    (lambda rows: rows[1].update(unit=[float("nan"), 1]), "nonfinite"),
])
def test_trajectory_shape_order_grid_and_one_at_a_time_validation(mutation, match):
    rows = points()[:3]
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        morris.validate_trajectory(rows, 2, 2 / 3, 4)


def test_bootstrap_reproducible_and_sample_sigma():
    rows = points()
    values = linear_values(rows)
    for point in rows[3:]:
        values[point["sample_id"]] = {key: 2 * value for key, value in values[point["sample_id"]].items()}
    left = morris.analyze(plan(), rows, values, bootstrap_resamples=100, bootstrap_seed=17)
    right = morris.analyze(plan(), rows, values, bootstrap_resamples=100, bootstrap_seed=17)
    assert left == right
    target = next(row for row in left["indices"] if row["variable"] == "x" and row["metric"] == "worst_s11_linear")
    assert target["mu"] == pytest.approx(3)
    assert target["sigma"] == pytest.approx(np.sqrt(2))
    assert target["mu_ci95_low"] <= target["mu"] <= target["mu_ci95_high"]


def test_linear_domain_conversion_trapezoid_and_exact_endpoint_interpolation(tmp_path):
    file = tmp_path / "curve.csv"
    write_curve(file, [(3, 10 * np.log10(.2)), (4, 10 * np.log10(.8)), (5, 10 * np.log10(.4))])
    rad = morris.reduce_curve(file, [3.1, 4.8], efficiency=True)
    expected = (.9 * (.26 + .8) / 2 + .8 * (.8 + .48) / 2) / 1.7
    assert rad["value"] == pytest.approx(expected)
    assert rad["value"] != pytest.approx(10 ** (np.mean([10 * np.log10(.2), 10 * np.log10(.8), 10 * np.log10(.4)]) / 10))
    write_curve(file, [(3, -20), (4, -20), (5, 0)])
    assert morris.reduce_curve(file, [3.1, 4.8], efficiency=False)["value"] == pytest.approx(.82)


def test_unphysical_efficiencies_counted_and_full_band_coverage_required(tmp_path):
    file = tmp_path / "curve.csv"
    write_curve(file, [(3, -10), (4, 1), (5, -10)])
    result = morris.reduce_curve(file, [3.1, 4.8], efficiency=True)
    assert result["value"] == pytest.approx(.1)
    assert result["removed_unphysical_count"] == 1
    assert result["removed_unphysical_in_band_count"] == 1
    write_curve(file, [(3, 1), (4, -10), (5, -10)])
    with pytest.raises(ValueError, match="cover both"):
        morris.reduce_curve(file, [3.1, 4.8], efficiency=True)


@pytest.mark.parametrize("content,match", [
    ("3 -10\n4 nan\n5 -10\n", "Nonfinite"),
    ("3 -10\n4 bad\n5 -10\n", "Malformed"),
    ("3 -10\n4 -10 extra\n5 -10\n", "two columns"),
    ("3 -10\n5 -10\n4 -10\n", "strictly increasing"),
    ("3 -10\n3 -10\n5 -10\n", "strictly increasing"),
    ("3 -10\n", "Fewer than two"),
    ("3 -10\n4 -10\ntruncated\n", "Malformed"),
    ("3 -10\n4 -10\n", "cover both"),
    ("3 -10\n5 99999\n", "Nonfinite linear"),
])
def test_curve_bad_rows_are_not_silently_dropped(tmp_path, content, match):
    file = tmp_path / "curve.csv"
    file.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        morris.reduce_curve(file, [3.1, 4.8], efficiency=True)


def test_result_provenance_success_duplicates_and_conflict(tmp_path):
    frozen, rows = plan(), points()
    first = write_result(tmp_path / "one", rows[0], frozen)
    second = write_result(tmp_path / "two", rows[0], frozen)
    write_result(tmp_path / "unknown", {**rows[0], "sample_id": "not-in-campaign"}, frozen)
    values, issues = morris.collect_results(frozen, rows, [tmp_path, first])
    assert not issues
    assert set(values) == {rows[0]["sample_id"]}
    assert len(values[rows[0]["sample_id"]]["result_manifests"]) == 2
    assert values[rows[0]["sample_id"]]["normalized_substrate_area"] == 1
    assert str(second) in values[rows[0]["sample_id"]]["result_manifests"]
    write_result(tmp_path / "conflict", rows[0], frozen, offset_db=-1)
    values, issues = morris.collect_results(frozen, rows, [tmp_path])
    assert values == {}
    assert any(issue["kind"] == "ambiguous_result" for issue in issues)


@pytest.mark.parametrize("field,value,match", [
    ("schema_version", 2, "schema_version"),
    ("status", "failed", "completed"),
    ("dry_run", True, "non-dry-run"),
    ("dry_run", None, "non-dry-run"),
    ("simulation_mode", "legacy", "mode"),
    ("parameters", {}, "request differs"),
    ("tree_request_sha256", "0" * 64, "request fingerprint"),
    ("source_sha256", {}, "Source provenance"),
    ("manufactured_copper_sha256", "0" * 64, "copper fingerprint"),
    ("geometry_sha256", "0" * 64, "Geometry audit"),
])
def test_result_rejects_provenance_mismatch(tmp_path, field, value, match):
    frozen, rows = plan(), points()
    path = write_result(tmp_path, rows[0], frozen)
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    values, issues = morris.collect_results(frozen, rows, [tmp_path])
    assert values == {}
    assert match in issues[0]["error"]


def test_tampered_artifact_and_traversal_are_rejected(tmp_path):
    frozen, rows = plan(), points()
    path = write_result(tmp_path, rows[0], frozen)
    curve = path.parent / "S11.csv"
    curve.write_text(curve.read_text().replace("-10", "-11"))
    values, issues = morris.collect_results(frozen, rows, [tmp_path])
    assert values == {} and "SHA-256" in issues[0]["error"]
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["s11"]["path"] = "../S11.csv"
    path.write_text(json.dumps(manifest))
    _, issues = morris.collect_results(frozen, rows, [tmp_path])
    assert "Unsafe artifact" in issues[0]["error"]


def test_exact_geometry_reuse_retains_all_rows_and_rejects_bad_reuse(tmp_path):
    frozen, rows = plan(), points()
    identity = {"manufactured_copper_sha256": "c" * 64, "substrate_bounds": [0, 0, 10, 10],
                "reflector": {"exterior": [], "holes": []}, "quantum_mm": .01}
    for row in rows:
        row["simulation_key"] = common.digest(identity)
        row["simulation_sample_id"] = rows[0]["sample_id"]
    write_result(tmp_path, rows[0], frozen)
    values, issues = morris.collect_results(frozen, rows, [tmp_path])
    assert not issues and len(values) == len(rows)
    assert not values[rows[0]["sample_id"]]["reused_simulation"]
    assert values[rows[-1]["sample_id"]]["reused_simulation"]
    rows[-1]["simulation_key"] = "f" * 64
    values, issues = morris.collect_results(frozen, rows, [tmp_path])
    assert rows[-1]["sample_id"] not in values
    assert issues[-1]["kind"] == "invalid_reuse"
    rows[-1]["simulation_key"] = rows[0]["simulation_key"]
    rows[0]["simulation_key"] = "0" * 64
    values, issues = morris.collect_results(frozen, rows, [tmp_path])
    assert values == {}
    assert "Full geometry fingerprint" in issues[0]["error"]


def test_invalid_attempt_does_not_shadow_valid_attempt(tmp_path):
    frozen, rows = plan(), points()
    path = write_result(tmp_path / "failed", rows[0], frozen)
    manifest = json.loads(path.read_text())
    manifest["status"] = "failed"
    path.write_text(json.dumps(manifest))
    write_result(tmp_path / "retry", rows[0], frozen)
    values, issues = morris.collect_results(frozen, rows, [tmp_path])
    assert rows[0]["sample_id"] in values
    assert len(issues) == 1


def test_cli_status_partial_and_stale_rankings_are_cleared(tmp_path):
    frozen, rows = plan(), points()
    campaign = tmp_path / "campaign"
    freeze_campaign(campaign, frozen, rows)
    for row in rows[3:]:
        write_result(campaign / "results", row, frozen)
    assert morris.main(["--campaign", str(campaign), "--bootstrap-resamples", "10"]) == 2
    output = campaign / "analysis"
    report = json.loads((output / "analysis.json").read_text())
    assert report["valid_point_count"] == 3
    assert len(report["valid_point_data"]) == 3
    assert report["indices"] == []
    assert morris.main(["--campaign", str(campaign), "--allow-partial", "--bootstrap-resamples", "10"]) == 0
    assert len((output / "indices.csv").read_text().splitlines()) > 1
    for curve in (campaign / "results").rglob("S11.csv"):
        curve.write_text("broken")
    assert morris.main(["--campaign", str(campaign)]) == 2
    assert len((output / "indices.csv").read_text().splitlines()) == 1
    assert json.loads((output / "status.json").read_text())["ranking_available"] is False


def test_cli_refuses_changed_frozen_campaign_and_clears_stale_ranking(tmp_path):
    frozen, rows = plan(), points()
    campaign = tmp_path / "campaign"
    freeze_campaign(campaign, frozen, rows)
    output = campaign / "analysis"
    output.mkdir()
    (output / "indices.csv").write_text("old ranking")
    file = campaign / "plan.json"
    broken = deepcopy(frozen)
    broken["delta"] = .5
    file.write_text(json.dumps(broken))
    assert morris.main(["--campaign", str(campaign)]) == 2
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "invalid_campaign"
    assert "Frozen plan changed" in status["error"]
    assert "old ranking" not in (output / "indices.csv").read_text()
