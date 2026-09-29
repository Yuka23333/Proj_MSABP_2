"""Audit and analyze frozen tree-geometry Morris campaigns, without CST or SALib.

Effects use signed steps in normalized [0, 1] inputs, not output scaling. RF
curves are interpolated in the linear domain at the exact band endpoints.
Incomplete trajectories never contribute partial effects or penalty values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization import morris_tree_common as common  # noqa: E402


METRICS = (
    "worst_s11_linear",
    "mean_rad_eff_linear",
    "mean_tot_eff_linear",
    "normalized_substrate_area",
)
SOURCE_KEYS = {"builder", "model", "manufacturing", "runner"}
DEFAULT_BOOTSTRAP_SEED = 20260929


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"Invalid SHA-256 for {label}")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported campaign schema_version")
    variables = plan.get("variables", [])
    if not variables or len({v["name"] for v in variables}) != len(variables):
        raise ValueError("Expected nonempty, unique variable names")
    for variable in variables:
        lower, upper = float(variable["lower"]), float(variable["upper"])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid variable bounds: {variable['name']}")
    levels = plan.get("num_levels")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2 or levels % 2:
        raise ValueError("num_levels must be an even integer >= 2")
    if not math.isclose(float(plan["delta"]), levels / (2 * (levels - 1)), abs_tol=1e-12):
        raise ValueError("delta differs from the even-level Morris grid step")
    sources = plan.get("source_sha256", {})
    if set(sources) != SOURCE_KEYS:
        raise ValueError("Expected frozen builder/model/manufacturing/runner source hashes")
    for key, value in sources.items():
        _digest(value, key)
    lower, upper = plan["metrics"]["band_ghz"]
    if not 0 < _positive(lower, "band lower") < _positive(upper, "band upper"):
        raise ValueError("Invalid frequency band")
    _positive(plan["metrics"]["reference_area_mm2"], "reference_area_mm2")


def load_campaign(campaign: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Shared loader verifies the immutable plan, template, batch and CSV/audit
    # hashes. Historical analysis does not require the current checkout to be
    # identical; result manifests must match the frozen source hashes instead.
    plan, points, _ = common.load_campaign(campaign)
    validate_plan(plan)
    _point_index(points)
    if not points:
        raise ValueError("Campaign contains no committed batch points")
    return plan, points


def _point_index(points: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for point in points:
        sample_id = point["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in index:
            raise ValueError(f"Invalid or duplicate planned sample_id: {sample_id}")
        index[sample_id] = point
    return index


def load_db_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read two-column CST ASCII or CSV; never silently drop malformed data.

    CST heading lines before the numeric data, comments and dashed separators
    are allowed. Frequencies must already be strictly increasing in GHz.
    """
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("#", "//")) or set(line) <= {"-", "=", " "}:
            continue
        fields = re.split(r"[,;\s]+", line)
        try:
            frequency = float(fields[0])
        except ValueError as exc:
            # A heading is only allowed before any data; digits after an
            # unreadable first column signal a malformed numeric row.
            header = re.search(r"frequency|freq|s.?1.?1|efficien|ghz", line, re.I)
            if not rows and header:
                continue
            raise ValueError(f"Malformed curve row {line_number}: {path}") from exc
        if len(fields) != 2:
            raise ValueError(f"Expected two columns at row {line_number}: {path}")
        try:
            value = float(fields[1])
        except ValueError as exc:
            raise ValueError(f"Malformed curve row {line_number}: {path}") from exc
        if not math.isfinite(frequency) or not math.isfinite(value) or frequency <= 0:
            raise ValueError(f"Nonfinite/nonpositive frequency or nonfinite dB row {line_number}: {path}")
        rows.append((frequency, value))
    if len(rows) < 2:
        raise ValueError(f"Fewer than two curve samples: {path}")
    array = np.asarray(rows, dtype=float)
    if np.any(np.diff(array[:, 0]) <= 0):
        raise ValueError(f"Frequencies must be strictly increasing and unique: {path}")
    return array[:, 0], array[:, 1]


def band_curve(
    frequency: np.ndarray, linear: np.ndarray, band: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = map(float, band)
    if len(frequency) < 2 or frequency[0] > lower or frequency[-1] < upper:
        raise ValueError("Valid samples do not cover both exact band endpoints; extrapolation is forbidden")
    inside = (frequency > lower) & (frequency < upper)
    grid = np.r_[lower, frequency[inside], upper]
    return grid, np.interp(grid, frequency, linear)


def reduce_curve(path: Path, band: Sequence[float], *, efficiency: bool) -> dict[str, Any]:
    frequency, db = load_db_curve(path)
    with np.errstate(over="ignore", invalid="ignore"):
        linear = np.power(10.0, db / (10.0 if efficiency else 20.0))
    if not np.isfinite(linear).all():
        raise ValueError(f"Nonfinite linear conversion: {path}")
    removed = np.zeros(len(linear), dtype=bool)
    if efficiency:
        removed = (linear < 0) | (linear > 1)
    retained_frequency, retained = frequency[~removed], linear[~removed]
    grid, values = band_curve(retained_frequency, retained, band)
    # np.trapz supports the older NumPy versions used by the solver workers.
    mean = float(np.sum(np.diff(grid) * (values[:-1] + values[1:]) / 2) / (grid[-1] - grid[0]))
    return {
        "value": mean if efficiency else float(values.max()),
        "input_sample_count": len(frequency),
        "valid_sample_count": len(retained),
        "removed_unphysical_count": int(removed.sum()),
        "removed_unphysical_in_band_count": int(np.count_nonzero(
            removed & (frequency >= band[0]) & (frequency <= band[1])
        )),
        "retained_frequency_min_ghz": float(retained_frequency[0]),
        "retained_frequency_max_ghz": float(retained_frequency[-1]),
        "integration_max_gap_ghz": float(np.diff(grid).max()),
    }


def _artifact(case: Path, record: Mapping[str, Any], label: str) -> Path:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe artifact path: {label}")
    path = (case / relative).resolve()
    if not path.is_relative_to(case.resolve()):
        raise ValueError(f"Artifact escapes result directory: {label}")
    expected = _digest(record["sha256"], label)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing/empty artifact: {label}")
    if record.get("size_bytes") is not None and path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"Artifact size mismatch: {label}")
    if sha256(path) != expected:
        raise ValueError(f"Artifact SHA-256 mismatch: {label}")
    return path


def _validated_result(
    plan: Mapping[str, Any], point: Mapping[str, Any], manifest: Mapping[str, Any], path: Path
) -> tuple[dict[str, Any], str]:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema_version")
    if manifest.get("status") != "completed" or manifest.get("dry_run") is not False:
        raise ValueError("Result is not a completed non-dry-run solve")
    if manifest.get("simulation_mode") != "antenna_tree":
        raise ValueError("Wrong simulation mode")
    if manifest.get("parameters") != point["request"]:
        raise ValueError("Manifest request differs from frozen planned request")
    expected_request_hash = hashlib.sha256(json.dumps(point["request"], sort_keys=True).encode()).hexdigest()
    if manifest.get("tree_request_sha256") != expected_request_hash:
        raise ValueError("Tree request fingerprint mismatch")
    if manifest.get("source_sha256") != plan["source_sha256"]:
        raise ValueError("Source provenance differs from frozen campaign")
    expected_copper = _digest(point["manufactured_copper_sha256"], "planned manufactured copper")
    if manifest.get("manufactured_copper_sha256") != expected_copper:
        raise ValueError("Manufactured copper fingerprint mismatch")
    records = manifest["artifacts"]
    paths = {key: _artifact(path.parent, records[key], key) for key in ("s11", "rad_eff", "tot_eff")}
    if "simulation_key" in point and "geometry_tree" not in records:
        raise ValueError("Full geometry audit is required for geometry-reuse provenance")
    if "geometry_tree" in records:
        geometry_path = _artifact(path.parent, records["geometry_tree"], "geometry_tree")
        if manifest.get("geometry_sha256") != records["geometry_tree"]["sha256"]:
            raise ValueError("Geometry audit fingerprint mismatch")
        geometry = _json(geometry_path)
        if geometry.get("request") != point["request"] or geometry.get("manufactured_copper_sha256") != expected_copper:
            raise ValueError("Geometry audit differs from the planned point")
        if "simulation_key" in point:
            identity = {key: geometry[key] for key in (
                "manufactured_copper_sha256", "substrate_bounds", "reflector", "quantum_mm"
            )}
            if common.digest(identity) != point["simulation_key"]:
                raise ValueError("Full geometry fingerprint differs from the planned point")
            x0, y0, x1, y1 = geometry["substrate_bounds"]
            if not math.isclose((x1 - x0) * (y1 - y0), float(point["substrate_area_mm2"]), rel_tol=1e-12):
                raise ValueError("Geometry audit substrate area differs from the planned point")
    band = plan["metrics"]["band_ghz"]
    reduced = {key: reduce_curve(value, band, efficiency=key != "s11") for key, value in paths.items()}
    values = {
        "worst_s11_linear": reduced["s11"]["value"],
        "mean_rad_eff_linear": reduced["rad_eff"]["value"],
        "mean_tot_eff_linear": reduced["tot_eff"]["value"],
        "normalized_substrate_area": _positive(point["substrate_area_mm2"], "substrate area")
        / float(plan["metrics"]["reference_area_mm2"]),
        "curve_audit": reduced,
        "result_manifests": [str(path)],
        "simulation_sample_id": point["sample_id"],
        "reused_simulation": False,
    }
    # Different exported bytes or geometry/baseline provenance are ambiguous,
    # even if rounded scalar metric values happen to be the same.
    fingerprint = json.dumps({
        "artifacts": {key: records[key]["sha256"] for key in paths},
        "geometry_sha256": manifest.get("geometry_sha256"),
        "baseline_history_sha256": manifest.get("baseline_history_sha256"),
    }, sort_keys=True)
    return values, fingerprint


def collect_results(
    plan: Mapping[str, Any], points: Sequence[Mapping[str, Any]], sources: Sequence[str | Path]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return per-point valid metrics and audit issues; unknown IDs are ignored.

    Exact repeated exports are accepted; conflicting valid attempts remove that
    sample from the accepted map. Invalid attempts do not hide a later valid one.
    Aliases retain all Morris rows while reusing a proven identical geometry.
    """
    validate_plan(plan)
    index = _point_index(points)
    values: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    fingerprints = {}
    ambiguous = set()
    manifests: set[Path] = set()
    for source in sources:
        root = Path(source).expanduser().resolve()
        if root.is_file() and root.name == "manifest.json":
            manifests.add(root)
        elif root.is_dir():
            manifests.update(p.resolve() for p in root.rglob("manifest.json"))
        else:
            issues.append({"kind": "missing_source", "sample_id": None, "manifest": str(root), "error": "Result source does not exist"})
    for path in sorted(manifests):
        sample_id = None
        try:
            manifest = _json(path)
            sample_id = manifest.get("case_id", manifest.get("sample_id"))
            if sample_id not in index:
                continue
            point = index[sample_id]
            canonical = point.get("simulation_sample_id", sample_id)
            if canonical != sample_id:
                # A real independently solved alias is intentionally not used:
                # the committed design chooses the canonical geometry result.
                issues.append({"kind": "noncanonical_result", "sample_id": sample_id, "manifest": str(path), "error": f"Expected canonical result {canonical}"})
                continue
            result, fingerprint = _validated_result(plan, point, manifest, path)
            if sample_id in fingerprints and fingerprint != fingerprints[sample_id]:
                ambiguous.add(sample_id)
                values.pop(sample_id, None)
                issues.append({"kind": "ambiguous_result", "sample_id": sample_id, "manifest": str(path), "error": "Conflicting valid result attempts; no automatic selection"})
            elif sample_id not in ambiguous:
                if sample_id in values:
                    values[sample_id]["result_manifests"].append(str(path))
                else:
                    values[sample_id] = result
                    fingerprints[sample_id] = fingerprint
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            issues.append({"kind": "invalid_result", "sample_id": sample_id, "manifest": str(path), "error": str(exc)})
    for sample_id, point in index.items():
        canonical = point.get("simulation_sample_id", sample_id)
        if canonical == sample_id:
            continue
        try:
            original = index[canonical]
            if original.get("simulation_sample_id", canonical) != canonical:
                raise ValueError("Simulation reuse must point directly to a canonical sample")
            geometry_hash = _digest(point["simulation_key"], "reuse geometry")
            if geometry_hash != original.get("simulation_key"):
                raise ValueError("Reused full-geometry fingerprint mismatch")
            if point["manufactured_copper_sha256"] != original["manufactured_copper_sha256"]:
                raise ValueError("Reused copper fingerprint mismatch")
            if point["substrate_area_mm2"] != original["substrate_area_mm2"]:
                raise ValueError("Reused substrate area mismatch")
            if canonical in values:
                values[sample_id] = {**values[canonical], "reused_simulation": True}
        except (ValueError, KeyError, TypeError) as exc:
            issues.append({"kind": "invalid_reuse", "sample_id": sample_id, "manifest": None, "error": str(exc)})
    return values, issues


def validate_trajectory(
    points: Sequence[Mapping[str, Any]], dimensions: int, delta: float, num_levels: int
) -> np.ndarray:
    if len(points) != dimensions + 1:
        raise ValueError(f"Trajectory needs exactly D+1={dimensions + 1} points")
    if [p["step_index"] for p in points] != list(range(dimensions + 1)):
        raise ValueError("Trajectory step order must be exactly 0..D; rows are not reordered")
    if len({p["trajectory_id"] for p in points}) != 1:
        raise ValueError("Mixed trajectory IDs")
    unit = np.asarray([p["unit"] for p in points], dtype=float)
    if unit.shape != (dimensions + 1, dimensions) or not np.isfinite(unit).all():
        raise ValueError("Invalid normalized input dimensions or nonfinite coordinates")
    if np.any(unit < -1e-12) or np.any(unit > 1 + 1e-12):
        raise ValueError("Normalized inputs outside [0,1]")
    if not np.allclose(unit * (num_levels - 1), np.round(unit * (num_levels - 1)), atol=1e-10, rtol=0):
        raise ValueError("Normalized input is off the frozen Morris grid")
    steps = np.diff(unit, axis=0)
    changed = np.abs(steps) > 1e-12
    if not np.all(changed.sum(axis=1) == 1):
        raise ValueError("Each step must change exactly one input")
    if not np.all(changed.sum(axis=0) == 1):
        raise ValueError("Each input must change exactly once per trajectory")
    if not np.allclose(np.abs(steps[changed]), delta, atol=1e-10, rtol=0):
        raise ValueError("Actual normalized step differs from frozen delta")
    return steps


def elementary_effects(
    plan: Mapping[str, Any], points: Sequence[Mapping[str, Any]], values: Mapping[str, Mapping[str, Any]]
) -> np.ndarray:
    """One complete trajectory, returning [input, metric] signed effects."""
    dimensions = len(plan["variables"])
    steps = validate_trajectory(points, dimensions, float(plan["delta"]), plan["num_levels"])
    missing = [p["sample_id"] for p in points if p["sample_id"] not in values]
    if missing:
        raise ValueError(f"Incomplete trajectory; missing valid points: {missing}")
    outputs = np.asarray([[values[p["sample_id"]][metric] for metric in METRICS] for p in points], dtype=float)
    if not np.isfinite(outputs).all():
        raise ValueError("Nonfinite metric values; no effects may be calculated")
    effects = np.empty((dimensions, len(METRICS)))
    for step, difference in zip(steps, np.diff(outputs, axis=0), strict=True):
        dimension = int(np.flatnonzero(np.abs(step) > 1e-12)[0])
        effects[dimension] = difference / step[dimension]
    return effects


def analyze(
    plan: Mapping[str, Any], points: Sequence[Mapping[str, Any]],
    values: Mapping[str, Mapping[str, Any]], *, allow_partial: bool = False,
    bootstrap_resamples: int = 2000, bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    validate_plan(plan)
    _point_index(points)
    if bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be >= 2")
    trajectories: dict[Any, list[Mapping[str, Any]]] = {}
    for point in points:
        trajectories.setdefault(point["trajectory_id"], []).append(point)
    effects, complete_ids, excluded = [], [], []
    for trajectory_id, members in trajectories.items():
        try:
            effect = elementary_effects(plan, members, values)
        except (ValueError, KeyError, TypeError) as exc:
            excluded.append({"trajectory_id": trajectory_id, "reason": str(exc),
                             "missing_sample_ids": [p["sample_id"] for p in members if p["sample_id"] not in values]})
        else:
            complete_ids.append(trajectory_id)
            effects.append(effect)
    available = bool(effects) and (allow_partial or not excluded)
    conditional = bool(excluded and allow_partial and effects)
    report: dict[str, Any] = {
        "schema_version": 1, "campaign_id": plan.get("campaign_id"),
        "analysis_source_sha256": common.source_hash(__file__),
        "plan_sha256": plan.get("plan_sha256"),
        "source_sha256": dict(plan["source_sha256"]),
        "auxiliary_source_sha256": dict(plan.get("auxiliary_source_sha256", {})),
        "status": "conditional" if conditional else ("complete" if available else "incomplete"),
        "ranking_available": available, "allow_partial": allow_partial,
        "conditional_on_complete_trajectories": conditional,
        "planned_point_count": len(points), "valid_point_count": sum(p["sample_id"] in values for p in points),
        "planned_trajectory_count": len(trajectories), "effective_trajectory_count": len(effects),
        "complete_trajectory_ids": complete_ids, "excluded_trajectories": excluded,
        "missing_sample_ids": [p["sample_id"] for p in points if p["sample_id"] not in values],
        "method": {
            "sampling_scope": plan.get("sampling", "frozen committed campaign trajectories"),
            "effects": "(y_next-y_current)/(x_next-x_current), normalized inputs, no output scaling",
            "sigma": "sample standard deviation (ddof=1); null with fewer than two trajectories",
            "confidence": "95% percentile bootstrap, entire trajectories resampled together; null if n<2",
            "bootstrap_resamples": bootstrap_resamples, "bootstrap_seed": bootstrap_seed,
            "metrics": {"worst_s11_linear": "max 10**(S11_dB/20)",
                        "mean_rad_eff_linear": "trapezoid band mean of 10**(Rad_Eff_dB/10)",
                        "mean_tot_eff_linear": "trapezoid band mean of 10**(Tot_Eff_dB/10)",
                        "normalized_substrate_area": "frozen substrate_area_mm2 / reference_area_mm2"},
            "band_ghz": list(plan["metrics"]["band_ghz"]),
            "band_coverage": "retained samples bracket both exact endpoints, >=2 samples; no extrapolation",
            "efficiency_filter": "discard linear samples outside [0,1], count removals; nonfinite data invalidates result",
            "interpretation": "mu_star ranks sensitivity within each output only; sigma may indicate nonlinearity or interactions, not a causal decomposition",
        },
        "indices": [], "elementary_effects": [],
    }
    if not available:
        return report
    matrix = np.asarray(effects)
    mu = matrix.mean(axis=0)
    mu_star = np.abs(matrix).mean(axis=0)
    sigma = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else None
    intervals = {}
    if len(matrix) > 1:
        rng = np.random.default_rng(bootstrap_seed)
        samples = {name: [] for name in ("mu", "mu_star", "sigma")}
        for _ in range(bootstrap_resamples):
            # A single draw is shared across all inputs/outputs; pairing is kept.
            selected = matrix[rng.integers(0, len(matrix), size=len(matrix))]
            samples["mu"].append(selected.mean(axis=0))
            samples["mu_star"].append(np.abs(selected).mean(axis=0))
            samples["sigma"].append(selected.std(axis=0, ddof=1))
        intervals = {key: np.percentile(value, [2.5, 97.5], axis=0) for key, value in samples.items()}
    for metric_index, metric in enumerate(METRICS):
        ordering = sorted(range(len(plan["variables"])), key=lambda j: (-mu_star[j, metric_index], j))
        for rank, dimension in enumerate(ordering, 1):
            row = {"metric": metric, "variable": plan["variables"][dimension]["name"],
                   "rank": rank, "n_effective": len(matrix), "conditional": conditional,
                   "mu": float(mu[dimension, metric_index]), "mu_star": float(mu_star[dimension, metric_index]),
                   "sigma": float(sigma[dimension, metric_index]) if sigma is not None else None}
            for statistic in ("mu", "mu_star", "sigma"):
                for endpoint, endpoint_index in (("low", 0), ("high", 1)):
                    row[f"{statistic}_ci95_{endpoint}"] = (
                        float(intervals[statistic][endpoint_index, dimension, metric_index]) if intervals else None
                    )
            report["indices"].append(row)
    for trajectory_id, effect in zip(complete_ids, matrix, strict=True):
        for dimension, variable in enumerate(plan["variables"]):
            report["elementary_effects"].append({
                "trajectory_id": trajectory_id, "variable": variable["name"],
                **{metric: float(effect[dimension, j]) for j, metric in enumerate(METRICS)},
            })
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(
    output: Path, report: dict[str, Any], points: Sequence[Mapping[str, Any]],
    values: Mapping[str, Mapping[str, Any]], issues: Sequence[Mapping[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report["result_issues"] = list(issues)
    report["valid_point_data"] = dict(values)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    status = {key: value for key, value in report.items() if key not in {"indices", "elementary_effects", "valid_point_data"}}
    (output / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    rows = [{"sample_id": p["sample_id"], "trajectory_id": p["trajectory_id"], "step_index": p["step_index"],
             "simulation_sample_id": p.get("simulation_sample_id", p["sample_id"]),
             "status": "valid" if p["sample_id"] in values else "missing_or_invalid",
             **{metric: values.get(p["sample_id"], {}).get(metric) for metric in METRICS}} for p in points]
    _write_csv(output / "samples.csv", rows, ["sample_id", "trajectory_id", "step_index", "simulation_sample_id", "status", *METRICS])
    _write_csv(output / "elementary_effects.csv", report["elementary_effects"], ["trajectory_id", "variable", *METRICS])
    index_fields = ["metric", "variable", "rank", "n_effective", "conditional", "mu", "mu_star", "sigma"]
    index_fields += [f"{stat}_ci95_{endpoint}" for stat in ("mu", "mu_star", "sigma") for endpoint in ("low", "high")]
    # Rewrite even when blocked so an old ranking cannot masquerade as current.
    _write_csv(output / "indices.csv", report["indices"], index_fields)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", help="Recursive result source; repeat for several devices/attempts")
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly rank only complete trajectories, conditional on completion")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    campaign = args.campaign.expanduser().resolve()
    try:
        plan, points = load_campaign(campaign)
        values, issues = collect_results(plan, points, args.source or [campaign / "results"])
        report = analyze(plan, points, values, allow_partial=args.allow_partial,
                         bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.bootstrap_seed)
        write_analysis(campaign / "analysis", report, points, values, issues)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        output = campaign / "analysis"
        # Also invalidate old ranking artifacts on a failed campaign audit.
        write_analysis(output, {"status": "invalid_campaign", "ranking_available": False,
                                "error": str(exc), "indices": [], "elementary_effects": []}, [], {}, [])
        print(f"Morris analysis refused: {exc}")
        return 2
    print(f"Morris analysis: {report['status']}; {report['effective_trajectory_count']}/{report['planned_trajectory_count']} complete trajectories; {campaign / 'analysis'}")
    return 0 if report["ranking_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
