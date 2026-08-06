"""Prepare the first 512-candidate Latin-hypercube CST DoE worklist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import antenna_sampler  # noqa: E402
from scripts.automation import check_sampled_curve_intersections  # noqa: E402
from scripts.automation import cst_build_msabp_geometry  # noqa: E402
from scripts.geometry import shapely_antenna_model  # noqa: E402


DEFAULT_ROUND_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "optimization" / "doe_round1_lhs_512.json"
)
ROUND_CONFIG_SCHEMA_VERSION = 1
PARAMETER_COLUMNS = tuple(antenna_sampler.PARAMETER_REGISTRY)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _repository_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    path = (REPOSITORY_ROOT / value).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the repository") from exc
    return path


def load_round_config(path: str | Path = DEFAULT_ROUND_CONFIG_PATH) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    config = _mapping(json.loads(source.read_text(encoding="utf-8")), "round config")
    if int(config.get("schema_version", -1)) != ROUND_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"round config.schema_version must be {ROUND_CONFIG_SCHEMA_VERSION}"
        )
    required = {
        "round_id",
        "base_sampling_config",
        "sampling",
        "eligibility",
        "outputs",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"round config is missing keys: {sorted(missing)}")
    return dict(config)


def _split_pairs(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {item for item in str(value).split(";") if item}


def _initial_rejection_reason(
    row: Mapping[str, Any],
    *,
    allowed_non_bottom_overlap_pairs: set[str],
) -> str:
    if not bool(row["geometry_valid"]):
        return str(row.get("geometry_error", "")).strip() or "invalid curve"
    intersection_pairs = _split_pairs(row.get("non_bottom_intersection_pairs"))
    crossing_pairs = _split_pairs(row.get("non_bottom_crossing_pairs"))
    touching_pairs = _split_pairs(row.get("non_bottom_touching_pairs"))
    overlapping_pairs = _split_pairs(row.get("non_bottom_overlapping_pairs"))
    unexpected = intersection_pairs - allowed_non_bottom_overlap_pairs
    if unexpected:
        return "unexpected non-bottom curve intersection: " + ", ".join(
            sorted(unexpected)
        )
    disallowed_relation = (crossing_pairs | touching_pairs) & (
        allowed_non_bottom_overlap_pairs
    )
    if disallowed_relation:
        return "allowed pair has crossing/touching instead of fixed overlap: " + ", ".join(
            sorted(disallowed_relation)
        )
    missing_overlap = allowed_non_bottom_overlap_pairs - overlapping_pairs
    if missing_overlap:
        return "required fixed curve overlap is missing: " + ", ".join(
            sorted(missing_overlap)
        )
    return ""


def _full_model_preflight(row: Mapping[str, Any]) -> dict[str, float]:
    parameters = antenna_sampler.parameters_from_csv_row(row)
    specs, report = cst_build_msabp_geometry.build_sampled_polygon_specs(parameters)
    expected_labels = (
        "substrate",
        "Patch",
        "Slot",
        "CPW_Feed_Pin",
        "reflector",
        "reflector connector clearance",
    )
    labels = tuple(spec.label for spec in specs)
    if labels != expected_labels:
        raise ValueError(f"unexpected full-model spec order: {labels}")
    substrate_points = specs[0].points
    x_values = [point[0] for point in substrate_points]
    y_values = [point[1] for point in substrate_points]
    return {
        "substrate_width_mm": max(x_values) - min(x_values),
        "substrate_height_mm": max(y_values) - min(y_values),
        "reflector_cutout_width_mm": report.reflector_cutout_width_mm,
        "reflector_cutout_depth_mm": report.reflector_cutout_depth_mm,
    }


def _build_origin_audit_row(
    *,
    allowed_non_bottom_overlap_pairs: set[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    parameters = shapely_antenna_model.DEFAULT_PARAMETERS
    payload = shapely_antenna_model.polygon_export_payload(parameters)
    checks = payload["meta"]["self_intersection_check"]
    invalid_curves = [
        name
        for name, check in checks.items()
        if not (check["ring_is_simple"] and check["polygon_is_valid"])
    ]
    diagnostics = check_sampled_curve_intersections.inspect_curve_boundaries(payload)
    row: dict[str, Any] = {
        "sample_id": "origin",
        "doe_source": "origin",
        **{
            name: float(getattr(parameters, name))
            for name in PARAMETER_COLUMNS
        },
        "geometry_valid": not invalid_curves,
        "geometry_error": (
            "invalid exported polygon(s): " + ", ".join(invalid_curves)
            if invalid_curves
            else ""
        ),
        "final_conductor_components": 1 if not invalid_curves else None,
        **diagnostics,
    }
    reason = _initial_rejection_reason(
        row,
        allowed_non_bottom_overlap_pairs=allowed_non_bottom_overlap_pairs,
    )
    if reason:
        raise ValueError(f"origin does not satisfy Round 1 eligibility: {reason}")
    return row, _full_model_preflight(row)


def _verify_latin_hypercube(
    candidates: pd.DataFrame,
    plan: antenna_sampler.SamplingPlan,
) -> None:
    """Require exactly one candidate in each 1-D LHS stratum."""

    if plan.method != "latin":
        raise ValueError("Latin-hypercube verification requires method='latin'")
    if len(candidates) != plan.n_samples:
        raise ValueError("candidate frame length does not match the sampling plan")
    for item in plan.resolved_parameters:
        if not item.effective_sample:
            continue
        if item.lower is None or item.upper is None:
            raise ValueError(f"sampled parameter {item.spec.name} has no range")
        span = item.upper - item.lower
        strata = {
            min(
                plan.n_samples - 1,
                int((float(value) - item.lower) / span * plan.n_samples),
            )
            for value in candidates[item.spec.name]
        }
        if len(strata) != plan.n_samples:
            raise ValueError(
                f"parameter {item.spec.name} does not occupy every LHS stratum"
            )


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(
            temporary,
            index=False,
            float_format=antenna_sampler.CSV_FLOAT_FORMAT,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_round(
    round_config_path: str | Path = DEFAULT_ROUND_CONFIG_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = load_round_config(round_config_path)
    sampling = _mapping(config["sampling"], "round config.sampling")
    eligibility = _mapping(config["eligibility"], "round config.eligibility")
    method = str(sampling.get("method", "latin")).lower()
    if method != "latin":
        raise ValueError("Round 1 sampling method must be 'latin'")
    candidate_count = int(sampling.get("candidate_count", 512))
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    seed = int(sampling.get("seed", 20260806))
    base_config_path = _repository_path(
        config["base_sampling_config"],
        "round config.base_sampling_config",
    )
    allowed_pairs = {
        str(value)
        for value in eligibility.get("allowed_non_bottom_overlap_pairs", ())
    }

    candidates, intersection_summary = (
        check_sampled_curve_intersections.run_check(
            sample_count=candidate_count,
            config_path=base_config_path,
            method=method,
            seed=seed,
        )
    )
    base_config = antenna_sampler.load_sampling_config(base_config_path)
    plan = antenna_sampler.resolve_sampling_plan(
        base_config,
        n_samples=candidate_count,
        method=method,
        seed=seed,
    )
    _verify_latin_hypercube(candidates, plan)
    candidates["doe_rejection_reason"] = [
        _initial_rejection_reason(
            row,
            allowed_non_bottom_overlap_pairs=allowed_pairs,
        )
        for row in candidates.to_dict(orient="records")
    ]

    model_dimensions: list[dict[str, float]] = []
    for row_index in candidates.index[candidates["doe_rejection_reason"].eq("")]:
        row = candidates.loc[row_index].to_dict()
        try:
            model_dimensions.append(_full_model_preflight(row))
        except Exception as exc:
            candidates.at[row_index, "doe_rejection_reason"] = (
                f"full CST model preflight failed: {type(exc).__name__}: {exc}"
            )

    candidates["doe_eligible"] = candidates["doe_rejection_reason"].eq("")
    candidates["doe_source"] = "lhs"
    accepted_audit = candidates.loc[candidates["doe_eligible"]].copy()
    rejected = candidates.loc[~candidates["doe_eligible"]].copy()
    origin_row, origin_dimensions = _build_origin_audit_row(
        allowed_non_bottom_overlap_pairs=allowed_pairs,
    )
    model_dimensions.append(origin_dimensions)
    accepted_columns = (
        "sample_id",
        "doe_source",
        *PARAMETER_COLUMNS,
        "geometry_valid",
        "geometry_error",
        "final_conductor_components",
    )
    accepted_lhs = accepted_audit.loc[:, accepted_columns].copy()
    origin = pd.DataFrame([origin_row]).loc[:, accepted_columns]
    accepted = pd.concat((accepted_lhs, origin), ignore_index=True)

    reason_counts = Counter(rejected["doe_rejection_reason"].astype(str))
    dimension_summary = (
        {
            name: {
                "min": min(values[name] for values in model_dimensions),
                "max": max(values[name] for values in model_dimensions),
            }
            for name in (
                "substrate_width_mm",
                "substrate_height_mm",
                "reflector_cutout_width_mm",
                "reflector_cutout_depth_mm",
            )
        }
        if model_dimensions
        else {}
    )
    summary = {
        "schema_version": 1,
        "round_id": str(config["round_id"]),
        "method": method,
        "seed": seed,
        "independent_variable_count": len(PARAMETER_COLUMNS),
        "candidate_count": int(len(candidates)),
        "candidate_lhs_stratification_verified": True,
        "accepted_lhs_count": int(len(accepted_lhs)),
        "origin_included": True,
        "origin_sample_id": "origin",
        "accepted_count": int(len(accepted)),
        "worklist_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "lhs_acceptance_fraction": float(len(accepted_lhs) / len(candidates)),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "curve_intersection_summary": intersection_summary,
        "full_cst_model": {
            "source_specs": [
                "substrate",
                "Patch",
                "Slot",
                "CPW_Feed_Pin",
                "reflector",
                "reflector connector clearance",
            ],
            "preflight_passed_count": int(len(model_dimensions)),
            "substrate_material": (
                cst_build_msabp_geometry.DEFAULT_SUBSTRATE_MATERIAL_NAME
            ),
            "substrate_relative_permittivity": (
                cst_build_msabp_geometry.DEFAULT_SUBSTRATE_RELATIVE_PERMITTIVITY
            ),
            "substrate_thickness_mm": (
                cst_build_msabp_geometry.DEFAULT_SUBSTRATE_THICKNESS_MM
            ),
            "dimension_ranges": dimension_summary,
        },
    }
    return accepted, candidates, summary


def write_round_outputs(
    accepted: pd.DataFrame,
    candidates: pd.DataFrame,
    summary: dict[str, Any],
    *,
    round_config_path: str | Path = DEFAULT_ROUND_CONFIG_PATH,
) -> dict[str, Path]:
    config = load_round_config(round_config_path)
    outputs = _mapping(config["outputs"], "round config.outputs")
    paths = {
        name: _repository_path(outputs[name], f"round config.outputs.{name}")
        for name in (
            "accepted_csv",
            "candidates_csv",
            "rejected_csv",
            "summary_json",
        )
    }
    rejected = candidates.loc[~candidates["doe_eligible"]].copy()
    _atomic_write_csv(accepted, paths["accepted_csv"])
    _atomic_write_csv(candidates, paths["candidates_csv"])
    _atomic_write_csv(rejected, paths["rejected_csv"])
    summary["outputs"] = {
        key: {
            "path": str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
            "sha256": _sha256(path),
        }
        for key, path in paths.items()
        if key != "summary_json"
    }
    summary_path = paths["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, summary_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ROUND_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    accepted, candidates, summary = prepare_round(args.config)
    paths = write_round_outputs(
        accepted,
        candidates,
        summary,
        round_config_path=args.config,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[doe-round1] accepted worklist={paths['accepted_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
