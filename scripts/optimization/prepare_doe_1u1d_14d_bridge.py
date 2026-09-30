"""Prepare the 64-case 1U-to-1U1D bridge experiment.

The experiment reuses sixteen completed single-UP cases as exact ``K3=0``
baselines.  Every parent is crossed with the same four active DOWN-branch
settings, producing 64 new antenna-characterization solves.  This blocked
design is intended to measure whether the new branch response can be learned
as a transferable delta instead of fitting a fresh global 14-D surrogate.

F5/default execution writes the solver worklist, a parent-lineage audit, and a
machine-readable plan manifest.  It never starts Princess or CST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import phase2_krvea_data as phase2_data  # noqa: E402
from scripts.automation import antenna_sampler  # noqa: E402
from scripts.geometry import shapely_antenna_model  # noqa: E402


DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "optimization"
    / "doe_1u1d_14d_bridge_64.json"
)
DOWN_PARAMETER_NAMES = (
    "BRANCH_DOWN_1_K",
    "BRANCH_DOWN_1_K2",
    "BRANCH_DOWN_1_K3",
)
OBJECTIVE_COLUMNS = (
    phase2_data.WORST_S11_COLUMN,
    phase2_data.ROI_GAIN_LOSS_DBI_COLUMN,
    phase2_data.NORMALIZED_AREA_COLUMN,
)
DEFAULT_SIMULATION_MODE = "antenna_characterization"
CSV_FLOAT_FORMAT = "%.17g"


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return payload


def _repository_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    candidate = (REPOSITORY_ROOT / value).resolve()
    try:
        candidate.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} leaves the repository: {value}") from exc
    return candidate


def _repository_relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(
            temporary,
            index=False,
            float_format=CSV_FLOAT_FORMAT,
            lineterminator="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _as_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.casefold()
    valid = normalized.isin(("true", "false", "1", "0"))
    if not valid.all():
        raise ValueError(f"{label} contains a non-boolean value")
    return normalized.isin(("true", "1"))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")


def load_completed_observations(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int], Path]:
    """Load and validate the frozen 1U source pool."""

    plan = config["plan"]
    contract = config["source_contract"]
    source_path = _repository_path(
        plan["source_observations"],
        "plan.source_observations",
    )
    frame = pd.read_csv(source_path)
    required = (
        "source",
        "case_id",
        "case_directory",
        "status",
        "is_penalty",
        *phase2_data.ACTIVE_PARAMETER_NAMES,
        *OBJECTIVE_COLUMNS,
    )
    _require_columns(frame, required, "source observations")
    expected_total = int(contract["expected_observation_count"])
    if len(frame) != expected_total:
        raise ValueError(
            f"expected {expected_total} source observations, found {len(frame)}"
        )

    frame = frame.copy()
    frame["source"] = frame["source"].astype(str)
    frame["case_id"] = frame["case_id"].astype(str)
    frame["_identity"] = frame["source"] + "::" + frame["case_id"]
    if frame["_identity"].duplicated().any():
        duplicate = frame.loc[frame["_identity"].duplicated(), "_identity"].iloc[0]
        raise ValueError(f"source observations contain duplicate identity {duplicate}")

    is_penalty = _as_boolean(frame["is_penalty"], "is_penalty")
    completed = frame["status"].astype(str).eq("completed") & ~is_penalty
    expected_completed = int(contract["expected_completed_non_penalty_count"])
    if int(completed.sum()) != expected_completed:
        raise ValueError(
            "expected "
            f"{expected_completed} completed non-penalty observations, "
            f"found {int(completed.sum())}"
        )
    frame = frame.loc[completed].copy()
    frame.sort_values(["source", "case_id"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    numeric_columns = (
        *phase2_data.ACTIVE_PARAMETER_NAMES,
        *OBJECTIVE_COLUMNS,
    )
    for name in numeric_columns:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame.loc[:, numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("completed source observations contain non-finite values")

    base_down = contract["base_down_parameters"]
    for name in DOWN_PARAMETER_NAMES:
        expected = float(base_down[name])
        fixed = float(phase2_data.FIXED_PARAMETER_VALUES[name])
        if not math.isclose(fixed, expected, abs_tol=1.0e-12, rel_tol=0.0):
            raise ValueError(
                f"source contract disagrees with the frozen 11-D space: "
                f"{name}={fixed:g}, expected {expected:g}"
            )

    input_space = phase2_data.authoritative_input_space()
    unit = input_space.normalize(
        frame.loc[:, phase2_data.ACTIVE_PARAMETER_NAMES].to_numpy(dtype=np.float64)
    )
    if np.any(unit < -1.0e-12) or np.any(unit > 1.0 + 1.0e-12):
        raise ValueError("completed source observation lies outside the frozen 11-D space")

    summary = {
        "observation_count": expected_total,
        "completed_non_penalty_count": len(frame),
        "excluded_count": expected_total - len(frame),
    }
    return frame, summary, source_path


def _pairwise_squared_distances(values: np.ndarray) -> np.ndarray:
    norms = np.sum(values * values, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * values @ values.T
    return np.maximum(distances, 0.0)


def _local_surprise_scores(
    x_unit: np.ndarray,
    objectives: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    """Return scale-free leave-one-out local residual magnitudes."""

    if not 1 <= neighbors < len(x_unit):
        raise ValueError("local surprise neighbor count is out of range")
    squared = _pairwise_squared_distances(x_unit)
    np.fill_diagonal(squared, np.inf)
    nearest = np.argpartition(squared, kth=neighbors - 1, axis=1)[:, :neighbors]
    nearest_squared = np.take_along_axis(squared, nearest, axis=1)
    weights = 1.0 / np.maximum(nearest_squared, 1.0e-12)
    neighbor_objectives = objectives[nearest]
    predictions = np.sum(weights[:, :, None] * neighbor_objectives, axis=1)
    predictions /= np.sum(weights, axis=1)[:, None]

    q75, q25 = np.percentile(objectives, (75.0, 25.0), axis=0)
    scale = q75 - q25
    standard_deviation = np.std(objectives, axis=0)
    scale = np.where(scale > 0.0, scale, standard_deviation)
    scale = np.where(scale > 0.0, scale, 1.0)
    residual = (objectives - predictions) / scale
    return np.linalg.norm(residual, axis=1)


def _minimum_distance_to_selected(
    x_unit: np.ndarray,
    candidate: int,
    selected: Sequence[int],
) -> float:
    if not selected:
        return math.inf
    differences = x_unit[np.asarray(selected, dtype=int)] - x_unit[candidate]
    return float(np.sqrt(np.sum(differences * differences, axis=1)).min())


def _choose_farthest(
    x_unit: np.ndarray,
    candidates: Sequence[int],
    selected: Sequence[int],
    identities: Sequence[str],
) -> tuple[int, float]:
    scored = [
        (
            _minimum_distance_to_selected(x_unit, candidate, selected),
            identities[candidate],
            candidate,
        )
        for candidate in candidates
    ]
    maximum = max(item[0] for item in scored)
    tied = [item for item in scored if math.isclose(item[0], maximum, abs_tol=1e-14)]
    _, _, index = min(tied, key=lambda item: item[1])
    return index, maximum


def _nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    mask = np.ones(len(objectives), dtype=bool)
    for index, value in enumerate(objectives):
        dominates = np.all(objectives <= value, axis=1) & np.any(
            objectives < value, axis=1
        )
        dominates[index] = False
        if np.any(dominates):
            mask[index] = False
    return mask


def select_parents(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Select six labeled modes, six maximin, and four surprise parents."""

    selection = config["parent_selection"]
    input_space = phase2_data.authoritative_input_space()
    x_unit = input_space.normalize(
        frame.loc[:, phase2_data.ACTIVE_PARAMETER_NAMES].to_numpy(dtype=np.float64)
    )
    identities = frame["_identity"].astype(str).tolist()
    identity_lookup = {identity: index for index, identity in enumerate(identities)}
    selected: list[int] = []
    annotations: dict[int, dict[str, Any]] = {}

    for item in selection["phase2_objective_modes"]:
        identity = f"{item['source']}::{item['case_id']}"
        if identity not in identity_lookup:
            raise ValueError(f"configured Phase-2 objective-mode parent is absent: {identity}")
        index = identity_lookup[identity]
        if index in selected:
            raise ValueError(f"duplicate configured parent: {identity}")
        selected.append(index)
        annotations[index] = {
            "parent_group": "phase2_propagation_labeled_mode",
            "parent_role": str(item["role"]),
            "selection_distance": math.nan,
            "local_surprise_score": math.nan,
        }

    maximin_count = int(selection["design_space_maximin_count"])
    for _ in range(maximin_count):
        candidates = [index for index in range(len(frame)) if index not in selected]
        index, distance = _choose_farthest(
            x_unit,
            candidates,
            selected,
            identities,
        )
        selected.append(index)
        annotations[index] = {
            "parent_group": "design_space_maximin",
            "parent_role": "11d_farthest_first",
            "selection_distance": distance,
            "local_surprise_score": math.nan,
        }

    surprise_objectives = tuple(selection["local_surprise_objectives"])
    _require_columns(frame, surprise_objectives, "source observations")
    surprise_scores = _local_surprise_scores(
        x_unit,
        frame.loc[:, surprise_objectives].to_numpy(dtype=np.float64),
        int(selection["local_surprise_neighbors"]),
    )
    top_fraction = float(selection["local_surprise_top_fraction"])
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("local_surprise_top_fraction must lie inside (0,1]")
    top_count = max(
        int(selection["local_surprise_count"]),
        int(math.ceil(top_fraction * len(frame))),
    )
    surprise_order = sorted(
        range(len(frame)),
        key=lambda index: (-surprise_scores[index], identities[index]),
    )
    surprise_pool = surprise_order[:top_count]
    for _ in range(int(selection["local_surprise_count"])):
        candidates = [index for index in surprise_pool if index not in selected]
        if not candidates:
            raise ValueError("local-surprise candidate pool was exhausted")
        index, distance = _choose_farthest(
            x_unit,
            candidates,
            selected,
            identities,
        )
        selected.append(index)
        annotations[index] = {
            "parent_group": "local_surprise",
            "parent_role": "12nn_leave_one_out_residual",
            "selection_distance": distance,
            "local_surprise_score": float(surprise_scores[index]),
        }

    expected_parent_count = (
        len(selection["phase2_objective_modes"])
        + maximin_count
        + int(selection["local_surprise_count"])
    )
    if len(selected) != expected_parent_count or len(set(selected)) != len(selected):
        raise RuntimeError("parent selector did not produce the expected unique set")

    parents = frame.iloc[selected].copy().reset_index(drop=True)
    parents.insert(0, "parent_slot", np.arange(1, len(parents) + 1))
    parents.insert(
        1,
        "parent_group",
        [annotations[index]["parent_group"] for index in selected],
    )
    parents.insert(
        2,
        "parent_role",
        [annotations[index]["parent_role"] for index in selected],
    )
    parents["selection_distance"] = [
        annotations[index]["selection_distance"] for index in selected
    ]
    parents["local_surprise_score"] = [
        annotations[index]["local_surprise_score"] for index in selected
    ]
    global_front = _nondominated_mask(
        frame.loc[:, OBJECTIVE_COLUMNS].to_numpy(dtype=np.float64)
    )
    parents["global_nondominated"] = [bool(global_front[index]) for index in selected]
    return parents


def _artifact_path(
    case_directory: Path,
    manifest: Mapping[str, Any],
    artifact_name: str,
    fallback: str,
) -> tuple[Path, str | None]:
    artifacts = manifest.get("artifacts")
    record = artifacts.get(artifact_name) if isinstance(artifacts, Mapping) else None
    relative = record.get("path", fallback) if isinstance(record, Mapping) else fallback
    declared = record.get("sha256") if isinstance(record, Mapping) else None
    return case_directory / str(relative), None if declared is None else str(declared)


def add_parent_lineage(parents: pd.DataFrame) -> pd.DataFrame:
    """Attach portable paths and verified hashes for every reused baseline."""

    result = parents.copy()
    records: list[dict[str, str]] = []
    for row in result.itertuples(index=False):
        case_directory = Path(str(row.case_directory)).resolve()
        try:
            relative_case = case_directory.relative_to(REPOSITORY_ROOT)
        except ValueError as exc:
            raise ValueError(f"parent case directory leaves repository: {case_directory}") from exc
        manifest_path = case_directory / "manifest.json"
        manifest = _read_json_object(manifest_path)
        if str(manifest.get("status")) != "completed":
            raise ValueError(f"parent manifest is not completed: {manifest_path}")
        if str(manifest.get("case_id")) != str(row.case_id):
            raise ValueError(f"parent case identity mismatch: {manifest_path}")
        parameters = manifest.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"parent manifest has no parameter mapping: {manifest_path}")
        for name in DOWN_PARAMETER_NAMES:
            expected = float(phase2_data.FIXED_PARAMETER_VALUES[name])
            actual = float(parameters[name])
            if not math.isclose(actual, expected, abs_tol=1.0e-12, rel_tol=0.0):
                raise ValueError(
                    f"parent is not on the strict 1U slice: {name}={actual:g} "
                    f"in {manifest_path}"
                )

        hashes: dict[str, str] = {}
        for key, fallback in (
            ("s11", "S11.csv"),
            ("farfield_source", phase2_data.FARFIELD_FILENAME),
        ):
            artifact_path, declared = _artifact_path(
                case_directory,
                manifest,
                key,
                fallback,
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            digest = _sha256(artifact_path)
            if declared is not None and digest.casefold() != declared.casefold():
                raise ValueError(f"parent artifact hash mismatch: {artifact_path}")
            hashes[key] = digest
        records.append(
            {
                "parent_case_relpath": relative_case.as_posix(),
                "parent_manifest_sha256": _sha256(manifest_path),
                "parent_s11_sha256": hashes["s11"],
                "parent_ffs_sha256": hashes["farfield_source"],
            }
        )
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def _complete_parameters(parent: Mapping[str, Any]) -> dict[str, float]:
    parameters = {
        name: float(value)
        for name, value in asdict(shapely_antenna_model.DEFAULT_PARAMETERS).items()
    }
    for name in phase2_data.ACTIVE_PARAMETER_NAMES:
        parameters[name] = float(parent[name])
    return parameters


def _checked_vertices(parameters: Mapping[str, float], quantum_mm: float) -> dict[str, Any]:
    typed = antenna_sampler.parameters_from_csv_row(parameters)
    payload = shapely_antenna_model.polygon_export_payload(
        typed,
        quantize_step_mm=quantum_mm,
    )
    invalid = [
        name
        for name, check in payload["meta"]["self_intersection_check"].items()
        if not (check["ring_is_simple"] and check["polygon_is_valid"])
    ]
    if invalid:
        raise ValueError("invalid exported polygon(s): " + ", ".join(invalid))
    return payload["vertices"]


def _down_settings(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    branch = config["down_branch"]
    lower, upper = (float(value) for value in branch["active_range"])
    if not 0.0 < lower < upper <= 1.0:
        raise ValueError("down_branch.active_range must lie inside (0,1]")
    settings: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in branch["settings"]:
        identifier = str(raw["id"])
        if identifier in seen_ids:
            raise ValueError(f"duplicate down setting id {identifier}")
        seen_ids.add(identifier)
        setting = {"id": identifier}
        for name in DOWN_PARAMETER_NAMES:
            value = float(raw[name])
            if not lower <= value <= upper:
                raise ValueError(f"{identifier}.{name} lies outside active range")
            setting[name] = value
        settings.append(setting)
    if not settings:
        raise ValueError("down_branch.settings must not be empty")
    return settings


def build_worklist(
    parents: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Cross each parent with each active DOWN setting and preflight geometry."""

    plan = config["plan"]
    policy = config["geometry_policy"]
    quantum_mm = float(policy["coordinate_quantum_mm"])
    settings = _down_settings(config)
    rows: list[dict[str, Any]] = []
    neutral_checks = 0
    for parent in parents.to_dict(orient="records"):
        base_parameters = _complete_parameters(parent)
        base_vertices = _checked_vertices(base_parameters, quantum_mm)
        for setting in settings:
            neutral_parameters = dict(base_parameters)
            neutral_parameters.update(
                {
                    "BRANCH_DOWN_1_K": setting["BRANCH_DOWN_1_K"],
                    "BRANCH_DOWN_1_K2": setting["BRANCH_DOWN_1_K2"],
                    "BRANCH_DOWN_1_K3": 0.0,
                }
            )
            if bool(policy["require_exact_inactive_down_equivalence"]):
                neutral_vertices = _checked_vertices(neutral_parameters, quantum_mm)
                if neutral_vertices != base_vertices:
                    raise ValueError(
                        "K3=0 does not reproduce the exact parent geometry: "
                        f"parent slot {int(parent['parent_slot'])}, setting {setting['id']}"
                    )
            neutral_checks += 1

            parameters = dict(base_parameters)
            parameters.update({name: float(setting[name]) for name in DOWN_PARAMETER_NAMES})
            _checked_vertices(parameters, quantum_mm)
            parent_slot = int(parent["parent_slot"])
            sample_id = f"bridge_p{parent_slot:02d}_{setting['id']}"
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "simulation_mode": str(plan["simulation_mode"]),
                "doe_source": "1u1d_14d_bridge",
                "topology_id": str(plan["topology_id"]),
                "design_space_dimension": int(plan["effective_dimension"]),
                "parent_slot": parent_slot,
                "parent_group": str(parent["parent_group"]),
                "parent_role": str(parent["parent_role"]),
                "parent_source": str(parent["source"]),
                "parent_case_id": str(parent["case_id"]),
                "parent_case_relpath": str(parent["parent_case_relpath"]),
                "parent_manifest_sha256": str(parent["parent_manifest_sha256"]),
                "down_setting_id": str(setting["id"]),
                "reuses_k3_zero_baseline": True,
            }
            row.update(parameters)
            row.update(
                geometry_valid=True,
                geometry_error="",
                final_conductor_components=1,
            )
            rows.append(row)

    worklist = pd.DataFrame.from_records(rows)
    expected = int(plan["new_solver_evaluations"])
    if len(worklist) != expected:
        raise ValueError(f"configured design produces {len(worklist)} rows, expected {expected}")
    if worklist["sample_id"].duplicated().any():
        raise RuntimeError("bridge sample IDs are not unique")
    return worklist, {
        "active_geometry_preflight_count": len(worklist),
        "active_geometry_valid_count": int(worklist["geometry_valid"].sum()),
        "inactive_equivalence_check_count": neutral_checks,
    }


def _parent_audit_columns() -> tuple[str, ...]:
    return (
        "parent_slot",
        "parent_group",
        "parent_role",
        "source",
        "case_id",
        "parent_case_relpath",
        "parent_manifest_sha256",
        "parent_s11_sha256",
        "parent_ffs_sha256",
        "selection_distance",
        "local_surprise_score",
        "global_nondominated",
        *phase2_data.ACTIVE_PARAMETER_NAMES,
        *OBJECTIVE_COLUMNS,
    )


def prepare_plan(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    overwrite: bool = False,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Prepare and audit the complete bridge plan without launching a solver."""

    config_source = Path(config_path).expanduser().resolve()
    config = _read_json_object(config_source)
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("unsupported bridge-plan schema_version")
    plan = config["plan"]
    if str(plan["simulation_mode"]) != DEFAULT_SIMULATION_MODE:
        raise ValueError("bridge plan must use antenna_characterization mode")
    if int(plan["effective_dimension"]) != 14:
        raise ValueError("1U1D bridge plan must declare a 14-D design space")

    paths = {
        "worklist": _repository_path(plan["output_csv"], "plan.output_csv"),
        "parents": _repository_path(
            plan["parent_audit_csv"],
            "plan.parent_audit_csv",
        ),
        "manifest": _repository_path(
            plan["plan_manifest_json"],
            "plan.plan_manifest_json",
        ),
        "project_template": _repository_path(
            plan["project_template"],
            "plan.project_template",
        ),
    }
    existing = [path for key, path in paths.items() if key != "project_template" and path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"bridge-plan output already exists: {existing[0]}")
    if not paths["project_template"].is_file():
        raise FileNotFoundError(paths["project_template"])

    observations, source_summary, source_path = load_completed_observations(config)
    parents = add_parent_lineage(select_parents(observations, config))
    worklist, preflight = build_worklist(parents, config)
    parent_audit = parents.loc[:, _parent_audit_columns()].copy()
    _atomic_write_csv(worklist, paths["worklist"])
    _atomic_write_csv(parent_audit, paths["parents"])

    settings = _down_settings(config)
    windows_output_csv = str(plan["output_csv"]).replace("/", "\\")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": str(plan["plan_id"]),
        "title": str(plan["title"]),
        "purpose": (
            "paired base-plus-delta bridge from the completed 1U slice to the "
            "active 1U1D topology"
        ),
        "simulation_mode": DEFAULT_SIMULATION_MODE,
        "topology": {
            "base": str(plan["base_topology_id"]),
            "active": str(plan["topology_id"]),
            "base_dimension": len(phase2_data.ACTIVE_PARAMETER_NAMES),
            "new_branch_dimension": len(DOWN_PARAMETER_NAMES),
            "effective_dimension": int(plan["effective_dimension"]),
        },
        "evaluation_accounting": {
            "new_solver_evaluations": len(worklist),
            "reused_k3_zero_baselines": len(parents),
            "analysis_rows_after_pairing": len(worklist) + len(parents),
            "parent_count": len(parents),
            "active_down_settings_per_parent": len(settings),
        },
        "source_pool": {
            **source_summary,
            "path": _repository_relative(source_path),
            "sha256": _sha256(source_path),
        },
        "selection": {
            "phase2_propagation_labeled_mode_count": int(
                parents["parent_group"].eq("phase2_propagation_labeled_mode").sum()
            ),
            "design_space_maximin_count": int(
                parents["parent_group"].eq("design_space_maximin").sum()
            ),
            "local_surprise_count": int(
                parents["parent_group"].eq("local_surprise").sum()
            ),
            "holdout_rule": "split by parent_slot, never by individual child row",
        },
        "down_branch": {
            "inactive_contract": "K3=0; K1 and K2 are conditionally inactive",
            "active_range": list(config["down_branch"]["active_range"]),
            "design_type": str(config["down_branch"]["design_type"]),
            "seed": int(config["down_branch"]["seed"]),
            "shared_settings": settings,
        },
        "preflight": preflight,
        "inputs": {
            "configuration": {
                "path": _repository_relative(config_source),
                "sha256": _sha256(config_source),
            },
            "preparation_script": {
                "path": _repository_relative(Path(__file__)),
                "sha256": _sha256(Path(__file__)),
            },
            "project_template": {
                "path": _repository_relative(paths["project_template"]),
                "sha256": _sha256(paths["project_template"]),
            },
        },
        "outputs": {
            "worklist": {
                "path": _repository_relative(paths["worklist"]),
                "sha256": _sha256(paths["worklist"]),
                "row_count": len(worklist),
            },
            "parent_audit": {
                "path": _repository_relative(paths["parents"]),
                "sha256": _sha256(paths["parents"]),
                "row_count": len(parent_audit),
            },
        },
        "princess_command": (
            "C:\\Users\\David\\.conda\\envs\\cstpy\\python.exe "
            "scripts\\simulation\\princess.py start "
            f"--csv {windows_output_csv} "
            f"--run-id {plan['plan_id']} "
            "--device convallariag5 --device coconutg2"
        ),
    }
    _atomic_write_json(manifest, paths["manifest"])
    return paths, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing worklist and audit outputs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths, manifest = prepare_plan(args.config, overwrite=args.overwrite)
    accounting = manifest["evaluation_accounting"]
    print(
        f"[Bridge DoE] plan={manifest['plan_id']} topology=1U1D dimensions=14 "
        f"new={accounting['new_solver_evaluations']} "
        f"reused_baselines={accounting['reused_k3_zero_baselines']}"
    )
    print(f"[Bridge DoE] worklist={paths['worklist']}")
    print(f"[Bridge DoE] parents={paths['parents']}")
    print(f"[Bridge DoE] manifest={paths['manifest']}")
    print("[Bridge DoE] CST was not started.")
    print(manifest["princess_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
