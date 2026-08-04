"""Sample every independent antenna parameter from one strict JSON config.

The generated parameter rows are checked by rebuilding the six individual
polygon curves that CST will receive.  Coordinates are quantized before each
curve is checked for collapsed edges, self-intersection, winding, containment,
and the final Boolean topology.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import cst_build_msabp_geometry  # noqa: E402
from scripts.geometry import antenna_outline  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "optimization" / "antenna_sampling.json"
)
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "data" / "samples" / "antenna_samples.csv"

BRANCH_FIELDS = {
    "reserved_up_1": {
        "enabled": "inner_slot_order2_reserved_up_enabled",
        "parameters": (
            "inner_slot_order2_reserved_up_anchor_t",
            "inner_slot_order2_reserved_up_length_mm",
            "inner_slot_order2_reserved_up_width_mm",
        ),
    },
    "reserved_down_1": {
        "enabled": "inner_slot_order2_reserved_down1_enabled",
        "parameters": (
            "inner_slot_order2_reserved_down1_anchor_t",
            "inner_slot_order2_reserved_down1_length_mm",
            "inner_slot_order2_reserved_down1_width_mm",
        ),
    },
    "reserved_down_2": {
        "enabled": "inner_slot_order2_reserved_down2_enabled",
        "parameters": (
            "inner_slot_order2_reserved_down2_anchor_t",
            "inner_slot_order2_reserved_down2_length_mm",
            "inner_slot_order2_reserved_down2_width_mm",
        ),
    },
}

PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "outline": (
        "rectangle_length_mm",
        "rectangle_width_mm",
    ),
    "outer.upper.order1": (
        "upper_outer_slot_order1_width_mm",
        "upper_outer_slot_order1_bottom_y_mm",
    ),
    "outer.upper.order2": (
        "upper_outer_slot_order2_lower_y_mm",
        "upper_outer_slot_order2_inward_extension_mm",
    ),
    "outer.lower.order1": (
        "lower_outer_slot_order1_opposite_corner_x_mm",
        "lower_outer_slot_order1_opposite_corner_y_mm",
    ),
    "outer.lower.order2.branch1": (
        "lower_outer_slot_order2_branch1_inner_x_mm",
        "lower_outer_slot_order2_branch1_lower_y_mm",
        "lower_outer_slot_order2_branch1_upper_y_mm",
    ),
    "outer.lower.order2.branch2": (
        "lower_outer_slot_order2_branch2_inner_x_mm",
        "lower_outer_slot_order2_branch2_lower_y_mm",
        "lower_outer_slot_order2_branch2_upper_y_mm",
    ),
    "outer.symmetry": ("outer_slot_symmetry_axis_x_mm",),
    "inner.order1": (
        "inner_slot_order1_left_x_mm",
        "inner_slot_order1_right_x_mm",
        "inner_slot_order1_lower_y_mm",
        "inner_slot_order1_upper_y_mm",
    ),
    "inner.order2.primary": (
        "inner_slot_order2_cap_left_x_mm",
        "inner_slot_order2_cap_right_x_mm",
        "inner_slot_order2_cap_y_mm",
    ),
    "inner.order2.reserved.up.control": (
        "inner_slot_order2_reserved_up_enabled",
    ),
    "inner.order2.reserved.up.position": (
        "inner_slot_order2_reserved_up_anchor_t",
    ),
    "inner.order2.reserved.up.size": (
        "inner_slot_order2_reserved_up_length_mm",
        "inner_slot_order2_reserved_up_width_mm",
    ),
    "inner.order2.reserved.down1.control": (
        "inner_slot_order2_reserved_down1_enabled",
    ),
    "inner.order2.reserved.down1.position": (
        "inner_slot_order2_reserved_down1_anchor_t",
    ),
    "inner.order2.reserved.down1.size": (
        "inner_slot_order2_reserved_down1_length_mm",
        "inner_slot_order2_reserved_down1_width_mm",
    ),
    "inner.order2.reserved.down2.control": (
        "inner_slot_order2_reserved_down2_enabled",
    ),
    "inner.order2.reserved.down2.position": (
        "inner_slot_order2_reserved_down2_anchor_t",
    ),
    "inner.order2.reserved.down2.size": (
        "inner_slot_order2_reserved_down2_length_mm",
        "inner_slot_order2_reserved_down2_width_mm",
    ),
    "cpw.guide.fixed": (
        "cpw_guide_p1_x_mm",
        "cpw_guide_p1_y_mm",
        "cpw_guide_p2_x_mm",
        "cpw_guide_p2_y_mm",
        "cpw_guide_p3_y_mm",
        "cpw_guide_p7_x_mm",
    ),
    "cpw.guide.design": (
        "cpw_guide_p3_p4_x_mm",
        "cpw_guide_p4_y_mm",
        "cpw_guide_p5_p6_x_mm",
    ),
    "cpw.slot.fixed": (
        "cpw_slot_p0_x_mm",
        "cpw_slot_p0_y_mm",
        "cpw_slot_p1_y_mm",
        "cpw_slot_p5_x_mm",
    ),
    "cpw.slot.design": (
        "cpw_slot_p1_p2_x_mm",
        "cpw_slot_p2_y_mm",
        "cpw_slot_p3_p4_x_mm",
    ),
    "cpw.matching_stub1.fixed": (
        "cpw_matching_stub1_cap_x_mm",
        "cpw_matching_stub1_lower_y_mm",
        "cpw_matching_stub1_upper_y_mm",
    ),
    "cpw.matching_stub2.fixed": (
        "cpw_matching_stub2_cap_x_mm",
        "cpw_matching_stub2_lower_y_mm",
        "cpw_matching_stub2_upper_y_mm",
    ),
    "reflector.clearance.fixed": (
        "reflector_connector_board_thickness_mm",
        "reflector_cutout_width_adjustment_mm",
        "reflector_cutout_depth_mm",
    ),
}

STRUCTURAL_PARAMETER_NAMES = {
    details["enabled"] for details in BRANCH_FIELDS.values()
}
FIXED_BY_DEFAULT_PARAMETER_NAMES = {
    "outer_slot_symmetry_axis_x_mm",
    "inner_slot_order1_left_x_mm",
    *PARAMETER_GROUPS["cpw.guide.fixed"],
    *PARAMETER_GROUPS["cpw.slot.fixed"],
    *PARAMETER_GROUPS["cpw.matching_stub1.fixed"],
    *PARAMETER_GROUPS["cpw.matching_stub2.fixed"],
    *PARAMETER_GROUPS["reflector.clearance.fixed"],
}
ACTIVE_BRANCH_BY_PARAMETER = {
    parameter_name: branch_name
    for branch_name, details in BRANCH_FIELDS.items()
    for parameter_name in details["parameters"]
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    group: str
    unit: str
    kind: str
    sample_default: bool
    code_default: float | bool
    hard_min: float | None = None
    hard_max: float | None = None
    active_if_branch: str | None = None


@dataclass(frozen=True)
class ResolvedParameter:
    spec: ParameterSpec
    nominal: float | bool
    declared_sample: bool
    effective_sample: bool
    lower: float | None
    upper: float | None
    range_source: str | None


@dataclass(frozen=True)
class GeometryPolicy:
    coordinate_quantum_mm: float = antenna_outline.COORDINATE_QUANTUM_MM
    reject_self_intersection: bool = True
    allow_disconnected_conductor: bool = False


@dataclass(frozen=True)
class SamplingPlan:
    method: str
    n_samples: int
    seed: int | None
    parameters: antenna_outline.AntennaOutlineParameters
    resolved_parameters: tuple[ResolvedParameter, ...]
    geometry_policy: GeometryPolicy
    raw_config: Mapping[str, Any]


@dataclass(frozen=True)
class SamplingResult:
    frame: pd.DataFrame
    plan: SamplingPlan


def _group_by_parameter() -> dict[str, str]:
    result: dict[str, str] = {}
    for group, names in PARAMETER_GROUPS.items():
        for name in names:
            if name in result:
                raise RuntimeError(f"parameter {name!r} appears in multiple groups")
            result[name] = group
    return result


def build_parameter_registry() -> dict[str, ParameterSpec]:
    """Build the exhaustive registry for dataclass fields, excluding properties."""

    defaults = antenna_outline.DEFAULT_ANTENNA_PARAMETERS
    dataclass_names = {item.name for item in fields(type(defaults))}
    group_by_parameter = _group_by_parameter()
    missing = dataclass_names - set(group_by_parameter)
    extra = set(group_by_parameter) - dataclass_names
    if missing or extra:
        raise RuntimeError(
            "parameter registry does not match AntennaOutlineParameters: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    registry: dict[str, ParameterSpec] = {}
    for name in sorted(dataclass_names):
        default = getattr(defaults, name)
        if name in STRUCTURAL_PARAMETER_NAMES:
            kind = "structural"
            sample_default = False
            unit = "bool"
        elif name in FIXED_BY_DEFAULT_PARAMETER_NAMES:
            kind = "fixed_by_default"
            sample_default = False
            unit = "ratio" if name.endswith("_t") else "mm"
        else:
            kind = "design"
            sample_default = True
            unit = "ratio" if name.endswith("_t") else "mm"

        hard_min: float | None = None
        hard_max: float | None = None
        if name.endswith("_anchor_t"):
            hard_min, hard_max = 0.0, 1.0
        elif name.startswith("rectangle_"):
            hard_min = antenna_outline.COORDINATE_QUANTUM_MM
        elif name.endswith(("_length_mm", "_width_mm")):
            hard_min = 0.0

        registry[name] = ParameterSpec(
            name=name,
            group=group_by_parameter[name],
            unit=unit,
            kind=kind,
            sample_default=sample_default,
            code_default=default,
            hard_min=hard_min,
            hard_max=hard_max,
            active_if_branch=ACTIVE_BRANCH_BY_PARAMETER.get(name),
        )
    return registry


PARAMETER_REGISTRY = build_parameter_registry()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _reject_unknown_keys(
    mapping: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{label} contains unknown keys: {sorted(unknown)}")


def load_sampling_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load one strict UTF-8 JSON configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config = dict(_require_mapping(config, "configuration root"))
    _reject_unknown_keys(
        config,
        {"schema_version", "sampling", "branches", "geometry_policy"},
        "configuration root",
    )
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got "
            f"{config.get('schema_version')!r}"
        )
    return config


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _parse_range(
    raw_range: Any,
    *,
    nominal: float,
    label: str,
) -> tuple[float, float]:
    data = _require_mapping(raw_range, label)
    mode = data.get("mode")
    if mode == "absolute":
        _reject_unknown_keys(data, {"mode", "min", "max"}, label)
        if "min" not in data or "max" not in data:
            raise ValueError(f"{label} absolute range requires min and max")
        lower = _finite_number(data["min"], f"{label}.min")
        upper = _finite_number(data["max"], f"{label}.max")
    elif mode == "relative":
        _reject_unknown_keys(
            data,
            {"mode", "lower", "upper", "reference"},
            label,
        )
        if "lower" not in data or "upper" not in data:
            raise ValueError(f"{label} relative range requires lower and upper")
        if data.get("reference", "nominal") != "nominal":
            raise ValueError(f"{label}.reference currently supports only 'nominal'")
        if nominal <= 0.0:
            raise ValueError(
                f"{label} cannot be relative because nominal={nominal:g}; "
                "use an absolute range"
            )
        relative_lower = _finite_number(data["lower"], f"{label}.lower")
        relative_upper = _finite_number(data["upper"], f"{label}.upper")
        lower = nominal * (1.0 + relative_lower)
        upper = nominal * (1.0 + relative_upper)
    else:
        raise ValueError(f"{label}.mode must be 'absolute' or 'relative'")
    if not lower < upper:
        raise ValueError(f"{label} requires lower < upper, got {lower:g}..{upper:g}")
    return lower, upper


def _resolve_geometry_policy(config: Mapping[str, Any]) -> GeometryPolicy:
    raw_policy = _require_mapping(config.get("geometry_policy", {}), "geometry_policy")
    _reject_unknown_keys(
        raw_policy,
        {
            "coordinate_quantum_mm",
            "reject_self_intersection",
            "allow_disconnected_conductor",
        },
        "geometry_policy",
    )
    quantum = _finite_number(
        raw_policy.get(
            "coordinate_quantum_mm",
            antenna_outline.COORDINATE_QUANTUM_MM,
        ),
        "geometry_policy.coordinate_quantum_mm",
    )
    if quantum <= 0.0:
        raise ValueError("geometry_policy.coordinate_quantum_mm must be positive")
    reject_self_intersection = raw_policy.get("reject_self_intersection", True)
    allow_disconnected = raw_policy.get("allow_disconnected_conductor", False)
    if not isinstance(reject_self_intersection, bool):
        raise ValueError("geometry_policy.reject_self_intersection must be boolean")
    if not reject_self_intersection:
        raise ValueError(
            "self-intersection rejection cannot be disabled because CST cannot "
            "create a polygon from a self-crossing source curve"
        )
    if not isinstance(allow_disconnected, bool):
        raise ValueError("geometry_policy.allow_disconnected_conductor must be boolean")
    return GeometryPolicy(
        coordinate_quantum_mm=quantum,
        reject_self_intersection=reject_self_intersection,
        allow_disconnected_conductor=allow_disconnected,
    )


def resolve_sampling_plan(
    config: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    method: str | None = None,
    seed: int | None | object = ...,
) -> SamplingPlan:
    """Resolve JSON inheritance independently for every registered parameter."""

    sampling = _require_mapping(config.get("sampling", {}), "sampling")
    _reject_unknown_keys(
        sampling,
        {"method", "n_samples", "seed", "global", "groups", "parameters"},
        "sampling",
    )
    resolved_method = str(method or sampling.get("method", "sobol")).lower()
    if resolved_method not in {"sobol", "latin"}:
        raise ValueError("sampling.method must be 'sobol' or 'latin'")
    resolved_n_samples = int(
        n_samples if n_samples is not None else sampling.get("n_samples", 8)
    )
    if resolved_n_samples <= 0:
        raise ValueError("sampling.n_samples must be positive")
    if seed is ...:
        raw_seed = sampling.get("seed", 20260803)
        resolved_seed = None if raw_seed is None else int(raw_seed)
    else:
        resolved_seed = None if seed is None else int(seed)

    global_config = _require_mapping(sampling.get("global", {}), "sampling.global")
    _reject_unknown_keys(global_config, {"range"}, "sampling.global")
    groups = _require_mapping(sampling.get("groups", {}), "sampling.groups")
    unknown_groups = set(groups) - set(PARAMETER_GROUPS)
    if unknown_groups:
        raise ValueError(f"sampling.groups contains unknown groups: {sorted(unknown_groups)}")
    for group, group_config in groups.items():
        group_mapping = _require_mapping(group_config, f"sampling.groups.{group}")
        _reject_unknown_keys(group_mapping, {"range"}, f"sampling.groups.{group}")

    parameter_overrides = _require_mapping(
        sampling.get("parameters", {}),
        "sampling.parameters",
    )
    unknown_parameters = set(parameter_overrides) - set(PARAMETER_REGISTRY)
    if unknown_parameters:
        raise ValueError(
            "sampling.parameters contains unknown parameters: "
            f"{sorted(unknown_parameters)}"
        )

    branches = _require_mapping(config.get("branches", {}), "branches")
    unknown_branches = set(branches) - set(BRANCH_FIELDS)
    if unknown_branches:
        raise ValueError(f"branches contains unknown names: {sorted(unknown_branches)}")

    value_overrides: dict[str, float | bool] = {}
    for branch_name, details in BRANCH_FIELDS.items():
        branch_config = _require_mapping(
            branches.get(branch_name, {}),
            f"branches.{branch_name}",
        )
        _reject_unknown_keys(branch_config, {"enabled"}, f"branches.{branch_name}")
        enabled = branch_config.get(
            "enabled",
            getattr(
                antenna_outline.DEFAULT_ANTENNA_PARAMETERS,
                str(details["enabled"]),
            ),
        )
        if not isinstance(enabled, bool):
            raise ValueError(f"branches.{branch_name}.enabled must be boolean")
        value_overrides[str(details["enabled"])] = enabled

    for name, raw_override in parameter_overrides.items():
        override = _require_mapping(raw_override, f"sampling.parameters.{name}")
        _reject_unknown_keys(
            override,
            {"value", "sample", "range"},
            f"sampling.parameters.{name}",
        )
        if name in STRUCTURAL_PARAMETER_NAMES:
            raise ValueError(
                f"{name} is structural; configure it through the branches section"
            )
        if "value" in override:
            value_overrides[name] = _finite_number(
                override["value"],
                f"sampling.parameters.{name}.value",
            )

    parameters = replace(
        antenna_outline.DEFAULT_ANTENNA_PARAMETERS,
        **value_overrides,
    )
    resolved_parameters: list[ResolvedParameter] = []
    for name, spec in PARAMETER_REGISTRY.items():
        nominal = getattr(parameters, name)
        if spec.kind == "structural":
            resolved_parameters.append(
                ResolvedParameter(
                    spec=spec,
                    nominal=nominal,
                    declared_sample=False,
                    effective_sample=False,
                    lower=None,
                    upper=None,
                    range_source=None,
                )
            )
            continue

        raw_override = _require_mapping(
            parameter_overrides.get(name, {}),
            f"sampling.parameters.{name}",
        )
        declared_sample = raw_override.get("sample", spec.sample_default)
        if not isinstance(declared_sample, bool):
            raise ValueError(f"sampling.parameters.{name}.sample must be boolean")
        branch_enabled = True
        if spec.active_if_branch is not None:
            enabled_field = str(BRANCH_FIELDS[spec.active_if_branch]["enabled"])
            branch_enabled = bool(getattr(parameters, enabled_field))
        effective_sample = declared_sample and branch_enabled

        lower: float | None = None
        upper: float | None = None
        source: str | None = None
        if effective_sample:
            if "range" in raw_override:
                raw_range = raw_override["range"]
                source = "parameter"
            else:
                group_config = _require_mapping(
                    groups.get(spec.group, {}),
                    f"sampling.groups.{spec.group}",
                )
                if "range" in group_config:
                    raw_range = group_config["range"]
                    source = "group"
                elif "range" in global_config:
                    raw_range = global_config["range"]
                    source = "global"
                else:
                    raise ValueError(
                        f"sampled parameter {name} has no parameter, group, or "
                        "global range"
                    )
            lower, upper = _parse_range(
                raw_range,
                nominal=float(nominal),
                label=f"effective range for {name}",
            )
            if spec.hard_min is not None:
                lower = max(lower, spec.hard_min)
            if spec.hard_max is not None:
                upper = min(upper, spec.hard_max)
            if not lower < upper:
                raise ValueError(
                    f"effective range for {name} is empty after hard bounds"
                )

        resolved_parameters.append(
            ResolvedParameter(
                spec=spec,
                nominal=nominal,
                declared_sample=declared_sample,
                effective_sample=effective_sample,
                lower=lower,
                upper=upper,
                range_source=source,
            )
        )

    return SamplingPlan(
        method=resolved_method,
        n_samples=resolved_n_samples,
        seed=resolved_seed,
        parameters=parameters,
        resolved_parameters=tuple(resolved_parameters),
        geometry_policy=_resolve_geometry_policy(config),
        raw_config=config,
    )


def _sample_unit_cube(plan: SamplingPlan, dimension: int) -> np.ndarray:
    if dimension == 0:
        return np.empty((plan.n_samples, 0), dtype=float)
    if plan.method == "sobol":
        engine = qmc.Sobol(d=dimension, scramble=True, seed=plan.seed)
    else:
        engine = qmc.LatinHypercube(d=dimension, seed=plan.seed)
    return engine.random(n=plan.n_samples)


def _boolean_from_csv_value(value: Any, label: str) -> bool:
    """Parse one structural Boolean without treating non-empty strings as true."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if math.isfinite(number) and number in {0.0, 1.0}:
            return bool(number)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{label} must be a boolean, got {value!r}")


def parameters_from_csv_row(
    row: Mapping[str, Any],
) -> antenna_outline.AntennaOutlineParameters:
    """Return a strongly typed antenna parameter object from one sampler CSV row.

    Sampler CSV files contain metadata columns in addition to the exhaustive
    parameter columns.  Extra columns are intentionally ignored, but every
    registered parameter must be present and non-empty.  In particular,
    structural branch switches are parsed explicitly so the string ``"False"``
    cannot accidentally enable a branch.
    """

    missing = [
        name
        for name in PARAMETER_REGISTRY
        if name not in row or row[name] is None or str(row[name]).strip() == ""
    ]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"sample row is missing parameter columns: {preview}{suffix}")

    values: dict[str, float | bool] = {}
    for name, spec in PARAMETER_REGISTRY.items():
        label = f"sample row parameter {name}"
        if isinstance(spec.code_default, bool):
            values[name] = _boolean_from_csv_value(row[name], label)
            continue

        number = _finite_number(row[name], label)
        if spec.hard_min is not None and number < spec.hard_min:
            raise ValueError(
                f"{label}={number:g} is below hard minimum {spec.hard_min:g}"
            )
        if spec.hard_max is not None and number > spec.hard_max:
            raise ValueError(
                f"{label}={number:g} is above hard maximum {spec.hard_max:g}"
            )
        values[name] = number

    return replace(antenna_outline.DEFAULT_ANTENNA_PARAMETERS, **values)


def _parameters_from_row(
    row: Mapping[str, Any],
) -> antenna_outline.AntennaOutlineParameters:
    """Backward-compatible internal alias for generated, already typed rows."""

    return parameters_from_csv_row(row)


def generate_samples(plan: SamplingPlan) -> SamplingResult:
    """Generate full parameter rows and validate the quantized CST curves."""

    sampled = [item for item in plan.resolved_parameters if item.effective_sample]
    unit_samples = _sample_unit_cube(plan, len(sampled))
    lows = np.asarray([item.lower for item in sampled], dtype=float)
    highs = np.asarray([item.upper for item in sampled], dtype=float)
    scaled = (
        qmc.scale(unit_samples, lows, highs)
        if sampled
        else np.empty((plan.n_samples, 0), dtype=float)
    )

    rows: list[dict[str, Any]] = []
    for sample_index in range(plan.n_samples):
        row: dict[str, Any] = {
            item.spec.name: item.nominal for item in plan.resolved_parameters
        }
        for column_index, item in enumerate(sampled):
            row[item.spec.name] = float(scaled[sample_index, column_index])
        parameters = _parameters_from_row(row)
        row = {"sample_id": sample_index, **row}
        try:
            _, report = cst_build_msabp_geometry.build_polygon_specs(
                parameters=parameters,
                coordinate_quantum_mm=plan.geometry_policy.coordinate_quantum_mm,
                allow_disconnected_conductor=(
                    plan.geometry_policy.allow_disconnected_conductor
                ),
            )
        except (TypeError, ValueError) as exc:
            row.update(
                geometry_valid=False,
                geometry_error=f"{type(exc).__name__}: {exc}",
                final_conductor_components=None,
            )
        else:
            row.update(
                geometry_valid=True,
                geometry_error="",
                final_conductor_components=report.final_conductor_component_count,
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    return SamplingResult(frame=frame, plan=plan)


def resolved_plan_to_dict(plan: SamplingPlan) -> dict[str, Any]:
    """Return a serializable snapshot of every effective sampling decision."""

    return {
        "schema_version": SCHEMA_VERSION,
        "method": plan.method,
        "n_samples": plan.n_samples,
        "seed": plan.seed,
        "geometry_policy": {
            "coordinate_quantum_mm": plan.geometry_policy.coordinate_quantum_mm,
            "reject_self_intersection": plan.geometry_policy.reject_self_intersection,
            "allow_disconnected_conductor": (
                plan.geometry_policy.allow_disconnected_conductor
            ),
        },
        "parameters": [
            {
                "name": item.spec.name,
                "group": item.spec.group,
                "unit": item.spec.unit,
                "kind": item.spec.kind,
                "nominal": item.nominal,
                "sample_default": item.spec.sample_default,
                "declared_sample": item.declared_sample,
                "effective_sample": item.effective_sample,
                "range": (
                    None
                    if item.lower is None
                    else {"min": item.lower, "max": item.upper}
                ),
                "range_source": item.range_source,
                "hard_bounds": {
                    "min": item.spec.hard_min,
                    "max": item.spec.hard_max,
                },
                "active_if_branch": item.spec.active_if_branch,
            }
            for item in plan.resolved_parameters
        ],
    }


def save_sampling_result(
    result: SamplingResult,
    output_path: str | Path,
    *,
    valid_only: bool = False,
) -> tuple[Path, Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = result.frame
    if valid_only:
        frame = frame.loc[frame["geometry_valid"]].reset_index(drop=True)
    frame.to_csv(output, index=False, float_format="%.12g")
    resolved_path = output.with_suffix(".resolved.json")
    resolved_path.write_text(
        json.dumps(resolved_plan_to_dict(result.plan), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output, resolved_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample antenna parameters and validate every 0.01 mm-quantized CST curve."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--method", choices=("sobol", "latin"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Write only rows that pass quantized geometry validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve, sample, and validate without writing CSV or metadata.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed is not None and args.no_seed:
        raise ValueError("--seed and --no-seed cannot be used together")
    seed_override: int | None | object
    if args.no_seed:
        seed_override = None
    elif args.seed is not None:
        seed_override = args.seed
    else:
        seed_override = ...

    config = load_sampling_config(args.config)
    plan = resolve_sampling_plan(
        config,
        n_samples=args.n_samples,
        method=args.method,
        seed=seed_override,
    )
    result = generate_samples(plan)
    sampled_count = sum(
        item.effective_sample for item in plan.resolved_parameters
    )
    valid_count = int(result.frame["geometry_valid"].sum())
    print(
        f"generated {len(result.frame)} samples across {sampled_count} active "
        f"dimensions; quantized-geometry valid={valid_count}, "
        f"invalid={len(result.frame) - valid_count}"
    )
    if not args.dry_run:
        output, resolved = save_sampling_result(
            result,
            args.output,
            valid_only=args.valid_only,
        )
        print(f"samples: {output}")
        print(f"resolved plan: {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
