"""Sample the redesigned 23-variable Shapely antenna parameter space."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc
from shapely.errors import ShapelyError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.geometry import shapely_antenna_model as antenna_outline  # noqa: E402


SCHEMA_VERSION = 2
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "optimization" / "antenna_sampling.json"
)
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "data" / "samples" / "antenna_samples.csv"
CSV_FLOAT_FORMAT = "%.17g"

PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "absolute_mm": antenna_outline.ABSOLUTE_PARAMETER_NAMES,
    "upper_corner_ratio": (
        "UPPER_CORNER_NOTCH_1_K1",
        "UPPER_CORNER_NOTCH_1_K2",
        "UPPER_CORNER_EAR_1_K1",
        "UPPER_CORNER_EAR_1_K2",
    ),
    "lower_corner_ratio": (
        "LOWER_CORNER_NOTCH_1_K1",
        "LOWER_CORNER_NOTCH_1_K2",
        "LOWER_CORNER_EAR_1_K1",
        "LOWER_CORNER_EAR_1_K2",
        "LOWER_CORNER_EAR_2_K1",
        "LOWER_CORNER_EAR_2_K2",
    ),
    "branch_ratio": (
        "BRANCH_UP_1_K",
        "BRANCH_UP_1_K2",
        "BRANCH_UP_1_K3",
        "BRANCH_DOWN_1_K",
        "BRANCH_DOWN_1_K2",
        "BRANCH_DOWN_1_K3",
    ),
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    group: str
    unit: str
    kind: str
    sample_default: bool
    code_default: float
    hard_min: float | None = None
    hard_max: float | None = None


@dataclass(frozen=True)
class ResolvedParameter:
    spec: ParameterSpec
    nominal: float
    declared_sample: bool
    effective_sample: bool
    lower: float | None
    upper: float | None
    range_source: str | None


@dataclass(frozen=True)
class GeometryPolicy:
    coordinate_quantum_mm: float = antenna_outline.QUANTIZE_STEP_MM
    reject_self_intersection: bool = True
    allow_disconnected_conductor: bool = False


@dataclass(frozen=True)
class SamplingPlan:
    method: str
    n_samples: int
    seed: int | None
    parameters: antenna_outline.ShapelyAntennaParameters
    resolved_parameters: tuple[ResolvedParameter, ...]
    geometry_policy: GeometryPolicy
    raw_config: Mapping[str, Any]


@dataclass(frozen=True)
class SamplingResult:
    frame: pd.DataFrame
    plan: SamplingPlan


def _group_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group, names in PARAMETER_GROUPS.items():
        for name in names:
            if name in lookup:
                raise RuntimeError(f"parameter {name!r} appears in multiple groups")
            lookup[name] = group
    return lookup


def build_parameter_registry() -> dict[str, ParameterSpec]:
    groups = _group_lookup()
    expected = set(antenna_outline.PARAMETER_NAMES)
    if set(groups) != expected:
        raise RuntimeError("parameter groups do not cover the 23 design variables")
    registry: dict[str, ParameterSpec] = {}
    for name in antenna_outline.PARAMETER_NAMES:
        is_ratio = name in antenna_outline.RATIO_PARAMETER_NAMES
        registry[name] = ParameterSpec(
            name=name,
            group=groups[name],
            unit="ratio" if is_ratio else "mm",
            kind="ratio" if is_ratio else "absolute",
            sample_default=True,
            code_default=float(getattr(antenna_outline.DEFAULT_PARAMETERS, name)),
            hard_min=0.0 if is_ratio else None,
            hard_max=1.0 if is_ratio else None,
        )
    return registry


PARAMETER_REGISTRY = build_parameter_registry()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
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


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def load_sampling_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    config = _mapping(json.loads(source.read_text(encoding="utf-8")), "config")
    _reject_unknown_keys(
        config,
        {"schema_version", "sampling", "geometry_policy"},
        "config",
    )
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {SCHEMA_VERSION}")
    return dict(config)


def _parse_range(
    raw_range: Any,
    *,
    nominal: float,
    label: str,
) -> tuple[float, float]:
    value = _mapping(raw_range, label)
    mode = str(value.get("mode", "absolute")).lower()
    if mode == "absolute":
        _reject_unknown_keys(value, {"mode", "min", "max"}, label)
        lower = _number(value.get("min"), f"{label}.min")
        upper = _number(value.get("max"), f"{label}.max")
    elif mode == "relative":
        _reject_unknown_keys(
            value,
            {"mode", "lower", "upper", "reference"},
            label,
        )
        if value.get("reference", "nominal") != "nominal":
            raise ValueError(f"{label}.reference must be 'nominal'")
        lower = nominal * (1.0 + _number(value.get("lower"), f"{label}.lower"))
        upper = nominal * (1.0 + _number(value.get("upper"), f"{label}.upper"))
    else:
        raise ValueError(f"{label}.mode must be 'absolute' or 'relative'")
    if not lower < upper:
        raise ValueError(f"{label} must satisfy lower < upper")
    return lower, upper


def _resolve_geometry_policy(config: Mapping[str, Any]) -> GeometryPolicy:
    raw = _mapping(config.get("geometry_policy", {}), "geometry_policy")
    _reject_unknown_keys(
        raw,
        {
            "coordinate_quantum_mm",
            "reject_self_intersection",
            "allow_disconnected_conductor",
        },
        "geometry_policy",
    )
    quantum = _number(
        raw.get("coordinate_quantum_mm", antenna_outline.QUANTIZE_STEP_MM),
        "geometry_policy.coordinate_quantum_mm",
    )
    reject = raw.get("reject_self_intersection", True)
    disconnected = raw.get("allow_disconnected_conductor", False)
    if not isinstance(reject, bool) or not isinstance(disconnected, bool):
        raise ValueError("geometry policy switches must be booleans")
    return GeometryPolicy(quantum, reject, disconnected)


def resolve_sampling_plan(
    config: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    method: str | None = None,
    seed: int | None | object = ...,
) -> SamplingPlan:
    sampling = _mapping(config.get("sampling", {}), "sampling")
    _reject_unknown_keys(
        sampling,
        {"method", "n_samples", "seed", "parameters"},
        "sampling",
    )
    resolved_method = str(method or sampling.get("method", "sobol")).lower()
    if resolved_method not in {"sobol", "latin"}:
        raise ValueError("sampling.method must be 'sobol' or 'latin'")
    resolved_count = int(
        n_samples if n_samples is not None else sampling.get("n_samples", 1024)
    )
    if resolved_count <= 0:
        raise ValueError("sampling.n_samples must be positive")
    if seed is ...:
        raw_seed = sampling.get("seed", 20260806)
        resolved_seed = None if raw_seed is None else int(raw_seed)
    else:
        resolved_seed = None if seed is None else int(seed)

    overrides = _mapping(sampling.get("parameters", {}), "sampling.parameters")
    missing = set(PARAMETER_REGISTRY) - set(overrides)
    extra = set(overrides) - set(PARAMETER_REGISTRY)
    if missing or extra:
        raise ValueError(
            "sampling.parameters must contain exactly the 23 variables: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    nominal_values: dict[str, float] = {}
    for name, spec in PARAMETER_REGISTRY.items():
        override = _mapping(overrides[name], f"sampling.parameters.{name}")
        _reject_unknown_keys(
            override,
            {"value", "sample", "range"},
            f"sampling.parameters.{name}",
        )
        nominal_values[name] = _number(
            override.get("value", spec.code_default),
            f"sampling.parameters.{name}.value",
        )
    parameters = antenna_outline.ShapelyAntennaParameters(**nominal_values)

    resolved: list[ResolvedParameter] = []
    for name, spec in PARAMETER_REGISTRY.items():
        override = _mapping(overrides[name], f"sampling.parameters.{name}")
        sampled = override.get("sample", spec.sample_default)
        if not isinstance(sampled, bool):
            raise ValueError(f"sampling.parameters.{name}.sample must be boolean")
        lower: float | None = None
        upper: float | None = None
        source: str | None = None
        if sampled:
            if "range" not in override:
                raise ValueError(f"sampled parameter {name} has no range")
            lower, upper = _parse_range(
                override["range"],
                nominal=nominal_values[name],
                label=f"sampling.parameters.{name}.range",
            )
            if spec.hard_min is not None:
                lower = max(lower, spec.hard_min)
            if spec.hard_max is not None:
                upper = min(upper, spec.hard_max)
            if not lower < upper:
                raise ValueError(f"effective range for {name} is empty")
            source = "parameter"
        resolved.append(
            ResolvedParameter(
                spec=spec,
                nominal=nominal_values[name],
                declared_sample=sampled,
                effective_sample=sampled,
                lower=lower,
                upper=upper,
                range_source=source,
            )
        )
    return SamplingPlan(
        method=resolved_method,
        n_samples=resolved_count,
        seed=resolved_seed,
        parameters=parameters,
        resolved_parameters=tuple(resolved),
        geometry_policy=_resolve_geometry_policy(config),
        raw_config=config,
    )


def _sample_unit_cube(plan: SamplingPlan, dimension: int) -> np.ndarray:
    if plan.method == "sobol":
        engine = qmc.Sobol(d=dimension, scramble=True, seed=plan.seed)
        return engine.random(plan.n_samples)
    engine = qmc.LatinHypercube(d=dimension, seed=plan.seed)
    return engine.random(plan.n_samples)


def generate_parameter_frame(plan: SamplingPlan) -> pd.DataFrame:
    sampled = [item for item in plan.resolved_parameters if item.effective_sample]
    unit = _sample_unit_cube(plan, len(sampled))
    lower = np.asarray([item.lower for item in sampled], dtype=float)
    upper = np.asarray([item.upper for item in sampled], dtype=float)
    scaled = qmc.scale(unit, lower, upper)
    rows: list[dict[str, Any]] = []
    for sample_index in range(plan.n_samples):
        row = {item.spec.name: item.nominal for item in plan.resolved_parameters}
        for column_index, item in enumerate(sampled):
            row[item.spec.name] = float(scaled[sample_index, column_index])
        rows.append({"sample_id": sample_index, **row})
    return pd.DataFrame(rows)


def parameters_from_csv_row(
    row: Mapping[str, Any],
) -> antenna_outline.ShapelyAntennaParameters:
    missing = set(PARAMETER_REGISTRY) - set(row)
    if missing:
        raise ValueError(f"sample row is missing parameters: {sorted(missing)}")
    values = {
        name: _number(row[name], f"sample row parameter {name}")
        for name in PARAMETER_REGISTRY
    }
    for name in antenna_outline.RATIO_PARAMETER_NAMES:
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"sample row parameter {name} must be inside [0, 1]")
    return antenna_outline.ShapelyAntennaParameters(**values)


def generate_samples(plan: SamplingPlan) -> SamplingResult:
    frame = generate_parameter_frame(plan)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        try:
            parameters = parameters_from_csv_row(row)
            payload = antenna_outline.polygon_export_payload(
                parameters,
                quantize_step_mm=plan.geometry_policy.coordinate_quantum_mm,
            )
            checks = payload["meta"]["self_intersection_check"]
            invalid_curves = [
                name
                for name, item in checks.items()
                if not (item["ring_is_simple"] and item["polygon_is_valid"])
            ]
            if invalid_curves:
                raise ValueError(
                    "producer reported invalid exported polygon(s): "
                    + ", ".join(invalid_curves)
                )
        except (TypeError, ValueError, ShapelyError) as exc:
            row.update(
                geometry_valid=False,
                geometry_error=f"{type(exc).__name__}: {exc}",
                final_conductor_components=None,
            )
        else:
            row.update(
                geometry_valid=True,
                geometry_error="",
                final_conductor_components=1,
            )
        rows.append(row)
    return SamplingResult(pd.DataFrame(rows), plan)


def resolved_plan_to_dict(plan: SamplingPlan) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": plan.method,
        "n_samples": plan.n_samples,
        "seed": plan.seed,
        "geometry_policy": asdict(plan.geometry_policy),
        "parameters": [
            {
                "name": item.spec.name,
                "group": item.spec.group,
                "kind": item.spec.kind,
                "unit": item.spec.unit,
                "nominal": item.nominal,
                "sample": item.effective_sample,
                "lower": item.lower,
                "upper": item.upper,
                "range_source": item.range_source,
            }
            for item in plan.resolved_parameters
        ],
    }


def save_sampling_result(
    result: SamplingResult,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    valid_only: bool = False,
) -> tuple[Path, Path]:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = result.frame
    if valid_only:
        frame = frame.loc[frame["geometry_valid"]].reset_index(drop=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, float_format=CSV_FLOAT_FORMAT)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    resolved_path = output.with_suffix(".resolved.json")
    resolved_path.write_text(
        json.dumps(resolved_plan_to_dict(result.plan), indent=2),
        encoding="utf-8",
    )
    return output, resolved_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--method", choices=("sobol", "latin"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_sampling_config(args.config)
    kwargs: dict[str, Any] = {}
    if args.n_samples is not None:
        kwargs["n_samples"] = args.n_samples
    if args.method is not None:
        kwargs["method"] = args.method
    if args.seed is not None:
        kwargs["seed"] = args.seed
    plan = resolve_sampling_plan(config, **kwargs)
    print(
        f"[sampler] method={plan.method} samples={plan.n_samples} "
        f"dimensions={sum(item.effective_sample for item in plan.resolved_parameters)}"
    )
    if args.dry_run:
        print(json.dumps(resolved_plan_to_dict(plan), indent=2))
        return 0
    result = generate_samples(plan)
    output, resolved = save_sampling_result(
        result,
        args.output,
        valid_only=args.valid_only,
    )
    valid_count = int(result.frame["geometry_valid"].sum())
    print(f"[sampler] valid={valid_count}/{len(result.frame)} output={output}")
    print(f"[sampler] resolved={resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
