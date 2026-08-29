"""Prepare the 11-variable, branch-focused 512-point LHS DoE.

F5/default execution samples four absolute dimensions inside +/-10%, places
four upper-corner K variables inside a clipped interval of length ``P``, and
samples the three core upper-branch K variables over [0.05, 1].  All remaining
antenna parameters stay fixed at their code defaults.

The clipped P-window is exactly::

    lo = clip(v0 - P / 2, 0, 1 - P)
    window = [lo, lo + P]

Existing samplers and DoE preparation code are reused without modification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import antenna_sampler  # noqa: E402
from scripts.optimization import prepare_doe_round1  # noqa: E402


DEFAULT_ROUND_ID = "doe-11var-branch-up-lhs-512"
DEFAULT_SAMPLE_COUNT = 512
DEFAULT_SEED = 20260825
DEFAULT_P_WINDOW = 0.3
DEFAULT_ABSOLUTE_RELATIVE_HALF_WIDTH = 0.10
CORE_BRANCH_RANGE = (0.05, 1.0)

ABSOLUTE_VARIABLES = (
    "PATCH_BRICK_1_TOP_MARGIN",
    "PATCH_BRICK_2_HEIGHT_MARGIN",
    "PATCH_BRICK_1_SIDE_MARGIN",
    "SLOT_MAIN_LENGTH",
)
P_WINDOW_VARIABLES = (
    "UPPER_CORNER_NOTCH_1_K1",
    "UPPER_CORNER_NOTCH_1_K2",
    "UPPER_CORNER_EAR_1_K1",
    "UPPER_CORNER_EAR_1_K2",
)
CORE_BRANCH_VARIABLES = (
    "BRANCH_UP_1_K",
    "BRANCH_UP_1_K2",
    "BRANCH_UP_1_K3",
)
SAMPLED_VARIABLES = (
    *ABSOLUTE_VARIABLES,
    *P_WINDOW_VARIABLES,
    *CORE_BRANCH_VARIABLES,
)

_SAFE_ROUND_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def clipped_p_window(initial_value: float, p_window: float) -> tuple[float, float]:
    """Return a length-P interval in [0,1] whose midpoint best follows v0."""

    initial_value = float(initial_value)
    p_window = float(p_window)
    if not math.isfinite(initial_value) or not 0.0 <= initial_value <= 1.0:
        raise ValueError("initial K value must be finite and inside [0,1]")
    if not math.isfinite(p_window) or not 0.0 < p_window <= 1.0:
        raise ValueError("P must be finite and inside (0,1]")
    lower = min(max(initial_value - p_window / 2.0, 0.0), 1.0 - p_window)
    return lower, lower + p_window


def build_sampling_config(
    *,
    p_window: float = DEFAULT_P_WINDOW,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    absolute_relative_half_width: float = DEFAULT_ABSOLUTE_RELATIVE_HALF_WIDTH,
) -> dict[str, Any]:
    """Build a schema-v2 sampler config with exactly 11 active variables."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 < absolute_relative_half_width < 1.0:
        raise ValueError("absolute half-width must be inside (0,1)")

    parameters: dict[str, dict[str, Any]] = {
        name: {"value": spec.code_default, "sample": False}
        for name, spec in antenna_sampler.PARAMETER_REGISTRY.items()
    }
    for name in ABSOLUTE_VARIABLES:
        nominal = antenna_sampler.PARAMETER_REGISTRY[name].code_default
        parameters[name] = {
            "value": nominal,
            "sample": True,
            "range": {
                "mode": "absolute",
                "min": nominal * (1.0 - absolute_relative_half_width),
                "max": nominal * (1.0 + absolute_relative_half_width),
            },
        }
    for name in P_WINDOW_VARIABLES:
        nominal = antenna_sampler.PARAMETER_REGISTRY[name].code_default
        lower, upper = clipped_p_window(nominal, p_window)
        parameters[name] = {
            "value": nominal,
            "sample": True,
            "range": {"mode": "absolute", "min": lower, "max": upper},
        }
    for name in CORE_BRANCH_VARIABLES:
        nominal = antenna_sampler.PARAMETER_REGISTRY[name].code_default
        parameters[name] = {
            "value": nominal,
            "sample": True,
            "range": {
                "mode": "absolute",
                "min": CORE_BRANCH_RANGE[0],
                "max": CORE_BRANCH_RANGE[1],
            },
        }

    return {
        "schema_version": antenna_sampler.SCHEMA_VERSION,
        "sampling": {
            "method": "latin",
            "n_samples": sample_count,
            "seed": seed,
            "parameters": parameters,
        },
        "geometry_policy": {
            "coordinate_quantum_mm": 0.01,
            "reject_self_intersection": True,
            "allow_disconnected_conductor": False,
        },
    }


def _repository_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT)).replace("\\", "/")


def _paths(round_id: str) -> dict[str, Path]:
    if not _SAFE_ROUND_ID.fullmatch(round_id):
        raise ValueError(
            "round_id may contain only letters, digits, dot, dash, underscore"
        )
    processed = REPOSITORY_ROOT / "results" / "processed"
    samples = REPOSITORY_ROOT / "data" / "samples"
    return {
        "sampling_config": processed / f"{round_id}.sampling.json",
        "round_config": processed / f"{round_id}.round.json",
        "resolved_plan": processed / f"{round_id}.resolved.json",
        "accepted_csv": samples / f"{round_id}.csv",
        "candidates_csv": samples / f"{round_id}_candidates.csv",
        "rejected_csv": samples / f"{round_id}_rejected.csv",
        "summary_json": processed / f"{round_id}_summary.json",
    }


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
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


def _round_config(round_id: str, paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": prepare_doe_round1.ROUND_CONFIG_SCHEMA_VERSION,
        "round_id": round_id,
        "base_sampling_config": _repository_relative(paths["sampling_config"]),
        "sampling": {
            "method": "latin",
            "candidate_count": DEFAULT_SAMPLE_COUNT,
            "seed": DEFAULT_SEED,
            "include_origin": False,
        },
        "eligibility": {
            "require_individually_valid_curves": True,
            "ignored_bottom_edge_contact": True,
            "allowed_non_bottom_overlap_pairs": ["Slot__CPW_Feed_Pin"],
        },
        "outputs": {
            name: _repository_relative(paths[name])
            for name in (
                "accepted_csv",
                "candidates_csv",
                "rejected_csv",
                "summary_json",
            )
        },
    }


def prepare_design(
    *,
    round_id: str = DEFAULT_ROUND_ID,
    p_window: float = DEFAULT_P_WINDOW,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Generate, audit, and persist the branch-focused LHS design."""

    paths = _paths(round_id)
    sampling_config = build_sampling_config(
        p_window=p_window,
        sample_count=sample_count,
        seed=seed,
    )
    plan = antenna_sampler.resolve_sampling_plan(sampling_config)
    active = tuple(
        item.spec.name for item in plan.resolved_parameters if item.effective_sample
    )
    if set(active) != set(SAMPLED_VARIABLES) or len(active) != 11:
        raise RuntimeError(
            f"expected exactly the configured 11 variables, got {active}"
        )

    _atomic_write_json(sampling_config, paths["sampling_config"])
    round_config = _round_config(round_id, paths)
    round_config["sampling"]["candidate_count"] = sample_count
    round_config["sampling"]["seed"] = seed
    _atomic_write_json(round_config, paths["round_config"])
    _atomic_write_json(
        antenna_sampler.resolved_plan_to_dict(plan),
        paths["resolved_plan"],
    )

    accepted, candidates, summary = prepare_doe_round1.prepare_round(
        paths["round_config"]
    )
    ranges = {
        item.spec.name: {
            "initial": item.nominal,
            "lower": item.lower,
            "upper": item.upper,
        }
        for item in plan.resolved_parameters
        if item.effective_sample
    }
    summary.update(
        independent_variable_count=len(active),
        sampled_variables=list(active),
        p_window={
            "P": p_window,
            "rule": "lo=clip(v0-P/2,0,1-P); window=[lo,lo+P]",
            "variables": list(P_WINDOW_VARIABLES),
        },
        core_branch_range={
            "lower": CORE_BRANCH_RANGE[0],
            "upper": CORE_BRANCH_RANGE[1],
            "variables": list(CORE_BRANCH_VARIABLES),
        },
        absolute_relative_half_width=DEFAULT_ABSOLUTE_RELATIVE_HALF_WIDTH,
        resolved_ranges=ranges,
        generated_configuration={
            name: {
                "path": _repository_relative(paths[name]),
                "sha256": _sha256(paths[name]),
            }
            for name in ("sampling_config", "round_config", "resolved_plan")
        },
    )
    prepare_doe_round1.write_round_outputs(
        accepted,
        candidates,
        summary,
        round_config_path=paths["round_config"],
    )
    return paths, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-id", default=DEFAULT_ROUND_ID)
    parser.add_argument("--p", type=float, default=DEFAULT_P_WINDOW)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved 11-variable plan without generating samples.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_sampling_config(
        p_window=args.p,
        sample_count=args.samples,
        seed=args.seed,
    )
    plan = antenna_sampler.resolve_sampling_plan(config)
    if args.dry_run:
        print(json.dumps(antenna_sampler.resolved_plan_to_dict(plan), indent=2))
        return 0

    paths, summary = prepare_design(
        round_id=args.round_id,
        p_window=args.p,
        sample_count=args.samples,
        seed=args.seed,
    )
    print(
        f"[DoE] round={args.round_id} method=latin dimensions=11 "
        f"candidates={summary['candidate_count']} "
        f"accepted={summary['accepted_sample_count']} "
        f"rejected={summary['rejected_count']}"
    )
    print(f"[DoE] worklist={paths['accepted_csv']}")
    print(f"[DoE] summary={paths['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
