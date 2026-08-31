"""Run the isolated late-stage four-objective K-RVEA continuation.

This entrypoint deliberately does not replace :mod:`run_krvea`.  Its policy is
calibrated only for the dense, late-stage archive after the first three K-RVEA
campaigns and must not be assumed to work during initial optimization.

F5 uses the constants below.  A real run still requires typing ``RUN`` unless
``--yes`` is supplied.  ``--prepare-only`` never starts SSH, Princess, Maid,
or CST.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import krvea_relay  # noqa: E402
from scripts.optimization import run_krvea as baseline  # noqa: E402


F5_PLAN_ID = "msabp-krvea-11var-deep-64-004"
F5_SOURCE_DIRECTORIES = (
    REPOSITORY_ROOT / "results" / "raw" / "doe-11var-branch-up-lhs-512-001",
    REPOSITORY_ROOT / "results" / "raw" / "msabp-krvea-11var-smoke-128-001",
    REPOSITORY_ROOT / "results" / "raw" / "msabp-krvea-11var-calibrated-64-002",
    REPOSITORY_ROOT / "results" / "raw" / "msabp-krvea-11var-calibrated-64-003",
)
F5_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "raw" / F5_PLAN_ID
F5_TOTAL_BUDGET = 64
F5_Q = 4
F5_SEED = 20260901
F5_REQUIRE_CONFIRMATION = True

STRATEGY_NAME = "deep_late_stage_s11_guard_v1"
S11_UNCERTAINTY_CALIBRATION_FACTOR = 2.5
EXPENSIVE_UNCERTAINTY_CALIBRATION_FACTORS = (
    S11_UNCERTAINTY_CALIBRATION_FACTOR,
    1.1,
    1.25,
)
EXPLORATION_PERIOD_BATCHES = 2
UNCERTAINTY_CALIBRATION_SOURCE = (
    "round3_63_success_signed_s11_replay_60_of_63_upper_coverage"
)


def deep_surrogate_fit_settings() -> krvea_relay.SurrogateFitSettings:
    """Return the late-stage-only surrogate contract.

    In round three, the baseline S11 factor 1.1 covered 57/63 successful
    observations with the one-sided ``mean + 1.645 * std`` guard.  A
    scale-only replay of the same signed residuals with factor 2.5 covers
    60/63 (95.2%).
    """

    return krvea_relay.SurrogateFitSettings(
        gp_training_steps=baseline.GP_TRAINING_STEPS,
        gp_kernel=baseline.GP_KERNEL,
        gp_noise_mode=baseline.GP_NOISE_MODE,
        gp_fixed_noise_variance=baseline.GP_FIXED_NOISE_VARIANCE,
        gp_posterior_observation_noise=(
            baseline.GP_POSTERIOR_OBSERVATION_NOISE
        ),
        gp_timeout_seconds=baseline.GP_TIMEOUT_SECONDS,
        uncertainty_calibration_factors=(
            EXPENSIVE_UNCERTAINTY_CALIBRATION_FACTORS
        ),
        uncertainty_calibration_source=UNCERTAINTY_CALIBRATION_SOURCE,
    )


def build_config(
    *,
    plan_id: str = F5_PLAN_ID,
    source_directories: Sequence[Path] = F5_SOURCE_DIRECTORIES,
    output_directory: Path = F5_OUTPUT_DIRECTORY,
    total_budget: int = F5_TOTAL_BUDGET,
    q: int = F5_Q,
    device_ids: Sequence[str] = baseline.F5_DEVICE_IDS,
) -> baseline.CampaignConfig:
    """Build the isolated deep-stage campaign configuration."""

    proposal = replace(
        baseline.F5_PROPOSAL,
        q=int(q),
        seed=F5_SEED,
        exploration_slots=1,
    )
    return baseline.CampaignConfig(
        plan_id=str(plan_id),
        source_directories=tuple(Path(path) for path in source_directories),
        output_directory=Path(output_directory),
        total_budget=int(total_budget),
        band_ghz=baseline.F5_BAND_GHZ,
        device_ids=tuple(str(value) for value in device_ids),
        sampling_config=baseline.SAMPLING_CONFIG,
        device_config=baseline.DEVICE_CONFIG,
        project_template=baseline.PROJECT_TEMPLATE,
        proposal=proposal,
        proposal_remote=baseline.F5_PROPOSAL_REMOTE,
        coordinate_quantum_mm=0.01,
        allow_disconnected_conductor=False,
        max_attempts=3,
        surrogate_settings=deep_surrogate_fit_settings(),
        exploration_period_batches=EXPLORATION_PERIOD_BATCHES,
        strategy_name=STRATEGY_NAME,
        strategy_source=Path(__file__).resolve(),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--source", action="append", dest="sources", type=Path)
    parser.add_argument("--output", type=Path, default=F5_OUTPUT_DIRECTORY)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--device", action="append", dest="device_ids")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-after-proposal", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args(argv)


def _existing_plan(output: Path) -> Mapping[str, Any]:
    path = output / baseline.PLAN_FILENAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid existing deep K-RVEA plan: {path}")
    return payload


def config_from_args(args: argparse.Namespace) -> baseline.CampaignConfig:
    output = args.output.expanduser().resolve()
    existing = _existing_plan(output)
    simulation = existing.get("simulation", {})
    if not isinstance(simulation, Mapping):
        simulation = {}
    sources = (
        tuple(args.sources)
        if args.sources
        else tuple(Path(value) for value in existing.get("source_directories", ()))
        or F5_SOURCE_DIRECTORIES
    )
    devices = (
        tuple(args.device_ids)
        if args.device_ids
        else tuple(str(value) for value in simulation.get("device_ids", ()))
        or baseline.F5_DEVICE_IDS
    )
    return build_config(
        plan_id=str(args.plan_id or existing.get("plan_id", F5_PLAN_ID)),
        source_directories=sources,
        output_directory=output,
        total_budget=int(
            args.budget
            if args.budget is not None
            else existing.get("total_budget", F5_TOTAL_BUDGET)
        ),
        q=int(args.q if args.q is not None else existing.get("q", F5_Q)),
        device_ids=devices,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    if not args.prepare_only and not args.stop_after_proposal:
        if F5_REQUIRE_CONFIRMATION and not args.yes:
            answer = input(
                "Type RUN to start/resume deep K-RVEA plan "
                f"{config.plan_id} ({config.total_budget} new evaluations): "
            )
            if answer.strip() != "RUN":
                print("Cancelled; no proposal worker or solver was started.")
                return 1
    try:
        return baseline.run_campaign(
            config,
            prepare_only=args.prepare_only,
            stop_after_proposal=args.stop_after_proposal,
        )
    except KeyboardInterrupt:
        print("[Deep K-RVEA] interrupted; plan and active batch remain resumable")
        return 130
    except Exception as exc:
        print(f"Deep K-RVEA error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
