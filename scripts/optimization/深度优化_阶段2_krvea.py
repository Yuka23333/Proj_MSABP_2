"""Run the 64-point first arm of sixth-round deep optimization.

This entrypoint is isolated from all earlier K-RVEA strategies.  It uses a
learned-noise Matérn GP, frozen physical uncertainty floors from Stage-1
rolling replay, and reserves one of every four proposals for novelty search.
The remaining 64 points of the approved sixth-round budget are intentionally
not part of this campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for _import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from msabp_opt.optimization import krvea_relay  # noqa: E402
from msabp_opt.simulation.distributed.config import load_device_registry  # noqa: E402
from msabp_opt.simulation.distributed.runtime import select_devices  # noqa: E402
from scripts.optimization import run_krvea as baseline  # noqa: E402
from scripts.optimization import 深度优化_krvea as deep_config  # noqa: E402
from scripts.optimization import 深度优化_阶段2_relay as stage2_relay  # noqa: E402


STRATEGY_NAME = "deep_stage2_learned_noise_guard_v1"
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "optimization"
    / "deep_krvea_stage2_round6.json"
)
F5_REQUIRE_CONFIRMATION = True
_STAGE2_POLICY_KEYS = {
    "schema_version",
    "physical_std_floor",
    "calibration_source",
    "holdout_source",
    "holdout_catastrophic_optimism_count",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Stage-2 {label} must be a JSON object")
    return value


def load_config_document(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[Path, Mapping[str, Any], stage2_relay.Stage2Policy]:
    """Load the Stage-2 extension and validate the inherited base document."""

    path = Path(config_path).expanduser().resolve()
    payload = _mapping(
        json.loads(path.read_text(encoding="utf-8-sig")), "document"
    )
    expected = set(deep_config._TOP_LEVEL_KEYS) | {"stage2_policy"}
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown or missing:
        raise ValueError(
            f"Stage-2 document fields mismatch: missing={missing}, unknown={unknown}"
        )
    policy_payload = _mapping(payload["stage2_policy"], "stage2_policy")
    policy_unknown = sorted(set(policy_payload) - _STAGE2_POLICY_KEYS)
    policy_missing = sorted(_STAGE2_POLICY_KEYS - set(policy_payload))
    if policy_unknown or policy_missing:
        raise ValueError(
            "Stage-2 policy fields mismatch: "
            f"missing={policy_missing}, unknown={policy_unknown}"
        )
    inherited = {
        key: value for key, value in payload.items() if key != "stage2_policy"
    }
    document = deep_config.validate_config_document(inherited)
    policy = stage2_relay.Stage2Policy.from_mapping(policy_payload)
    return path, document, policy


def build_config(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    plan_id: str | None = None,
    source_directories: Sequence[Path] | None = None,
    output_directory: Path | None = None,
    total_budget: int | None = None,
    q: int | None = None,
    device_ids: Sequence[str] | None = None,
) -> tuple[baseline.CampaignConfig, stage2_relay.Stage2Policy]:
    path, document, policy = load_config_document(config_path)
    config = deep_config.build_config(
        config_path=path,
        plan_id=plan_id,
        source_directories=source_directories,
        output_directory=output_directory,
        total_budget=total_budget,
        q=q,
        device_ids=device_ids,
        _document=document,
        _strategy_name=STRATEGY_NAME,
        _strategy_source=Path(__file__),
    )
    if config.surrogate_settings.gp_noise_mode != "learned":
        raise ValueError("Stage-2 configuration requires learned GP noise")
    if config.proposal.exploration_slots != 1:
        raise ValueError("Stage-2 requires exactly one reserved exploration slot")
    if config.exploration_period_batches != 1:
        raise ValueError("Stage-2 requires exploration in every batch")
    return config, policy


def _request_remote_proposal(
    policy: stage2_relay.Stage2Policy,
    config: baseline.CampaignConfig,
    dataset: Any,
    *,
    batch_index: int,
    q: int,
    remaining_budget: int,
    previous_empty_reference_count: int | None,
) -> Any:
    settings = baseline.proposal_settings_for_batch(
        config, batch_index=batch_index, q=q
    )
    penalty_mask = ~dataset.metadata["has_completed_result"].to_numpy(dtype=bool)
    request = krvea_relay.build_request_payload(
        dataset.x_unit,
        dataset.objectives[:, [0, 1, 3]],
        dataset.objectives,
        penalty_mask,
        dataset.input_space,
        config=settings,
        iteration=batch_index,
        remaining_expensive_budget=remaining_budget,
        previous_empty_reference_count=previous_empty_reference_count,
        compute_device=config.proposal_remote.compute_device,
        surrogate_settings=baseline.campaign_surrogate_fit_settings(config),
    )
    request = stage2_relay.attach_policy(request, policy)
    control = baseline._control_directory(config)
    request_path = control / f"batch_{batch_index:04d}_proposal_request.json"
    response_path = control / f"batch_{batch_index:04d}_proposal_response.json"
    krvea_relay.write_request(request_path, request)
    registry = load_device_registry(config.device_config)
    device = select_devices(registry, (config.proposal_remote.device_id,))[0]
    return stage2_relay.relay_remote_proposal(
        device=device,
        remote=config.proposal_remote,
        plan_id=config.plan_id,
        batch_index=batch_index,
        local_request_path=request_path,
        local_response_path=response_path,
        expected_q=q,
        expected_dimension=len(dataset.input_space.names),
        observed_x_unit=np.asarray(dataset.x_unit, dtype=np.float64),
        input_space=dataset.input_space,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--source", action="append", dest="sources", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--device", action="append", dest="device_ids")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-after-proposal", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args(argv)


def config_from_args(
    args: argparse.Namespace,
) -> tuple[baseline.CampaignConfig, stage2_relay.Stage2Policy]:
    return build_config(
        config_path=args.config,
        plan_id=args.plan_id,
        source_directories=(tuple(args.sources) if args.sources else None),
        output_directory=args.output,
        total_budget=args.budget,
        q=args.q,
        device_ids=(tuple(args.device_ids) if args.device_ids else None),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config, policy = config_from_args(args)
        print(f"[Deep Stage 2] config={Path(args.config).resolve()}", flush=True)
        print(
            "[Deep Stage 2] physical std floors="
            f"{policy.physical_std_floor}; exploration=1/{config.proposal.q}",
            flush=True,
        )
        if not args.prepare_only and not args.stop_after_proposal:
            if F5_REQUIRE_CONFIRMATION and not args.yes:
                answer = input(
                    "Type RUN to start/resume Stage-2 K-RVEA plan "
                    f"{config.plan_id} ({config.total_budget} evaluations): "
                )
                if answer.strip() != "RUN":
                    print("Cancelled; no proposal worker or solver was started.")
                    return 1

        original = baseline._request_remote_proposal
        baseline._request_remote_proposal = partial(
            _request_remote_proposal, policy
        )
        try:
            return baseline.run_campaign(
                config,
                prepare_only=args.prepare_only,
                stop_after_proposal=args.stop_after_proposal,
            )
        finally:
            baseline._request_remote_proposal = original
    except KeyboardInterrupt:
        print("[Deep Stage 2] interrupted; campaign state remains resumable")
        return 130
    except Exception as exc:
        print(f"Deep Stage-2 error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
