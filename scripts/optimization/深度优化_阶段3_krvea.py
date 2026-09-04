"""Run a feasibility-filtered 32-point continuation after Stage 2.

The GPU proposes an oversized K-RVEA pool.  Before campaign state or case
directories are created, the controller runs the existing full Shapely
preflight and selects exactly four valid candidates.  Rejected pool members
are retained in proposal diagnostics but consume no simulation budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
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
from scripts.optimization import 深度优化_阶段2_relay as learned_relay  # noqa: E402


STRATEGY_NAME = "deep_stage3_feasible_pool_v1"
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "optimization"
    / "deep_krvea_stage3_round7.json"
)
F5_REQUIRE_CONFIRMATION = True
_STAGE2_POLICY_KEYS = {
    "schema_version",
    "physical_std_floor",
    "calibration_source",
    "holdout_source",
    "holdout_catastrophic_optimism_count",
}
_STAGE3_POLICY_KEYS = {
    "schema_version",
    "proposal_pool_size",
    "proposal_pool_exploration_slots",
    "selection_order",
    "insufficient_valid_action",
    "rejected_candidates_consume_budget",
}
_SELECTION_VECTOR_KEYS = (
    "selected_reference_indices",
    "selected_apd",
    "selected_mean_std",
    "selected_nearest_archive_distance",
    "selected_boundary_distance",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Stage-3 {label} must be a JSON object")
    return value


@dataclass(frozen=True)
class FeasiblePoolPolicy:
    proposal_pool_size: int
    proposal_pool_exploration_slots: int
    selection_order: str
    insufficient_valid_action: str
    rejected_candidates_consume_budget: bool

    def __post_init__(self) -> None:
        if self.proposal_pool_size < 8:
            raise ValueError("Stage-3 proposal_pool_size must be at least eight")
        if not 1 <= self.proposal_pool_exploration_slots < self.proposal_pool_size:
            raise ValueError("Stage-3 pool exploration slots are out of range")
        if self.selection_order != (
            "first_valid_preserve_exploitation_then_exploration"
        ):
            raise ValueError("unsupported Stage-3 pool selection_order")
        if self.insufficient_valid_action != "fail_before_campaign_state_mutation":
            raise ValueError("unsupported Stage-3 insufficient-valid action")
        if self.rejected_candidates_consume_budget:
            raise ValueError("Stage-3 rejected pool candidates cannot consume budget")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeasiblePoolPolicy":
        if int(value.get("schema_version", -1)) != 1:
            raise ValueError("unsupported Stage-3 policy schema")
        return cls(
            proposal_pool_size=int(value["proposal_pool_size"]),
            proposal_pool_exploration_slots=int(
                value["proposal_pool_exploration_slots"]
            ),
            selection_order=str(value["selection_order"]),
            insufficient_valid_action=str(value["insufficient_valid_action"]),
            rejected_candidates_consume_budget=bool(
                value["rejected_candidates_consume_budget"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "proposal_pool_size": self.proposal_pool_size,
            "proposal_pool_exploration_slots": (
                self.proposal_pool_exploration_slots
            ),
            "selection_order": self.selection_order,
            "insufficient_valid_action": self.insufficient_valid_action,
            "rejected_candidates_consume_budget": (
                self.rejected_candidates_consume_budget
            ),
        }


def _preflight_raw_candidate(
    raw: np.ndarray,
    input_space: Any,
    coordinate_quantum_mm: float,
) -> tuple[bool, str | None, Mapping[str, Any]]:
    active = input_space.values(raw)
    parameters = baseline.full_parameter_mapping(active)
    valid, error, geometry = baseline.preflight_full_parameters(
        parameters,
        coordinate_quantum_mm=coordinate_quantum_mm,
    )
    return bool(valid), error, geometry


def filter_feasible_pool(
    proposal: krvea_relay.ProposalResult,
    *,
    input_space: Any,
    desired_q: int,
    desired_exploration_slots: int,
    coordinate_quantum_mm: float,
    actual_remaining_budget: int,
    policy: FeasiblePoolPolicy,
) -> krvea_relay.ProposalResult:
    """Select a valid expensive batch without creating penalty cases."""

    raw_values = np.asarray(proposal.raw_values, dtype=np.float64)
    unit_values = np.asarray(proposal.unit_values, dtype=np.float64)
    pool_count = len(raw_values)
    if pool_count != policy.proposal_pool_size:
        raise ValueError(
            f"Stage-3 worker returned {pool_count} candidates; "
            f"expected {policy.proposal_pool_size}"
        )
    diagnostics = dict(proposal.diagnostics)
    pool_exploration = int(diagnostics.get("reserved_exploration_count", 0))
    if desired_exploration_slots > pool_exploration:
        raise RuntimeError("Stage-3 pool omitted required exploration candidates")
    split = pool_count - pool_exploration

    valid: list[bool] = []
    errors: list[str | None] = []
    for raw in raw_values:
        candidate_valid, error, _ = _preflight_raw_candidate(
            raw, input_space, coordinate_quantum_mm
        )
        valid.append(candidate_valid)
        errors.append(error)
    exploitation_valid = [index for index in range(split) if valid[index]]
    exploration_valid = [index for index in range(split, pool_count) if valid[index]]
    exploitation_needed = desired_q - desired_exploration_slots
    if len(exploitation_valid) < exploitation_needed or len(
        exploration_valid
    ) < desired_exploration_slots:
        raise RuntimeError(
            "Stage-3 feasibility pool cannot fill the requested batch: "
            f"valid exploitation={len(exploitation_valid)}/{exploitation_needed}, "
            f"valid exploration={len(exploration_valid)}/"
            f"{desired_exploration_slots}; no campaign state was changed"
        )
    selected = [
        *exploitation_valid[:exploitation_needed],
        *exploration_valid[:desired_exploration_slots],
    ]
    selected_set = set(selected)
    references = list(diagnostics.get("selected_reference_indices", ()))
    rejected = []
    for index, candidate_valid in enumerate(valid):
        if candidate_valid:
            continue
        rejected.append(
            {
                "pool_index": index,
                "mode": "exploration" if index >= split else "exploitation",
                "reference_index": (
                    int(references[index]) if index < len(references) else None
                ),
                "error": errors[index],
                "unit_values": unit_values[index].tolist(),
                "raw_values": raw_values[index].tolist(),
            }
        )
    unused_valid = [
        index for index, candidate_valid in enumerate(valid)
        if candidate_valid and index not in selected_set
    ]

    pool_selection = {
        key: list(diagnostics.get(key, ())) for key in _SELECTION_VECTOR_KEYS
    }
    for key in _SELECTION_VECTOR_KEYS:
        values = list(diagnostics.get(key, ()))
        diagnostics[key] = [values[index] for index in selected]
    diagnostics.update(
        {
            "requested_q": desired_q,
            "proposed_count": desired_q,
            "expensive_budget_remaining": actual_remaining_budget,
            "reserved_exploration_count": desired_exploration_slots,
            "feasibility_pool": {
                **policy.to_dict(),
                "pool_candidate_count": pool_count,
                "pool_reserved_exploration_count": pool_exploration,
                "valid_candidate_count": int(sum(valid)),
                "infeasible_candidate_count": int(pool_count - sum(valid)),
                "selected_pool_indices": selected,
                "unused_valid_pool_indices": unused_valid,
                "rejected_infeasible": rejected,
                "remote_pool_selection": pool_selection,
            },
        }
    )
    if isinstance(diagnostics.get("krvea"), Mapping):
        filtered_core = dict(diagnostics["krvea"])
        for key in (*_SELECTION_VECTOR_KEYS, "requested_q", "proposed_count"):
            filtered_core[key] = diagnostics[key]
        filtered_core["expensive_budget_remaining"] = actual_remaining_budget
        filtered_core["reserved_exploration_count"] = desired_exploration_slots
        diagnostics["krvea"] = filtered_core

    return krvea_relay.ProposalResult(
        unit_values=unit_values[selected],
        raw_values=raw_values[selected],
        predicted_mean=np.asarray(proposal.predicted_mean)[selected],
        predicted_std=np.asarray(proposal.predicted_std)[selected],
        predicted_mean_standardized=np.asarray(
            proposal.predicted_mean_standardized
        )[selected],
        predicted_std_standardized=np.asarray(
            proposal.predicted_std_standardized
        )[selected],
        diagnostics=diagnostics,
    )


def load_config_document(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[
    Path,
    Mapping[str, Any],
    learned_relay.Stage2Policy,
    FeasiblePoolPolicy,
]:
    path = Path(config_path).expanduser().resolve()
    payload = _mapping(
        json.loads(path.read_text(encoding="utf-8-sig")), "document"
    )
    expected = set(deep_config._TOP_LEVEL_KEYS) | {
        "stage2_policy",
        "stage3_policy",
    }
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown or missing:
        raise ValueError(
            f"Stage-3 document fields mismatch: missing={missing}, unknown={unknown}"
        )
    stage2_payload = _mapping(payload["stage2_policy"], "stage2_policy")
    stage3_payload = _mapping(payload["stage3_policy"], "stage3_policy")
    for label, value, keys in (
        ("Stage-2 policy", stage2_payload, _STAGE2_POLICY_KEYS),
        ("Stage-3 policy", stage3_payload, _STAGE3_POLICY_KEYS),
    ):
        section_unknown = sorted(set(value) - keys)
        section_missing = sorted(keys - set(value))
        if section_unknown or section_missing:
            raise ValueError(
                f"{label} fields mismatch: missing={section_missing}, "
                f"unknown={section_unknown}"
            )
    inherited = {
        key: value
        for key, value in payload.items()
        if key not in {"stage2_policy", "stage3_policy"}
    }
    return (
        path,
        deep_config.validate_config_document(inherited),
        learned_relay.Stage2Policy.from_mapping(stage2_payload),
        FeasiblePoolPolicy.from_mapping(stage3_payload),
    )


def build_config(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    plan_id: str | None = None,
    source_directories: Sequence[Path] | None = None,
    output_directory: Path | None = None,
    total_budget: int | None = None,
    q: int | None = None,
    device_ids: Sequence[str] | None = None,
) -> tuple[
    baseline.CampaignConfig,
    learned_relay.Stage2Policy,
    FeasiblePoolPolicy,
]:
    path, document, uncertainty_policy, pool_policy = load_config_document(
        config_path
    )
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
        raise ValueError("Stage-3 configuration requires learned GP noise")
    if config.proposal.q >= pool_policy.proposal_pool_size:
        raise ValueError("Stage-3 proposal pool must exceed expensive q")
    if config.proposal.exploration_slots != 1:
        raise ValueError("Stage-3 requires one exploration slot when scheduled")
    if config.exploration_period_batches != 4:
        raise ValueError("Stage-3 exploration must run every fourth batch")
    return config, uncertainty_policy, pool_policy


def _request_remote_proposal(
    uncertainty_policy: learned_relay.Stage2Policy,
    pool_policy: FeasiblePoolPolicy,
    config: baseline.CampaignConfig,
    dataset: Any,
    *,
    batch_index: int,
    q: int,
    remaining_budget: int,
    previous_empty_reference_count: int | None,
) -> krvea_relay.ProposalResult:
    actual_settings = baseline.proposal_settings_for_batch(
        config, batch_index=batch_index, q=q
    )
    desired_exploration = actual_settings.exploration_slots
    pool_exploration = (
        pool_policy.proposal_pool_exploration_slots
        if desired_exploration
        else 0
    )
    pool_settings = replace(
        actual_settings,
        q=pool_policy.proposal_pool_size,
        exploration_slots=pool_exploration,
    )
    penalty_mask = ~dataset.metadata["has_completed_result"].to_numpy(dtype=bool)
    request = krvea_relay.build_request_payload(
        dataset.x_unit,
        dataset.objectives[:, [0, 1, 3]],
        dataset.objectives,
        penalty_mask,
        dataset.input_space,
        config=pool_settings,
        iteration=batch_index,
        remaining_expensive_budget=pool_policy.proposal_pool_size,
        previous_empty_reference_count=previous_empty_reference_count,
        compute_device=config.proposal_remote.compute_device,
        surrogate_settings=baseline.campaign_surrogate_fit_settings(config),
    )
    request = learned_relay.attach_policy(request, uncertainty_policy)
    control = baseline._control_directory(config)
    request_path = control / f"batch_{batch_index:04d}_proposal_request.json"
    response_path = control / f"batch_{batch_index:04d}_proposal_response.json"
    krvea_relay.write_request(request_path, request)
    registry = load_device_registry(config.device_config)
    device = select_devices(registry, (config.proposal_remote.device_id,))[0]
    pool = learned_relay.relay_remote_proposal(
        device=device,
        remote=config.proposal_remote,
        plan_id=config.plan_id,
        batch_index=batch_index,
        local_request_path=request_path,
        local_response_path=response_path,
        expected_q=pool_policy.proposal_pool_size,
        expected_dimension=len(dataset.input_space.names),
        observed_x_unit=np.asarray(dataset.x_unit, dtype=np.float64),
        input_space=dataset.input_space,
    )
    return filter_feasible_pool(
        pool,
        input_space=dataset.input_space,
        desired_q=q,
        desired_exploration_slots=desired_exploration,
        coordinate_quantum_mm=config.coordinate_quantum_mm,
        actual_remaining_budget=remaining_budget,
        policy=pool_policy,
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
) -> tuple[
    baseline.CampaignConfig,
    learned_relay.Stage2Policy,
    FeasiblePoolPolicy,
]:
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
        config, uncertainty_policy, pool_policy = config_from_args(args)
        print(f"[Deep Stage 3] config={Path(args.config).resolve()}", flush=True)
        print(
            f"[Deep Stage 3] expensive q={config.proposal.q}, "
            f"proposal pool={pool_policy.proposal_pool_size}, "
            f"exploration period={config.exploration_period_batches}",
            flush=True,
        )
        if not args.prepare_only and not args.stop_after_proposal:
            if F5_REQUIRE_CONFIRMATION and not args.yes:
                answer = input(
                    "Type RUN to start/resume Stage-3 feasible K-RVEA plan "
                    f"{config.plan_id} ({config.total_budget} valid proposals): "
                )
                if answer.strip() != "RUN":
                    print("Cancelled; no proposal worker or solver was started.")
                    return 1

        original = baseline._request_remote_proposal
        baseline._request_remote_proposal = partial(
            _request_remote_proposal,
            uncertainty_policy,
            pool_policy,
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
        print("[Deep Stage 3] interrupted; campaign state remains resumable")
        return 130
    except Exception as exc:
        print(f"Deep Stage-3 error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
