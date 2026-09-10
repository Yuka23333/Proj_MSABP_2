"""Run the isolated 11-D, three-objective Phase-2 K-RVEA campaign.

Objectives (all minimized internally): worst in-band |S11|, negative frozen
ROI E-theta radiation gain in dBi, and exact normalized substrate area.  The
ROI radiation-gain objective already contains radiation efficiency, so Phase 2
does not fit Rad_Eff as a separate fourth target.

The historical controller is reused only for durable state, Princess/Maid
dispatch, and recovery.  This entrypoint installs a process-local Phase-2 data
and relay adapter and restores the historical module before exit.  It never
changes the meaning of an existing four-objective campaign.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import krvea  # noqa: E402
from msabp_opt.optimization import phase2_krvea_data as phase2_data  # noqa: E402
from msabp_opt.optimization import phase2_krvea_relay as phase2_relay  # noqa: E402
from msabp_opt.simulation.distributed.config import load_device_registry  # noqa: E402
from msabp_opt.simulation.distributed.runtime import (  # noqa: E402
    select_devices,
    validate_run_id,
)
from scripts.optimization import run_krvea as baseline  # noqa: E402


CONFIG_SCHEMA_VERSION = 2
PLAN_SCHEMA_VERSION = 3
STRATEGY_NAME = "phase2_fixed_roi_radiation_gain_v1"
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "optimization"
    / "phase2_krvea_roi_radiation_gain_64.json"
)
F5_REQUIRE_CONFIRMATION = True
ROI_CACHE_DIRECTORY_NAME = "phase2_roi_radiation_gain_cache"
PENALTY_WORST_S11 = 1.0

_TOP_LEVEL_KEYS = {
    "schema_version",
    "campaign",
    "strategy",
    "surrogate",
    "proposal_remote",
    "simulation",
    "phase2_metric",
}
_CAMPAIGN_KEYS = {
    "plan_id",
    "source_directories",
    "output_directory",
    "total_budget",
    "q",
    "band_ghz",
    "device_ids",
}
_STRATEGY_KEYS = {
    "name",
    "seed",
    "reference_partitions",
    "inner_evaluations",
    "population_size",
    "crossover_probability",
    "crossover_eta",
    "mutation_probability",
    "mutation_eta",
    "apd_alpha",
    "empty_growth_fraction",
    "uniqueness_tolerance",
    "conservative_beta",
    "uncertainty_scale_mode",
    "exploration_slots",
    "exploration_period_batches",
    "exploration_novelty_weight",
    "exploration_pool_size",
}
_SURROGATE_KEYS = {
    "gp_training_steps",
    "gp_kernel",
    "gp_noise_mode",
    "gp_fixed_noise_variance",
    "gp_learned_noise_floor",
    "gp_learned_noise_initial_variance",
    "gp_posterior_observation_noise",
    "gp_timeout_seconds",
    "uncertainty_calibration_factors",
    "uncertainty_calibration_source",
    "bounded_moment_quadrature_order",
    "support_distance_quantile",
    "support_uncertainty_power",
    "support_uncertainty_cap",
}
_REMOTE_KEYS = {"device_id", "python_path", "compute_device", "timeout_seconds"}
_SIMULATION_KEYS = {
    "sampling_config",
    "device_config",
    "project_template",
    "coordinate_quantum_mm",
    "allow_disconnected_conductor",
    "max_attempts",
}
_METRIC_KEYS = {
    "name",
    "frequency_ghz",
    "theta_bounds_deg",
    "phi_center_deg",
    "phi_half_width_deg",
    "component",
    "gain_type",
    "spatial_average",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase-2 {label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Phase-2 {label} must be a JSON array")
    return value


def _check_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(
            f"Phase-2 {label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _validate_metric(metric: Mapping[str, Any]) -> None:
    _check_fields(metric, _METRIC_KEYS, "phase2_metric")
    expected = {
        "name": "frozen_roi_theta_radiation_gain_v1",
        "frequency_ghz": phase2_data.ROI_FREQUENCY_GHZ,
        "theta_bounds_deg": list(phase2_data.ROI_THETA_BOUNDS_DEG),
        "phi_center_deg": phase2_data.ROI_PHI_CENTER_DEG,
        "phi_half_width_deg": phase2_data.ROI_PHI_HALF_WIDTH_DEG,
        "component": "E_theta",
        "gain_type": "radiation_gain",
        "spatial_average": "linear_power_solid_angle_weighted",
    }
    actual = json.loads(json.dumps(dict(metric)))
    if actual != expected:
        raise ValueError(
            "Phase-2 metric contract is frozen; create a new entry/schema for another ROI"
        )


def load_config_document(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[Path, Mapping[str, Any]]:
    path = Path(config_path).expanduser().resolve()
    document = _mapping(
        json.loads(path.read_text(encoding="utf-8-sig")),
        "document",
    )
    _check_fields(document, _TOP_LEVEL_KEYS, "document")
    if int(document["schema_version"]) != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Phase-2 config schema {document['schema_version']!r}; "
            f"expected {CONFIG_SCHEMA_VERSION}"
        )
    for name, fields in (
        ("campaign", _CAMPAIGN_KEYS),
        ("strategy", _STRATEGY_KEYS),
        ("surrogate", _SURROGATE_KEYS),
        ("proposal_remote", _REMOTE_KEYS),
        ("simulation", _SIMULATION_KEYS),
    ):
        _check_fields(_mapping(document[name], name), fields, name)
    _validate_metric(_mapping(document["phase2_metric"], "phase2_metric"))
    return path, document


def build_config(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    plan_id: str | None = None,
    source_directories: Sequence[Path] | None = None,
    output_directory: Path | None = None,
    total_budget: int | None = None,
    q: int | None = None,
    device_ids: Sequence[str] | None = None,
) -> baseline.CampaignConfig:
    path, document = load_config_document(config_path)
    campaign = _mapping(document["campaign"], "campaign")
    strategy = _mapping(document["strategy"], "strategy")
    surrogate = _mapping(document["surrogate"], "surrogate")
    remote = _mapping(document["proposal_remote"], "proposal_remote")
    simulation = _mapping(document["simulation"], "simulation")
    if str(strategy["name"]) != STRATEGY_NAME:
        raise ValueError(
            f"this entrypoint only accepts strategy.name={STRATEGY_NAME!r}"
        )

    campaign_q = int(q if q is not None else campaign["q"])
    population_size = strategy["population_size"]
    mutation_probability = strategy["mutation_probability"]
    proposal = krvea.KRVEAConfig(
        n_variables=len(phase2_data.ACTIVE_PARAMETER_NAMES),
        n_objectives=len(phase2_data.OBJECTIVE_NAMES),
        reference_partitions=int(strategy["reference_partitions"]),
        q=campaign_q,
        inner_evaluations=int(strategy["inner_evaluations"]),
        population_size=None if population_size is None else int(population_size),
        seed=int(strategy["seed"]),
        crossover_probability=float(strategy["crossover_probability"]),
        crossover_eta=float(strategy["crossover_eta"]),
        mutation_probability=(
            None if mutation_probability is None else float(mutation_probability)
        ),
        mutation_eta=float(strategy["mutation_eta"]),
        apd_alpha=float(strategy["apd_alpha"]),
        empty_growth_fraction=float(strategy["empty_growth_fraction"]),
        uniqueness_tolerance=float(strategy["uniqueness_tolerance"]),
        conservative_beta=float(strategy["conservative_beta"]),
        uncertainty_scale_mode=str(strategy["uncertainty_scale_mode"]),
        exploration_slots=int(strategy["exploration_slots"]),
        exploration_novelty_weight=float(strategy["exploration_novelty_weight"]),
        exploration_pool_size=int(strategy["exploration_pool_size"]),
    )
    factors = tuple(
        float(value)
        for value in _sequence(
            surrogate["uncertainty_calibration_factors"],
            "surrogate.uncertainty_calibration_factors",
        )
    )
    surrogate_settings = phase2_relay.SurrogateFitSettings(
        gp_training_steps=int(surrogate["gp_training_steps"]),
        gp_kernel=str(surrogate["gp_kernel"]),
        gp_noise_mode=str(surrogate["gp_noise_mode"]),
        gp_fixed_noise_variance=float(surrogate["gp_fixed_noise_variance"]),
        gp_learned_noise_floor=float(surrogate["gp_learned_noise_floor"]),
        gp_learned_noise_initial_variance=float(
            surrogate["gp_learned_noise_initial_variance"]
        ),
        gp_posterior_observation_noise=bool(
            surrogate["gp_posterior_observation_noise"]
        ),
        gp_timeout_seconds=float(surrogate["gp_timeout_seconds"]),
        uncertainty_calibration_factors=factors,  # type: ignore[arg-type]
        uncertainty_calibration_source=str(
            surrogate["uncertainty_calibration_source"]
        ),
        bounded_moment_quadrature_order=int(
            surrogate["bounded_moment_quadrature_order"]
        ),
        support_distance_quantile=float(surrogate["support_distance_quantile"]),
        support_uncertainty_power=float(surrogate["support_uncertainty_power"]),
        support_uncertainty_cap=float(surrogate["support_uncertainty_cap"]),
    )
    configured_sources = tuple(
        _repo_path(value)
        for value in _sequence(
            campaign["source_directories"],
            "campaign.source_directories",
        )
    )
    configured_devices = tuple(
        str(value)
        for value in _sequence(campaign["device_ids"], "campaign.device_ids")
    )
    band = _sequence(campaign["band_ghz"], "campaign.band_ghz")
    if len(band) != 2:
        raise ValueError("Phase-2 campaign.band_ghz needs two values")
    return baseline.CampaignConfig(
        plan_id=str(plan_id if plan_id is not None else campaign["plan_id"]),
        source_directories=(
            configured_sources
            if source_directories is None
            else tuple(Path(value) for value in source_directories)
        ),
        output_directory=(
            _repo_path(campaign["output_directory"])
            if output_directory is None
            else Path(output_directory)
        ),
        total_budget=int(
            total_budget if total_budget is not None else campaign["total_budget"]
        ),
        band_ghz=(float(band[0]), float(band[1])),
        device_ids=(
            configured_devices
            if device_ids is None
            else tuple(str(value) for value in device_ids)
        ),
        sampling_config=_repo_path(simulation["sampling_config"]),
        device_config=_repo_path(simulation["device_config"]),
        project_template=_repo_path(simulation["project_template"]),
        proposal=proposal,
        proposal_remote=baseline.RemoteProposalConfig(
            device_id=str(remote["device_id"]),
            python_path=str(remote["python_path"]),
            compute_device=str(remote["compute_device"]),
            timeout_seconds=float(remote["timeout_seconds"]),
        ),
        coordinate_quantum_mm=float(simulation["coordinate_quantum_mm"]),
        allow_disconnected_conductor=bool(
            simulation["allow_disconnected_conductor"]
        ),
        max_attempts=int(simulation["max_attempts"]),
        surrogate_settings=surrogate_settings,
        exploration_period_batches=int(strategy["exploration_period_batches"]),
        strategy_name=STRATEGY_NAME,
        strategy_source=Path(__file__).resolve(),
        strategy_config_source=path,
    )


def _validate_config(config: baseline.CampaignConfig) -> baseline.CampaignConfig:
    validate_run_id(config.plan_id)
    if config.total_budget <= 0:
        raise ValueError("Phase-2 K-RVEA evaluation budget must be positive")
    if config.proposal.n_variables != len(phase2_data.ACTIVE_PARAMETER_NAMES):
        raise ValueError("Phase-2 K-RVEA must use the authoritative 11 variables")
    if config.proposal.n_objectives != len(phase2_data.OBJECTIVE_NAMES):
        raise ValueError("Phase-2 K-RVEA must use exactly three objectives")
    if config.exploration_period_batches <= 0:
        raise ValueError("exploration_period_batches must be positive")
    if not config.source_directories:
        raise ValueError("Phase-2 K-RVEA requires historical sources")
    if not config.device_ids or len(set(config.device_ids)) != len(config.device_ids):
        raise ValueError("Princess device IDs must be non-empty and unique")
    if not config.band_ghz[0] < config.band_ghz[1]:
        raise ValueError("band must satisfy low < high")
    sources = baseline._unique_resolved_paths(config.source_directories)
    output = config.output_directory.expanduser().resolve()
    if any(str(output).casefold() == str(source).casefold() for source in sources):
        raise ValueError("the output is an automatic source and cannot be historical")
    missing_sources = [source for source in sources if not source.is_dir()]
    if missing_sources:
        raise FileNotFoundError(f"historical source does not exist: {missing_sources[0]}")
    for candidate, label in (
        (config.sampling_config, "sampling config"),
        (config.device_config, "device config"),
        (config.project_template, "CST project template"),
    ):
        if not Path(candidate).is_file():
            raise FileNotFoundError(f"{label} does not exist: {candidate}")
    strategy_source = config.strategy_source.resolve() if config.strategy_source else None
    strategy_config = (
        config.strategy_config_source.resolve()
        if config.strategy_config_source
        else None
    )
    for candidate, label in (
        (strategy_source, "strategy source"),
        (strategy_config, "strategy config"),
    ):
        if candidate is not None and not candidate.is_file():
            raise FileNotFoundError(f"{label} does not exist: {candidate}")
    baseline._validate_sampling_contract(config.sampling_config)
    if not np.isclose(
        phase2_data.reference_substrate_area_mm2(),
        baseline.NOMINAL_AREA_REFERENCE_MM2,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError("authoritative nominal substrate area is no longer 2720.2 mm2")
    return replace(
        config,
        source_directories=sources,
        output_directory=output,
        sampling_config=config.sampling_config.resolve(),
        device_config=config.device_config.resolve(),
        project_template=config.project_template.resolve(),
        strategy_source=strategy_source,
        strategy_config_source=strategy_config,
    )


def _history_cache_contract(config: baseline.CampaignConfig) -> dict[str, Any]:
    sources = {
        "cap_gain": REPOSITORY_ROOT / "scripts" / "postprocessing" / "cap_gain.py",
        "prepare_link_ffs_tensor": (
            REPOSITORY_ROOT / "scripts" / "postprocessing" / "prepare_link_ffs_tensor.py"
        ),
        "search_link_spherical_roi": (
            REPOSITORY_ROOT / "scripts" / "postprocessing" / "search_link_spherical_roi.py"
        ),
    }
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_snapshots": [
            {"source_directory": str(source), **baseline._manifest_snapshot(source)}
            for source in config.source_directories
        ],
        "band_ghz": list(config.band_ghz),
        "sampling_config_sha256": baseline._sha256(config.sampling_config),
        "minimum_trainable_count": baseline.MINIMUM_INITIAL_TRAINING_COUNT,
        "objective_extractor": {
            "schema_version": phase2_data.SCHEMA_VERSION,
            "roi_metric_version": phase2_data.ROI_METRIC_VERSION,
            "phase2_data_sha256": baseline._sha256(phase2_data.__file__),
            **{
                f"{name}_sha256": baseline._sha256(source)
                for name, source in sources.items()
            },
        },
    }


def _plan_payload(
    config: baseline.CampaignConfig,
    input_space: phase2_data.InputSpace,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    historical_count = int(source_snapshot.get("trainable_count", 0))
    if historical_count < baseline.MINIMUM_INITIAL_TRAINING_COUNT:
        raise ValueError("historical snapshot lacks the minimum trainable count")
    implementation = {
        "phase2_entry": baseline._sha256(__file__),
        "historical_controller": baseline._sha256(baseline.__file__),
        "krvea": baseline._sha256(krvea.__file__),
        "phase2_data": baseline._sha256(phase2_data.__file__),
        "phase2_relay": baseline._sha256(phase2_relay.__file__),
    }
    if config.strategy_config_source is not None:
        implementation["strategy_config"] = baseline._sha256(
            config.strategy_config_source
        )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "objective_schema": "msabp_phase2_fixed_roi_three_objective_v1",
        "plan_id": config.plan_id,
        "created_at_utc": baseline._utc_now(),
        "algorithm": "K-RVEA",
        "strategy": {
            "name": config.strategy_name,
            "source": str(config.strategy_source),
            "config_source": str(config.strategy_config_source),
        },
        "provenance": {
            "implementation": "independent Python reimplementation",
            "upstream_repository": "https://github.com/tichugh/K-RVEA.git",
            "upstream_commit_reviewed": baseline.UPSTREAM_KRVEA_COMMIT,
            "phase_boundary": "does_not_reinterpret_phase1_four_objective_campaigns",
        },
        "campaign": (
            f"{historical_count} historical + {config.total_budget} new expensive evaluations"
        ),
        "total_budget": config.total_budget,
        "q": config.proposal.q,
        "band_ghz": list(config.band_ghz),
        "source_directories": [str(path) for path in config.source_directories],
        "historical_training_count": historical_count,
        "historical_source_snapshot": dict(source_snapshot),
        "output_directory": str(config.output_directory),
        "output_is_automatic_training_source": True,
        "sampling_config": str(config.sampling_config),
        "sampling_config_sha256": baseline._sha256(config.sampling_config),
        "input_space": {
            "parameter_names": list(input_space.names),
            "lower": input_space.lower.tolist(),
            "upper": input_space.upper.tolist(),
            "normalization": "x_unit=(x_raw-lower)/(upper-lower)",
            "fixed_parameter_values": dict(phase2_data.FIXED_PARAMETER_VALUES),
        },
        "objectives": [
            {
                "name": phase2_data.WORST_S11_COLUMN,
                "direction": "minimize",
                "domain": "linear_amplitude",
                "frequency_band_ghz": list(config.band_ghz),
                "reduction": "maximum",
                "penalty": PENALTY_WORST_S11,
            },
            {
                "name": phase2_data.ROI_GAIN_LOSS_DBI_COLUMN,
                "reported_metric": phase2_data.ROI_GAIN_DBI_COLUMN,
                "direction": "minimize negative radiation gain",
                "component": "E_theta",
                "gain_type": "radiation_gain",
                "frequency_ghz": phase2_data.ROI_FREQUENCY_GHZ,
                "theta_bounds_deg": list(phase2_data.ROI_THETA_BOUNDS_DEG),
                "phi_center_deg": phase2_data.ROI_PHI_CENTER_DEG,
                "phi_half_width_deg": phase2_data.ROI_PHI_HALF_WIDTH_DEG,
                "averaging": "spatial linear-power solid-angle average, then dBi",
                "rad_eff_handling": "absorbed_once_into_radiation_gain",
                "mismatch_handling": "excluded_to_avoid_double_counting_S11",
                "penalty": phase2_data.PENALTY_ROI_GAIN_LOSS_DBI,
            },
            {
                "name": phase2_data.NORMALIZED_AREA_COLUMN,
                "direction": "minimize",
                "model": "exact_deterministic_formula",
                "posterior_variance": 0.0,
                "definition": "substrate_area_mm2 / 2720.2",
                "nominal_area_reference_mm2": baseline.NOMINAL_AREA_REFERENCE_MM2,
                "nominal_design_value": 1.0,
                "penalty": phase2_data.PENALTY_NORMALIZED_AREA,
            },
        ],
        "proposal": {
            **asdict(config.proposal),
            "expensive_objective_indices": list(
                phase2_relay.EXPENSIVE_OBJECTIVE_INDICES
            ),
            "exact_objective_indices": list(phase2_relay.EXACT_OBJECTIVE_INDICES),
            "dtype": "float64",
            "target_standardization": (
                "logit for bounded S11, identity for negative ROI gain dBi, "
                "then per-target robust standardization"
            ),
            "bounded_target_moments": "Gauss-Hermite logit-normal moments",
            "penalty_rows_in_gp_fit": False,
            "selection_risk_rule": "mu + conservative_beta * calibrated_sigma",
            "exploration_schedule": {
                "slots_per_exploration_batch": config.proposal.exploration_slots,
                "period_batches": config.exploration_period_batches,
                "first_exploration_batch_index": 0,
            },
            "surrogate_settings": asdict(
                baseline.campaign_surrogate_fit_settings(config)
            ),
            "remote": config.proposal_remote.to_dict(),
        },
        "simulation": {
            "device_ids": list(config.device_ids),
            "device_config": str(config.device_config),
            "project_template": str(config.project_template),
            "coordinate_quantum_mm": config.coordinate_quantum_mm,
            "allow_disconnected_conductor": config.allow_disconnected_conductor,
            "max_attempts": config.max_attempts,
        },
        "software": {
            "controller_python": platform.python_version(),
            "implementation_sha256": implementation,
        },
    }


def _penalty_objectives(
    config: baseline.CampaignConfig,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    _, _, area = phase2_data.substrate_dimensions(parameters)
    roi_loss = phase2_data.PENALTY_ROI_GAIN_LOSS_DBI
    return {
        "band_ghz": list(config.band_ghz),
        phase2_data.WORST_S11_COLUMN: PENALTY_WORST_S11,
        phase2_data.ROI_GAIN_LOSS_DBI_COLUMN: roi_loss,
        phase2_data.ROI_GAIN_DBI_COLUMN: -roi_loss,
        phase2_data.ROI_GAIN_LINEAR_COLUMN: float(np.power(10.0, -roi_loss / 10.0)),
        phase2_data.AREA_COLUMN: area,
        phase2_data.NORMALIZED_AREA_COLUMN: phase2_data.PENALTY_NORMALIZED_AREA,
        "is_penalty": True,
    }


def _request_remote_proposal(
    config: baseline.CampaignConfig,
    dataset: phase2_data.Dataset,
    *,
    batch_index: int,
    q: int,
    remaining_budget: int,
    previous_empty_reference_count: int | None,
) -> phase2_relay.ProposalResult:
    settings = baseline.proposal_settings_for_batch(
        config,
        batch_index=batch_index,
        q=q,
    )
    penalty_mask = ~dataset.metadata["has_completed_result"].to_numpy(dtype=bool)
    request = phase2_relay.build_request_payload(
        dataset.x_unit,
        dataset.objectives[:, list(phase2_relay.EXPENSIVE_OBJECTIVE_INDICES)],
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
    control = baseline._control_directory(config)
    request_path = control / f"batch_{batch_index:04d}_proposal_request.json"
    response_path = control / f"batch_{batch_index:04d}_proposal_response.json"
    phase2_relay.write_request(request_path, request)
    registry = load_device_registry(config.device_config)
    device = select_devices(registry, (config.proposal_remote.device_id,))[0]
    remote = phase2_relay.RemoteProposalConfig(**config.proposal_remote.to_dict())
    return phase2_relay.relay_remote_proposal(
        device=device,
        remote=remote,
        plan_id=config.plan_id,
        batch_index=batch_index,
        local_request_path=request_path,
        local_response_path=response_path,
        expected_q=q,
        expected_dimension=len(dataset.input_space.names),
        observed_x_unit=dataset.x_unit,
        input_space=dataset.input_space,
    )


@contextmanager
def _phase2_controller_adapter() -> Iterator[None]:
    replacements = {
        "krvea_data": phase2_data,
        "CAP_CACHE_DIRECTORY_NAME": ROI_CACHE_DIRECTORY_NAME,
        "_validate_config": _validate_config,
        "_history_cache_contract": _history_cache_contract,
        "_plan_payload": _plan_payload,
        "_penalty_objectives": _penalty_objectives,
        "_request_remote_proposal": _request_remote_proposal,
    }
    original = {name: getattr(baseline, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(baseline, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(baseline, name, value)


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


def config_from_args(args: argparse.Namespace) -> baseline.CampaignConfig:
    return build_config(
        config_path=args.config,
        plan_id=args.plan_id,
        source_directories=tuple(args.sources) if args.sources else None,
        output_directory=args.output,
        total_budget=args.budget,
        q=args.q,
        device_ids=tuple(args.device_ids) if args.device_ids else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = config_from_args(args)
        print(f"[Phase-2 K-RVEA] config={Path(args.config).resolve()}", flush=True)
        print(
            "[Phase-2 K-RVEA] objectives=min(worst |S11|), "
            "max(3.6-GHz frozen-ROI E_theta radiation gain), min(area)",
            flush=True,
        )
        if not args.prepare_only and not args.stop_after_proposal:
            if F5_REQUIRE_CONFIRMATION and not args.yes:
                answer = input(
                    "Type RUN to start/resume Phase-2 K-RVEA plan "
                    f"{config.plan_id} ({config.total_budget} evaluations): "
                )
                if answer.strip() != "RUN":
                    print("Cancelled; no proposal worker or solver was started.")
                    return 1
        with _phase2_controller_adapter():
            return baseline.run_campaign(
                config,
                prepare_only=args.prepare_only,
                stop_after_proposal=args.stop_after_proposal,
            )
    except KeyboardInterrupt:
        print("[Phase-2 K-RVEA] interrupted; campaign state remains resumable")
        return 130
    except Exception as exc:
        print(f"Phase-2 K-RVEA error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
