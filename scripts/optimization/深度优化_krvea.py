"""Run JSON-configured late-stage four-objective K-RVEA continuations.

This entrypoint deliberately does not replace :mod:`run_krvea`.  The strategy
identity remains fixed in this file; round-specific numerical parameters and
campaign paths live in an immutable JSON file.  A real run still requires
typing ``RUN`` unless ``--yes`` is supplied.  ``--prepare-only`` never starts
SSH, Princess, Maid, or CST.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.optimization import krvea, krvea_data, krvea_relay  # noqa: E402
from scripts.optimization import run_krvea as baseline  # noqa: E402


CONFIG_SCHEMA_VERSION = 1
STRATEGY_NAME = "deep_late_stage_s11_guard_v1"
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "optimization" / "deep_krvea_round5.json"
)
F5_REQUIRE_CONFIRMATION = True

_TOP_LEVEL_KEYS = {
    "schema_version",
    "campaign",
    "strategy",
    "surrogate",
    "proposal_remote",
    "simulation",
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
_REMOTE_KEYS = {
    "device_id",
    "python_path",
    "compute_device",
    "timeout_seconds",
}
_SIMULATION_KEYS = {
    "sampling_config",
    "device_config",
    "project_template",
    "coordinate_quantum_mm",
    "allow_disconnected_conductor",
    "max_attempts",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"deep optimization {label} must be a JSON object")
    return value


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"unknown deep optimization {label} field(s): {', '.join(unknown)}"
        )


def _require(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ValueError(
            f"missing deep optimization {label} field(s): {', '.join(missing)}"
        )


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"deep optimization {label} must be a JSON array")
    return value


def validate_config_document(payload: Any) -> Mapping[str, Any]:
    """Strictly validate one already-loaded late-stage JSON document."""

    document = _mapping(payload, "document")
    _reject_unknown(document, _TOP_LEVEL_KEYS, "document")
    _require(document, _TOP_LEVEL_KEYS, "document")
    if int(document["schema_version"]) != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "unsupported deep optimization schema_version "
            f"{document['schema_version']!r}; expected {CONFIG_SCHEMA_VERSION}"
        )
    sections = (
        ("campaign", _CAMPAIGN_KEYS),
        ("strategy", _STRATEGY_KEYS),
        ("surrogate", _SURROGATE_KEYS),
        ("proposal_remote", _REMOTE_KEYS),
        ("simulation", _SIMULATION_KEYS),
    )
    for name, allowed in sections:
        section = _mapping(document[name], name)
        _reject_unknown(section, allowed, name)
        _require(section, allowed, name)
    return document


def load_config_document(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[Path, Mapping[str, Any]]:
    """Load and strictly validate the JSON document shape."""

    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    document = validate_config_document(payload)
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
    _document: Mapping[str, Any] | None = None,
    _strategy_name: str = STRATEGY_NAME,
    _strategy_source: Path | None = None,
) -> baseline.CampaignConfig:
    """Build one deep-stage campaign from JSON plus explicit CLI overrides."""

    path = Path(config_path).expanduser().resolve()
    if _document is None:
        path, document = load_config_document(path)
    else:
        document = validate_config_document(_document)
    campaign = _mapping(document["campaign"], "campaign")
    strategy = _mapping(document["strategy"], "strategy")
    surrogate = _mapping(document["surrogate"], "surrogate")
    remote = _mapping(document["proposal_remote"], "proposal_remote")
    simulation = _mapping(document["simulation"], "simulation")

    if str(strategy["name"]) != _strategy_name:
        raise ValueError(
            "this entrypoint only accepts strategy.name "
            f"{_strategy_name!r}; create another 深度优化_*.py for a new strategy"
        )
    campaign_q = int(q if q is not None else campaign["q"])
    population_size = strategy["population_size"]
    mutation_probability = strategy["mutation_probability"]
    proposal = krvea.KRVEAConfig(
        n_variables=len(krvea_data.ACTIVE_PARAMETER_NAMES),
        n_objectives=4,
        reference_partitions=int(strategy["reference_partitions"]),
        q=campaign_q,
        inner_evaluations=int(strategy["inner_evaluations"]),
        population_size=(
            None if population_size is None else int(population_size)
        ),
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
        exploration_novelty_weight=float(
            strategy["exploration_novelty_weight"]
        ),
        exploration_pool_size=int(strategy["exploration_pool_size"]),
    )
    factors = tuple(
        float(value)
        for value in _sequence(
            surrogate["uncertainty_calibration_factors"],
            "surrogate.uncertainty_calibration_factors",
        )
    )
    surrogate_settings = krvea_relay.SurrogateFitSettings(
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
        support_distance_quantile=float(
            surrogate["support_distance_quantile"]
        ),
        support_uncertainty_power=float(
            surrogate["support_uncertainty_power"]
        ),
        support_uncertainty_cap=float(surrogate["support_uncertainty_cap"]),
    )
    configured_sources = tuple(
        _repo_path(value)
        for value in _sequence(
            campaign["source_directories"], "campaign.source_directories"
        )
    )
    configured_devices = tuple(
        str(value)
        for value in _sequence(campaign["device_ids"], "campaign.device_ids")
    )
    band = _sequence(campaign["band_ghz"], "campaign.band_ghz")
    if len(band) != 2:
        raise ValueError("deep optimization campaign.band_ghz needs two values")

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
        strategy_name=_strategy_name,
        strategy_source=(
            Path(__file__).resolve()
            if _strategy_source is None
            else Path(_strategy_source).resolve()
        ),
        strategy_config_source=path,
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


def config_from_args(args: argparse.Namespace) -> baseline.CampaignConfig:
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
        config = config_from_args(args)
        print(
            f"[Deep K-RVEA] config={Path(args.config).expanduser().resolve()}",
            flush=True,
        )
        if not args.prepare_only and not args.stop_after_proposal:
            if F5_REQUIRE_CONFIRMATION and not args.yes:
                answer = input(
                    "Type RUN to start/resume deep K-RVEA plan "
                    f"{config.plan_id} ({config.total_budget} new evaluations): "
                )
                if answer.strip() != "RUN":
                    print("Cancelled; no proposal worker or solver was started.")
                    return 1
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
