"""Hash-addressed relay for K-RVEA proposals executed on ``coconutg2``.

The Princess-side controller owns campaign state and CST observations.  This
module sends only compact numeric arrays to the proposal worker, fits the
three expensive Gaussian-process models in float64 on the requested device,
and runs the NumPy K-RVEA model-management step.  Substrate area is evaluated
exactly and therefore has zero predictive uncertainty.

Every objective uses minimization semantics in this order::

    [worst |S11|, 1 - mean(Tot_Eff), normalized area, cap gain dBi]

The cap value is expected to have been averaged over angle and frequency in
linear power before conversion to dBi by :mod:`krvea_data`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from msabp_opt.simulation.distributed.config import DeviceConfig
from msabp_opt.simulation.distributed.transport import (
    pull_file_atomic,
    push_file_atomic,
    run_remote_powershell,
)

from . import krvea


REQUEST_SCHEMA_VERSION = 1
RESPONSE_SCHEMA_VERSION = 1
DEFAULT_REMOTE_TIMEOUT_SECONDS = 1800.0
DEFAULT_REMOTE_WORK_ROOT = PureWindowsPath("simulations", "runs", "krvea_gpu")

EXPENSIVE_OBJECTIVE_INDICES = (0, 1, 3)
EXACT_OBJECTIVE_INDICES = (2,)
OBJECTIVE_NAMES = (
    "worst_s11_linear_amplitude",
    "one_minus_mean_total_efficiency_linear",
    "normalized_substrate_area",
    "cap_realized_gain_dbi",
)
EXPENSIVE_OBJECTIVE_NAMES = tuple(
    OBJECTIVE_NAMES[index] for index in EXPENSIVE_OBJECTIVE_INDICES
)

EXACT_AREA_CONTRACT_TYPE = "msabp_normalized_substrate_area_v1"
REFERENCE_SUBSTRATE_AREA_MM2 = 2720.2
RESPONSE_DUPLICATE_TOLERANCE = 1e-10


class InputSpaceLike(Protocol):
    names: Sequence[str]
    lower: Sequence[float]
    upper: Sequence[float]


@dataclass(frozen=True)
class WireInputSpace:
    """Minimal input-space representation shared by both hosts."""

    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        expected = (len(self.names),)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("input-space parameter names must be non-empty and unique")
        if lower.shape != expected or upper.shape != expected:
            raise ValueError("input-space arrays do not match parameter names")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("input-space bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("every input-space upper bound must exceed its lower bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_names": list(self.names),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }

    def denormalize(self, unit_values: np.ndarray) -> np.ndarray:
        unit = np.asarray(unit_values, dtype=np.float64)
        return self.lower + unit * (self.upper - self.lower)


@dataclass(frozen=True)
class SurrogateFitSettings:
    """Numerical settings for the three independent batched GPs."""

    gp_training_steps: int = 50
    gp_fixed_noise_variance: float = 1e-6
    gp_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if self.gp_training_steps <= 0:
            raise ValueError("gp_training_steps must be positive")
        if self.gp_fixed_noise_variance <= 0.0:
            raise ValueError("gp_fixed_noise_variance must be positive")
        if self.gp_timeout_seconds <= 0.0:
            raise ValueError("gp_timeout_seconds must be positive")


@dataclass(frozen=True)
class RemoteProposalConfig:
    """How Princess reaches the proposal-only GPU worker."""

    device_id: str = "coconutg2"
    python_path: str = r"C:\Users\telecom\miniforge3\envs\bocuda\python.exe"
    compute_device: str = "cuda"
    timeout_seconds: float = DEFAULT_REMOTE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("proposal device_id must be non-empty")
        if not PureWindowsPath(self.python_path).is_absolute():
            raise ValueError("proposal python_path must be an absolute Windows path")
        if not self.compute_device.strip():
            raise ValueError("proposal compute_device must be non-empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("proposal timeout_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveScaler:
    """Robust physical-to-model scaling for the four objectives."""

    center: np.ndarray
    scale: np.ndarray
    standardized_penalty: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        penalty = np.asarray(self.standardized_penalty, dtype=np.float64)
        expected = (len(OBJECTIVE_NAMES),)
        if center.shape != expected or scale.shape != expected or penalty.shape != expected:
            raise ValueError("objective scaler arrays must have shape (4,)")
        if not (
            np.isfinite(center).all()
            and np.isfinite(scale).all()
            and np.isfinite(penalty).all()
        ):
            raise ValueError("objective scaler values must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("objective scales must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "standardized_penalty", penalty)

    def transform(
        self,
        values: np.ndarray,
        *,
        is_penalty: np.ndarray | None = None,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        transformed = (array - self.center) / self.scale
        if is_penalty is not None:
            mask = np.asarray(is_penalty, dtype=bool)
            if mask.shape != (len(array),):
                raise ValueError("penalty mask has the wrong shape")
            transformed[mask] = self.standardized_penalty
        return transformed

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.center

    def std_to_physical(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "median_and_normalized_iqr_or_mad",
            "objective_names": list(OBJECTIVE_NAMES),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "standardized_penalty": self.standardized_penalty.tolist(),
        }


@dataclass
class ProposalResult:
    """Validated proposal batch returned to the campaign controller."""

    unit_values: np.ndarray
    raw_values: np.ndarray
    predicted_mean: np.ndarray
    predicted_std: np.ndarray
    predicted_mean_standardized: np.ndarray
    predicted_std_standardized: np.ndarray
    diagnostics: dict[str, Any]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_python_source(path: str | Path) -> str:
    """Hash source independently of checkout line endings."""

    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _wire_input_space(value: InputSpaceLike | Mapping[str, Any]) -> WireInputSpace:
    if isinstance(value, Mapping):
        names = value.get("parameter_names", value.get("names"))
        lower = value.get("lower")
        upper = value.get("upper")
    else:
        names = value.names
        lower = value.lower
        upper = value.upper
    if not isinstance(names, (list, tuple)) or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("input-space parameter_names must be a string list")
    return WireInputSpace(tuple(names), np.asarray(lower), np.asarray(upper))


def input_space_from_payload(payload: Mapping[str, Any]) -> WireInputSpace:
    return _wire_input_space(payload)


def _exact_area_contract(input_space: WireInputSpace) -> dict[str, Any]:
    required = (
        "SLOT_MAIN_LENGTH",
        "PATCH_BRICK_1_SIDE_MARGIN",
        "PATCH_BRICK_1_TOP_MARGIN",
        "PATCH_BRICK_2_HEIGHT_MARGIN",
    )
    missing = [name for name in required if name not in input_space.names]
    if missing:
        raise ValueError(f"input space lacks exact-area parameters: {missing}")
    return {
        "type": EXACT_AREA_CONTRACT_TYPE,
        "objective_index": 2,
        "reference_area_mm2": REFERENCE_SUBSTRATE_AREA_MM2,
        "width": {
            "slot_length_parameter": required[0],
            "side_margin_parameter": required[1],
            "formula": "slot_length + 2 * (1 + side_margin)",
        },
        "height": {
            "top_margin_parameter": required[2],
            "brick_2_height_parameter": required[3],
            "fixed_addition_mm": 23.0,
            "formula": "top_margin + brick_2_height + 23",
        },
    }


def exact_normalized_area(
    unit_values: np.ndarray,
    input_space: WireInputSpace,
    contract: Mapping[str, Any],
) -> np.ndarray:
    """Evaluate the explicit wire-level area contract for unit-cube rows."""

    if contract.get("type") != EXACT_AREA_CONTRACT_TYPE:
        raise ValueError("unsupported exact objective contract")
    if int(contract.get("objective_index", -1)) != 2:
        raise ValueError("exact area must occupy objective index 2")
    reference_area = float(contract.get("reference_area_mm2", math.nan))
    if not math.isfinite(reference_area) or reference_area <= 0.0:
        raise ValueError("exact area reference must be finite and positive")
    x = np.asarray(unit_values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(input_space.names):
        raise ValueError("unit values do not match the input space")
    raw = input_space.denormalize(x)
    index = {name: position for position, name in enumerate(input_space.names)}
    width_contract = _mapping(contract.get("width"), "exact_objective.width")
    height_contract = _mapping(contract.get("height"), "exact_objective.height")
    slot_index = index[str(width_contract.get("slot_length_parameter"))]
    side_index = index[str(width_contract.get("side_margin_parameter"))]
    top_index = index[str(height_contract.get("top_margin_parameter"))]
    brick_index = index[str(height_contract.get("brick_2_height_parameter"))]
    fixed_height = float(height_contract.get("fixed_addition_mm", math.nan))
    if not math.isfinite(fixed_height):
        raise ValueError("exact area fixed height must be finite")
    width = raw[:, slot_index] + 2.0 * (1.0 + raw[:, side_index])
    height = raw[:, top_index] + raw[:, brick_index] + fixed_height
    area = width * height / reference_area
    if not np.isfinite(area).all() or np.any(area <= 0.0):
        raise ValueError("exact normalized area must be finite and positive")
    return area[:, None]


def fit_objective_scaler(
    y_full_minimize: np.ndarray,
    is_penalty: np.ndarray,
) -> ObjectiveScaler:
    """Fit a robust scaler from successful rows and place penalties beyond them."""

    values = np.asarray(y_full_minimize, dtype=np.float64)
    penalty = np.asarray(is_penalty, dtype=bool)
    if values.ndim != 2 or values.shape[1] != len(OBJECTIVE_NAMES):
        raise ValueError("full objective array must have shape (n, 4)")
    if penalty.shape != (len(values),):
        raise ValueError("penalty mask has the wrong shape")
    if not np.isfinite(values).all():
        raise ValueError("objective values must be finite")
    valid = values[~penalty]
    if len(valid) == 0:
        raise ValueError("at least one non-penalty observation is required")

    center = np.median(valid, axis=0)
    q25, q75 = np.percentile(valid, [25.0, 75.0], axis=0)
    normalized_iqr = (q75 - q25) / 1.3489795003921634
    normalized_mad = (
        np.median(np.abs(valid - center), axis=0) * 1.482602218505602
    )
    scale = np.maximum(normalized_iqr, normalized_mad)
    standard_deviation = np.std(valid, axis=0)
    scale = np.where(scale > 1e-12, scale, standard_deviation)
    scale = np.where(scale > 1e-12, scale, 1.0)
    valid_standardized = (valid - center) / scale
    # The explicit replacement makes an invalid point dominated in every
    # standardized dimension, independent of its placeholder physical values.
    standardized_penalty = np.maximum(
        np.max(valid_standardized, axis=0) + 3.0,
        np.full(len(OBJECTIVE_NAMES), 3.0, dtype=np.float64),
    )
    return ObjectiveScaler(center, scale, standardized_penalty)


def _validate_training_arrays(
    x_unit: np.ndarray,
    y_expensive: np.ndarray,
    y_full: np.ndarray,
    is_penalty: np.ndarray,
    input_space: WireInputSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_unit, dtype=np.float64)
    expensive = np.asarray(y_expensive, dtype=np.float64)
    full = np.asarray(y_full, dtype=np.float64)
    penalty = np.asarray(is_penalty, dtype=bool)
    if x.ndim != 2 or x.shape[1] != len(input_space.names):
        raise ValueError("x_unit shape does not match the input space")
    if expensive.shape != (len(x), len(EXPENSIVE_OBJECTIVE_INDICES)):
        raise ValueError("y_expensive_minimize must have shape (n, 3)")
    if full.shape != (len(x), len(OBJECTIVE_NAMES)):
        raise ValueError("y_full_minimize must have shape (n, 4)")
    if penalty.shape != (len(x),):
        raise ValueError("is_penalty must have shape (n,)")
    if not (np.isfinite(x).all() and np.isfinite(expensive).all() and np.isfinite(full).all()):
        raise ValueError("training arrays must contain only finite values")
    if np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
        raise ValueError("normalized training values must lie inside [0, 1]")
    x = np.clip(x, 0.0, 1.0)
    if len(x) < 2 or len(np.unique(np.round(x, decimals=14), axis=0)) < 2:
        raise ValueError("at least two distinct observations are required")
    expected_expensive = full[:, EXPENSIVE_OBJECTIVE_INDICES]
    if not np.allclose(expensive, expected_expensive, rtol=1e-10, atol=1e-12):
        raise ValueError("expensive objectives disagree with full objective columns 0,1,3")
    return x, expensive, full, penalty


def build_request_payload(
    x_unit: np.ndarray,
    y_expensive_minimize: np.ndarray,
    y_full_minimize: np.ndarray,
    is_penalty: np.ndarray,
    input_space: InputSpaceLike | Mapping[str, Any],
    *,
    config: krvea.KRVEAConfig,
    iteration: int,
    remaining_expensive_budget: int,
    previous_empty_reference_count: int | None = None,
    compute_device: str = "cuda",
    surrogate_settings: SurrogateFitSettings = SurrogateFitSettings(),
) -> dict[str, Any]:
    """Build the compact, standalone proposal-worker request."""

    space = _wire_input_space(input_space)
    x, expensive, full, penalty = _validate_training_arrays(
        x_unit,
        y_expensive_minimize,
        y_full_minimize,
        is_penalty,
        space,
    )
    if config.n_variables != len(space.names) or config.n_objectives != 4:
        raise ValueError("K-RVEA config must match the 11-D/4-objective wire space")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if remaining_expensive_budget < 0:
        raise ValueError("remaining_expensive_budget must be non-negative")
    if not str(compute_device).strip():
        raise ValueError("compute_device must be non-empty")

    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "algorithm": "K-RVEA",
        "implementation": {
            "krvea_source_sha256": sha256_python_source(krvea.__file__),
            "krvea_relay_source_sha256": sha256_python_source(__file__),
        },
        "iteration": int(iteration),
        "remaining_expensive_budget": int(remaining_expensive_budget),
        "previous_empty_reference_count": previous_empty_reference_count,
        "compute": {"device": str(compute_device), "dtype": "float64"},
        "input_space": space.to_dict(),
        "objective_contract": {
            "semantics": "all_minimize",
            "names": list(OBJECTIVE_NAMES),
            "expensive_indices": list(EXPENSIVE_OBJECTIVE_INDICES),
            "exact_indices": list(EXACT_OBJECTIVE_INDICES),
            "cap_gain_note": "linear-power angle/frequency average converted to dBi",
            "exact_objective": _exact_area_contract(space),
        },
        "krvea_config": asdict(config),
        "surrogate_settings": asdict(surrogate_settings),
        "training": {
            "x_unit": x.tolist(),
            "y_expensive_minimize": expensive.tolist(),
            "y_full_minimize": full.tolist(),
            "is_penalty": penalty.tolist(),
            "summary": {
                "training_observations": int(len(x)),
                "penalty_observations": int(np.sum(penalty)),
                "successful_observations": int(np.sum(~penalty)),
            },
        },
    }


def _validate_objective_contract(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("semantics") != "all_minimize":
        raise ValueError("K-RVEA objectives must use all-minimize semantics")
    if tuple(payload.get("names", ())) != OBJECTIVE_NAMES:
        raise ValueError("K-RVEA objective order does not match this worker")
    if tuple(payload.get("expensive_indices", ())) != EXPENSIVE_OBJECTIVE_INDICES:
        raise ValueError("K-RVEA expensive objective indices do not match")
    if tuple(payload.get("exact_indices", ())) != EXACT_OBJECTIVE_INDICES:
        raise ValueError("K-RVEA exact objective indices do not match")
    return _mapping(payload.get("exact_objective"), "objective_contract.exact_objective")


def _botorch_imports() -> dict[str, Any]:
    """Import GPU-only dependencies lazily so relay tests stay lightweight."""

    try:
        import torch
        from botorch.exceptions.warnings import InputDataWarning
        from botorch.fit import fit_gpytorch_mll_torch
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:  # pragma: no cover - exercised on deployment hosts
        raise RuntimeError(
            "K-RVEA proposal fitting requires torch, botorch, and gpytorch"
        ) from exc
    return {
        "torch": torch,
        "InputDataWarning": InputDataWarning,
        "fit_gpytorch_mll_torch": fit_gpytorch_mll_torch,
        "SingleTaskGP": SingleTaskGP,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
    }


def _fit_surrogate_predictor(
    train_x: np.ndarray,
    train_y_standardized: np.ndarray,
    *,
    settings: SurrogateFitSettings,
    device_name: str,
    seed: int,
) -> tuple[krvea.SurrogatePredictor, dict[str, Any]]:
    runtime = _botorch_imports()
    torch = runtime["torch"]
    torch.set_default_dtype(torch.float64)
    requested = str(device_name).strip().lower()
    if not requested:
        raise ValueError("compute device must be non-empty")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA proposal requested but torch.cuda.is_available() is false")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {device.index}")
        torch.cuda.manual_seed_all(int(seed))
    torch.manual_seed(int(seed))

    x_tensor = torch.as_tensor(train_x, dtype=torch.float64, device=device)
    y_tensor = torch.as_tensor(
        train_y_standardized,
        dtype=torch.float64,
        device=device,
    )
    y_variance = torch.full_like(y_tensor, settings.gp_fixed_noise_variance)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Data .* is not standardized.*",
            category=runtime["InputDataWarning"],
        )
        model = runtime["SingleTaskGP"](
            x_tensor,
            y_tensor,
            train_Yvar=y_variance,
            outcome_transform=None,
        )
    mll = runtime["ExactMarginalLogLikelihood"](model.likelihood, model)
    fit_started = perf_counter()
    fit_result = runtime["fit_gpytorch_mll_torch"](
        mll,
        step_limit=settings.gp_training_steps,
        timeout_sec=settings.gp_timeout_seconds,
    )
    model.eval()
    fit_seconds = perf_counter() - fit_started

    def predictor(unit_values: np.ndarray) -> krvea.SurrogatePrediction:
        values = np.asarray(unit_values, dtype=np.float64)
        query = torch.as_tensor(values, dtype=torch.float64, device=device)
        with torch.no_grad():
            posterior = model.posterior(query, observation_noise=False)
            mean = posterior.mean.detach().cpu().numpy().astype(np.float64, copy=False)
            variance = posterior.variance.detach().cpu().numpy().astype(
                np.float64,
                copy=False,
            )
        return krvea.SurrogatePrediction(
            mean=mean,
            std=np.sqrt(np.maximum(variance, 0.0)),
        )

    diagnostics: dict[str, Any] = {
        "device": str(device),
        "dtype": "float64",
        "torch_version": str(torch.__version__),
        "gp_fit_seconds": float(fit_seconds),
        "gp_training_steps": int(settings.gp_training_steps),
        "gp_fixed_noise_variance": float(settings.gp_fixed_noise_variance),
        "gp_fit_result_type": type(fit_result).__name__,
        "gp_output_count": len(EXPENSIVE_OBJECTIVE_INDICES),
    }
    if device.type == "cuda":
        diagnostics.update(
            {
                "cuda_available": True,
                "cuda_device_name": str(torch.cuda.get_device_name(device)),
            }
        )
    return predictor, diagnostics


def run_request_payload(payload: Mapping[str, Any]) -> ProposalResult:
    """Validate, fit, evolve, and select one K-RVEA proposal batch."""

    if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported K-RVEA proposal request schema")
    if payload.get("algorithm") != "K-RVEA":
        raise ValueError("unsupported proposal algorithm")
    implementation = _mapping(payload.get("implementation"), "implementation")
    expected_implementation = {
        "krvea_source_sha256": sha256_python_source(krvea.__file__),
        "krvea_relay_source_sha256": sha256_python_source(__file__),
    }
    if dict(implementation) != expected_implementation:
        raise RuntimeError(
            "proposal request implementation fingerprints do not match this worker; "
            "pull the same Git commit on both hosts"
        )
    compute = _mapping(payload.get("compute"), "compute")
    if compute.get("dtype") != "float64":
        raise ValueError("K-RVEA GPU proposal requests must use float64")
    space = input_space_from_payload(_mapping(payload.get("input_space"), "input_space"))
    exact_contract = _validate_objective_contract(
        _mapping(payload.get("objective_contract"), "objective_contract")
    )
    config = krvea.KRVEAConfig(
        **dict(_mapping(payload.get("krvea_config"), "krvea_config"))
    )
    surrogate_settings = SurrogateFitSettings(
        **dict(_mapping(payload.get("surrogate_settings"), "surrogate_settings"))
    )
    training = _mapping(payload.get("training"), "training")
    x, expensive, full, penalty = _validate_training_arrays(
        np.asarray(training.get("x_unit"), dtype=np.float64),
        np.asarray(training.get("y_expensive_minimize"), dtype=np.float64),
        np.asarray(training.get("y_full_minimize"), dtype=np.float64),
        np.asarray(training.get("is_penalty"), dtype=bool),
        space,
    )
    scaler = fit_objective_scaler(full, penalty)
    full_standardized = scaler.transform(full, is_penalty=penalty)
    expensive_standardized = full_standardized[:, EXPENSIVE_OBJECTIVE_INDICES]

    predictor, fit_diagnostics = _fit_surrogate_predictor(
        x,
        expensive_standardized,
        settings=surrogate_settings,
        device_name=str(compute.get("device", "")),
        seed=config.seed,
    )

    def standardized_exact_area(unit_values: np.ndarray) -> np.ndarray:
        physical = exact_normalized_area(unit_values, space, exact_contract)
        return (physical - scaler.center[2]) / scaler.scale[2]

    engine = krvea.KRVEA(
        config,
        predictor,
        expensive_objective_indices=EXPENSIVE_OBJECTIVE_INDICES,
        exact_objective=standardized_exact_area,
        exact_objective_indices=EXACT_OBJECTIVE_INDICES,
    )
    batch = engine.propose(
        x,
        full_standardized,
        remaining_expensive_budget=int(payload.get("remaining_expensive_budget", 0)),
        previous_empty_reference_count=payload.get("previous_empty_reference_count"),
    )
    raw_values = space.denormalize(batch.unit_x)
    physical_mean = scaler.inverse(batch.predicted_mean)
    physical_std = scaler.std_to_physical(batch.predicted_std)
    physical_std[:, 2] = 0.0
    core_diagnostics = asdict(batch.diagnostics)
    diagnostics = {
        # These fields are intentionally also flat: the controller consumes
        # ``mode`` and the empty-reference count as continuation state.
        **core_diagnostics,
        "iteration": int(payload.get("iteration", 0)),
        "objective_scaler": scaler.to_dict(),
        "objective_contract": {
            "semantics": "all_minimize",
            "names": list(OBJECTIVE_NAMES),
            "expensive_indices": list(EXPENSIVE_OBJECTIVE_INDICES),
            "exact_indices": list(EXACT_OBJECTIVE_INDICES),
            "exact_objective": dict(exact_contract),
        },
        "surrogate": fit_diagnostics,
        "krvea": core_diagnostics,
        "training": dict(_mapping(training.get("summary"), "training.summary")),
    }
    return ProposalResult(
        unit_values=batch.unit_x,
        raw_values=raw_values,
        predicted_mean=physical_mean,
        predicted_std=physical_std,
        predicted_mean_standardized=batch.predicted_mean,
        predicted_std_standardized=batch.predicted_std,
        diagnostics=diagnostics,
    )


def response_payload(request_sha256: str, result: ProposalResult) -> dict[str, Any]:
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "status": "completed",
        "request_sha256": str(request_sha256),
        "result": {
            "unit_values": result.unit_values.tolist(),
            "raw_values": result.raw_values.tolist(),
            "predicted_mean_minimize": result.predicted_mean.tolist(),
            "predicted_std": result.predicted_std.tolist(),
            "predicted_mean_standardized": result.predicted_mean_standardized.tolist(),
            "predicted_std_standardized": result.predicted_std_standardized.tolist(),
            "diagnostics": result.diagnostics,
        },
    }


def _duplicates_within(values: np.ndarray, tolerance: float) -> bool:
    for index in range(len(values)):
        if index and np.any(
            np.max(np.abs(values[:index] - values[index]), axis=1) <= tolerance
        ):
            return True
    return False


def result_from_response(
    payload: Mapping[str, Any],
    *,
    expected_request_sha256: str,
    expected_q: int,
    expected_dimension: int,
    observed_x_unit: np.ndarray | None = None,
    input_space: InputSpaceLike | Mapping[str, Any] | None = None,
    duplicate_tolerance: float = RESPONSE_DUPLICATE_TOLERANCE,
) -> ProposalResult:
    """Validate response ownership, shapes, bounds, and candidate novelty."""

    if payload.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise ValueError("unsupported K-RVEA proposal response schema")
    if payload.get("status") != "completed":
        raise ValueError("K-RVEA proposal response is not completed")
    if payload.get("request_sha256") != expected_request_sha256:
        raise ValueError("proposal response belongs to a different request")
    if expected_q < 0 or expected_dimension <= 0:
        raise ValueError("invalid expected proposal shape")
    result = _mapping(payload.get("result"), "result")
    unit_values = np.asarray(result.get("unit_values"), dtype=np.float64)
    raw_values = np.asarray(result.get("raw_values"), dtype=np.float64)
    mean = np.asarray(result.get("predicted_mean_minimize"), dtype=np.float64)
    std = np.asarray(result.get("predicted_std"), dtype=np.float64)
    mean_standardized = np.asarray(
        result.get("predicted_mean_standardized"), dtype=np.float64
    )
    std_standardized = np.asarray(
        result.get("predicted_std_standardized"), dtype=np.float64
    )
    candidate_shape = (expected_q, expected_dimension)
    objective_shape = (expected_q, len(OBJECTIVE_NAMES))
    if unit_values.shape != candidate_shape or raw_values.shape != candidate_shape:
        raise ValueError(
            f"proposal response candidate shape must be {candidate_shape}, got "
            f"{unit_values.shape} and {raw_values.shape}"
        )
    for label, values in (
        ("predicted_mean_minimize", mean),
        ("predicted_std", std),
        ("predicted_mean_standardized", mean_standardized),
        ("predicted_std_standardized", std_standardized),
    ):
        if values.shape != objective_shape:
            raise ValueError(f"{label} shape must be {objective_shape}")
    arrays = (unit_values, raw_values, mean, std, mean_standardized, std_standardized)
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("proposal response contains non-finite values")
    if np.any(unit_values < -1e-12) or np.any(unit_values > 1.0 + 1e-12):
        raise ValueError("proposal response contains normalized values outside [0, 1]")
    if np.any(std < -1e-12) or np.any(std_standardized < -1e-12):
        raise ValueError("proposal response contains negative predictive uncertainty")
    if not np.allclose(std[:, 2], 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("exact area uncertainty must be zero")
    if not np.allclose(std_standardized[:, 2], 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("standardized exact area uncertainty must be zero")
    if _duplicates_within(unit_values, duplicate_tolerance):
        raise ValueError("proposal response contains duplicate candidates")
    if observed_x_unit is not None and len(unit_values):
        observed = np.asarray(observed_x_unit, dtype=np.float64)
        if observed.ndim != 2 or observed.shape[1] != expected_dimension:
            raise ValueError("observed_x_unit has the wrong shape")
        for candidate in unit_values:
            if len(observed) and np.any(
                np.max(np.abs(observed - candidate), axis=1) <= duplicate_tolerance
            ):
                raise ValueError("proposal response repeats an observed design")
    if input_space is not None:
        space = _wire_input_space(input_space)
        expected_raw = space.denormalize(unit_values)
        if not np.allclose(raw_values, expected_raw, rtol=1e-10, atol=1e-10):
            raise ValueError("proposal raw values do not match normalized candidates")
    diagnostics = dict(_mapping(result.get("diagnostics"), "result.diagnostics"))
    return ProposalResult(
        unit_values=np.clip(unit_values, 0.0, 1.0),
        raw_values=raw_values,
        predicted_mean=mean,
        predicted_std=np.maximum(std, 0.0),
        predicted_mean_standardized=mean_standardized,
        predicted_std_standardized=np.maximum(std_standardized, 0.0),
        diagnostics=diagnostics,
    )


def execute_request_file(request_path: str | Path, response_path: str | Path) -> bool:
    """Run or reuse one hash-addressed request; return True when reused."""

    request = Path(request_path).expanduser().resolve()
    response = Path(response_path).expanduser().resolve()
    request_sha = sha256_file(request)
    if response.is_file():
        existing = json.loads(response.read_text(encoding="utf-8-sig"))
        if existing.get("request_sha256") != request_sha:
            raise RuntimeError("existing proposal response has a different request hash")
        if existing.get("status") != "completed":
            raise RuntimeError("existing proposal response is not completed")
        return True
    payload = json.loads(request.read_text(encoding="utf-8-sig"))
    result = run_request_payload(_mapping(payload, "request"))
    _atomic_write_json(response, response_payload(request_sha, result))
    return False


def _ps_literal(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if any(character in text for character in ("\0", "\r", "\n")):
        raise ValueError("PowerShell literal contains a control character")
    return "'" + text.replace("'", "''") + "'"


def relay_remote_proposal(
    *,
    device: DeviceConfig,
    remote: RemoteProposalConfig,
    plan_id: str,
    batch_index: int,
    local_request_path: Path,
    local_response_path: Path,
    expected_q: int,
    expected_dimension: int,
    observed_x_unit: np.ndarray | None = None,
    input_space: InputSpaceLike | Mapping[str, Any] | None = None,
) -> ProposalResult:
    """Upload, execute, and retrieve one idempotent CoconutG2 proposal."""

    if device.id != remote.device_id:
        raise ValueError("selected GPU device does not match RemoteProposalConfig")
    if not device.is_remote:
        raise ValueError("remote GPU proposal device must be SSH-addressable")
    request_sha = sha256_file(local_request_path)
    stem = f"batch_{batch_index:04d}_{request_sha[:16]}"
    remote_root = PureWindowsPath(device.repo_root) / DEFAULT_REMOTE_WORK_ROOT / plan_id
    remote_request = remote_root / f"{stem}.request.json"
    remote_response = remote_root / f"{stem}.response.json"
    worker_path = (
        PureWindowsPath(device.repo_root)
        / "scripts"
        / "optimization"
        / "krvea_gpu_worker.py"
    )
    push_file_atomic(device, local_request_path, str(remote_request), overwrite=True)
    script = "\n".join(
        (
            "$ErrorActionPreference = 'Stop'",
            f"$python = {_ps_literal(remote.python_path)}",
            f"$worker = {_ps_literal(str(worker_path))}",
            "if (-not (Test-Path -LiteralPath $python -PathType Leaf)) "
            "{ throw 'bocuda Python executable does not exist' }",
            "if (-not (Test-Path -LiteralPath $worker -PathType Leaf)) "
            "{ throw 'K-RVEA GPU worker does not exist; pull the repository first' }",
            f"& $python $worker --request {_ps_literal(str(remote_request))} "
            f"--response {_ps_literal(str(remote_response))}",
            "if ($LASTEXITCODE -ne 0) "
            "{ throw ('K-RVEA GPU worker exited with code ' + $LASTEXITCODE) }",
        )
    )
    completed = run_remote_powershell(
        device,
        script,
        timeout=remote.timeout_seconds,
        action="run K-RVEA GPU proposal",
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    pull_file_atomic(device, str(remote_response), local_response_path, overwrite=True)
    payload = json.loads(local_response_path.read_text(encoding="utf-8-sig"))
    result = result_from_response(
        _mapping(payload, "response"),
        expected_request_sha256=request_sha,
        expected_q=expected_q,
        expected_dimension=expected_dimension,
        observed_x_unit=observed_x_unit,
        input_space=input_space,
    )
    result.diagnostics.update(
        {
            "proposal_executor": "remote_ssh",
            "proposal_device_id": device.id,
            "proposal_request_sha256": request_sha,
            "proposal_remote_request": str(remote_request),
            "proposal_remote_response": str(remote_response),
        }
    )
    return result


def write_request(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    _atomic_write_json(destination, payload)
    return destination


__all__ = [
    "DEFAULT_REMOTE_WORK_ROOT",
    "EXACT_OBJECTIVE_INDICES",
    "EXPENSIVE_OBJECTIVE_INDICES",
    "OBJECTIVE_NAMES",
    "ObjectiveScaler",
    "ProposalResult",
    "REFERENCE_SUBSTRATE_AREA_MM2",
    "RemoteProposalConfig",
    "SurrogateFitSettings",
    "WireInputSpace",
    "build_request_payload",
    "exact_normalized_area",
    "execute_request_file",
    "fit_objective_scaler",
    "input_space_from_payload",
    "relay_remote_proposal",
    "response_payload",
    "result_from_response",
    "run_request_payload",
    "sha256_file",
    "sha256_python_source",
    "write_request",
]
