"""Separate arbitrary-tree K-RVEA wire protocol, using the Phase-2 engine.

Only S11 and the frozen RoI gain are modelled.  Substrate area is calculated
from all seven millimetre parameters and the *CST-quantized* outer rectangle.
The legacy 11-variable protocol, worker and module globals remain untouched.
No CST, torch or CUDA dependency is imported until an explicit fit is requested.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path, PureWindowsPath
from types import FunctionType
from typing import Any, Mapping

import numpy as np

from msabp_opt.simulation.distributed.config import DeviceConfig
from msabp_opt.simulation.distributed.transport import (
    pull_file_atomic,
    push_file_atomic,
    run_remote_powershell,
)

from . import krvea, phase2_krvea_relay as legacy


REQUEST_SCHEMA_VERSION = "msabp-tree-krvea-request-v1"
RESPONSE_SCHEMA_VERSION = "msabp-tree-krvea-response-v1"
EXACT_AREA_CONTRACT_TYPE = "msabp_tree_quantized_substrate_area_v1"
DEFAULT_REMOTE_WORK_ROOT = PureWindowsPath("simulations", "runs", "krvea_tree_gpu")
WORKER_RELATIVE_PATH = Path("scripts/optimization/phase2_krvea_tree_gpu_worker.py")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ABSOLUTE_PARAMETERS = (
    "SLOT_MAIN_LENGTH", "SLOT_MAIN_HEIGHT", "PATCH_BRICK_1_SIDE_MARGIN",
    "PATCH_BRICK_1_TOP_MARGIN", "PATCH_BRICK_2_HEIGHT_MARGIN",
    "PATCH_BRICK_3_BOTTOM_MARGIN", "PATCH_BRICK_4_MARGIN",
)

WireInputSpace = legacy.WireInputSpace
InputSpaceLike = legacy.InputSpaceLike
SurrogateFitSettings = legacy.SurrogateFitSettings
RemoteProposalConfig = legacy.RemoteProposalConfig
ProposalResult = legacy.ProposalResult
OBJECTIVE_NAMES = legacy.OBJECTIVE_NAMES
EXPENSIVE_OBJECTIVE_INDICES = legacy.EXPENSIVE_OBJECTIVE_INDICES
EXACT_OBJECTIVE_INDICES = legacy.EXACT_OBJECTIVE_INDICES
input_space_from_payload = legacy.input_space_from_payload
sha256_file = legacy.sha256_file
sha256_python_source = legacy.sha256_python_source
write_request = legacy.write_request
_fit_surrogate_predictor = legacy._fit_surrogate_predictor


def _engine_function(name: str, **overrides: Any) -> FunctionType:
    """Bind proven engine bytecode to a private, per-call global namespace.

    This avoids copying the numerical implementation or monkey-patching shared
    legacy globals. Helpers that do not depend on the new protocol stay shared;
    the entry functions receive only the schema and exact-area replacements.
    """

    function = getattr(legacy, name)
    namespace = dict(vars(legacy))
    namespace.update(
        REQUEST_SCHEMA_VERSION=REQUEST_SCHEMA_VERSION,
        RESPONSE_SCHEMA_VERSION=RESPONSE_SCHEMA_VERSION,
        exact_normalized_area=exact_normalized_area,
        _fit_surrogate_predictor=_fit_surrogate_predictor,
    )
    namespace.update(overrides)
    bound = FunctionType(
        function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def _implementation() -> dict[str, str]:
    return {
        "krvea_source_sha256": sha256_python_source(krvea.__file__),
        "krvea_relay_source_sha256": sha256_python_source(legacy.__file__),
        "tree_relay_source_sha256": sha256_python_source(__file__),
        "tree_worker_source_sha256": sha256_python_source(REPOSITORY_ROOT / WORKER_RELATIVE_PATH),
    }


def exact_area_contract(
    input_space: InputSpaceLike | Mapping[str, Any], *, plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the rectangle construction and millimetre grid, not a legacy area formula."""

    space = legacy._wire_input_space(input_space)
    missing = set(ABSOLUTE_PARAMETERS).difference(space.names)
    if missing:
        raise ValueError(f"tree input space lacks exact-area parameters: {sorted(missing)}")
    variables = plan.get("variables")
    if variables is not None:
        if tuple(row["name"] for row in variables) != space.names:
            raise ValueError("input-space names/order differ from the frozen tree plan")
        if not (
            np.array_equal(space.lower, [row["lower"] for row in variables])
            and np.array_equal(space.upper, [row["upper"] for row in variables])
        ):
            raise ValueError("input-space bounds differ from the frozen tree plan")
    quantum = float(plan["config"]["coordinate_quantum_mm"])
    reference = float(plan["metrics"]["reference_area_mm2"])
    if not math.isfinite(quantum) or quantum <= 0:
        raise ValueError("coordinate quantum must be finite and positive")
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("reference area must be finite and positive")
    return {
        "type": EXACT_AREA_CONTRACT_TYPE, "objective_index": 2,
        **space.to_dict(), "coordinate_quantum_mm": quantum,
        "reference_area_mm2": reference, "fixed_offset_mm": 1.0,
        "brick2_width_mm": 12.0, "brick4_fixed_mm": 13.0,
    }


def exact_normalized_area(
    unit_values: np.ndarray, input_space: WireInputSpace, contract: Mapping[str, Any],
) -> np.ndarray:
    """Return the exact exported rectangle area, with zero statistical uncertainty.

    The builder rounds +/-x separately and shifts the lower y edge to zero
    before rounding. Preserve its arithmetic grouping at half-grid ties.
    """

    if contract.get("type") != EXACT_AREA_CONTRACT_TYPE:
        raise ValueError("unsupported tree exact objective contract")
    if contract.get("objective_index", 2) != 2:
        raise ValueError("exact area must occupy objective index 2")
    for key, expected in (("fixed_offset_mm", 1.0), ("brick2_width_mm", 12.0),
                          ("brick4_fixed_mm", 13.0)):
        if contract.get(key) != expected:
            raise ValueError(f"unsupported tree construction constant: {key}")
    contract_space = input_space_from_payload(contract)
    if (contract_space.names != input_space.names
            or not np.array_equal(contract_space.lower, input_space.lower)
            or not np.array_equal(contract_space.upper, input_space.upper)):
        raise ValueError("exact-area bounds/order do not match the input space")
    x = np.asarray(unit_values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(input_space.names) or not np.isfinite(x).all():
        raise ValueError("unit values do not match the input space")
    if np.any(x < -1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError("unit values must lie inside [0, 1]")
    x = np.clip(x, 0.0, 1.0)
    quantum = float(contract["coordinate_quantum_mm"])
    reference = float(contract["reference_area_mm2"])
    if not math.isfinite(quantum) or quantum <= 0 or not math.isfinite(reference) or reference <= 0:
        raise ValueError("exact-area quantum and reference must be finite and positive")
    raw = input_space.denormalize(x)
    index = {name: position for position, name in enumerate(input_space.names)}
    if set(ABSOLUTE_PARAMETERS).difference(index):
        raise ValueError("all seven absolute tree parameters are required")
    slot_length, slot_height, side, top, height2, bottom, margin4 = (
        raw[:, index[name]] for name in ABSOLUTE_PARAMETERS
    )
    half_width = np.maximum(slot_length / 2.0 + (1.0 + side), 6.0)
    y_high = slot_height / 2.0 + ((1.0 + top) + height2)
    y_low = -slot_height / 2.0 - ((1.0 + bottom) + margin4 + 13.0)
    # np.rint and Python round both use nearest-even rounding on these doubles.
    x_high = np.rint(half_width / quantum) * quantum
    x_low = np.rint(-half_width / quantum) * quantum
    y_extent = np.rint((y_high - y_low) / quantum) * quantum
    area = (x_high - x_low) * y_extent / reference
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("exact normalized area must be finite and positive")
    return area[:, None]


def build_request_payload(
    x_unit: np.ndarray, y_expensive_minimize: np.ndarray,
    y_full_minimize: np.ndarray, is_penalty: np.ndarray,
    input_space: InputSpaceLike | Mapping[str, Any], *, plan: Mapping[str, Any],
    config: krvea.KRVEAConfig, iteration: int, remaining_expensive_budget: int,
    previous_empty_reference_count: int | None = None, compute_device: str = "cuda",
    surrogate_settings: SurrogateFitSettings = SurrogateFitSettings(),
) -> dict[str, Any]:
    """Build a standalone tree request; old workers intentionally reject it."""

    contract = exact_area_contract(input_space, plan=plan)
    payload = _engine_function(
        "build_request_payload", _exact_area_contract=lambda space: contract,
    )(
        x_unit, y_expensive_minimize, y_full_minimize, is_penalty, input_space,
        config=config, iteration=iteration,
        remaining_expensive_budget=remaining_expensive_budget,
        previous_empty_reference_count=previous_empty_reference_count,
        compute_device=compute_device, surrogate_settings=surrogate_settings,
    )
    payload["implementation"] = _implementation()
    payload["tree_provenance"] = {
        "plan_sha256": str(plan.get("plan_sha256", "")),
        "template_sha256": str(plan.get("template_sha256", "")),
        "source_sha256": copy.deepcopy(plan.get("source_sha256", {})),
        "dimension": len(contract["parameter_names"]),
    }
    validate_request_payload(payload)
    return payload


def build_validation_request_payload(
    *args: Any, validation_x_unit: np.ndarray, **kwargs: Any,
) -> dict[str, Any]:
    """Use the same frozen training payload, sending held-out X but never Y."""

    payload = build_request_payload(*args, **kwargs)
    payload["task"] = "validate"
    payload["validation"] = {
        "x_unit": np.asarray(validation_x_unit, dtype=np.float64).tolist(),
    }
    validate_request_payload(payload)
    return payload


def validate_request_payload(payload: Mapping[str, Any]) -> None:
    """Validate source/metric contracts and numeric arrays without fitting."""

    if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported tree K-RVEA proposal request schema")
    if payload.get("algorithm") != "K-RVEA":
        raise ValueError("unsupported tree proposal algorithm")
    if payload.get("task", "propose") not in ("propose", "validate"):
        raise ValueError("unsupported tree worker task")
    if payload.get("implementation") != _implementation():
        raise RuntimeError("tree proposal source fingerprints differ; pull the same Git commit")
    if payload.get("compute", {}).get("dtype") != "float64":
        raise ValueError("tree K-RVEA requests must use float64")
    device_name = payload["compute"].get("device")
    if not isinstance(device_name, str) or not device_name.strip():
        raise ValueError("compute device must be a non-empty string")
    for name in ("iteration", "remaining_expensive_budget"):
        value = payload.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    space = input_space_from_payload(payload["input_space"])
    provenance = legacy._mapping(payload.get("tree_provenance"), "tree_provenance")
    if (type(provenance.get("dimension")) is not int
            or provenance["dimension"] != len(space.names)):
        raise ValueError("tree provenance dimension differs from the input space")
    sources = legacy._mapping(provenance.get("source_sha256"), "tree source_sha256")
    if not sources:
        raise ValueError("tree source fingerprints must not be empty")
    digests = [provenance.get("plan_sha256"), provenance.get("template_sha256"), *sources.values()]
    if any(not isinstance(value, str) or len(value) != 64
           or any(char not in "0123456789abcdefABCDEF" for char in value) for value in digests):
        raise ValueError("tree provenance requires SHA-256 fingerprints")
    contract = legacy._validate_objective_contract(payload["objective_contract"])
    settings = krvea.KRVEAConfig(**payload["krvea_config"])
    SurrogateFitSettings(**payload["surrogate_settings"])
    if settings.n_variables != len(space.names) or settings.n_objectives != 3:
        raise ValueError("K-RVEA dimensions must match the tree/3-objective input space")
    previous_empty = payload.get("previous_empty_reference_count")
    reference_count = math.comb(settings.reference_partitions + 2, 2)
    if previous_empty is not None and (
        type(previous_empty) is not int or not 0 <= previous_empty <= reference_count
    ):
        raise ValueError("previous empty-reference count is outside the valid range")
    training = payload["training"]
    x, expensive, full, penalty = legacy._validate_training_arrays(
        training["x_unit"], training["y_expensive_minimize"],
        training["y_full_minimize"], training["is_penalty"], space,
    )
    expected = exact_normalized_area(x, space, contract)[:, 0]
    if not np.allclose(full[~penalty, 2], expected[~penalty], rtol=1e-10, atol=1e-12):
        raise ValueError("training substrate area disagrees with the exact tree contract")
    if np.sum(~penalty) < 2:
        raise ValueError("at least two successful observations are required for the GPs")
    if len(np.unique(np.round(x[~penalty], decimals=14), axis=0)) < 2:
        raise ValueError("at least two distinct successful observations are required for the GPs")
    # This inexpensive transform check catches out-of-domain physical targets
    # even for --validate-only, before any torch import or model fitting.
    legacy.SurrogateTargetScaler.fit(expensive[~penalty])
    if payload.get("task") == "validate":
        validation = payload.get("validation", {})
        if set(validation) != {"x_unit"}:
            raise ValueError("validation request must contain held-out X only, never labels")
        validation_x = np.asarray(validation["x_unit"], dtype=np.float64)
        exact_normalized_area(validation_x, space, contract)
        if not len(validation_x):
            raise ValueError("validation requires at least one held-out design")
        if legacy._duplicates_within(validation_x, legacy.RESPONSE_DUPLICATE_TOLERANCE):
            raise ValueError("validation contains duplicate held-out designs")
        for candidate in validation_x:
            if np.any(np.max(np.abs(x - candidate), axis=1) <= legacy.RESPONSE_DUPLICATE_TOLERANCE):
                raise ValueError("validation design overlaps training inputs")


def run_request_payload(payload: Mapping[str, Any]) -> ProposalResult:
    validate_request_payload(payload)
    if payload.get("task", "propose") != "propose":
        raise ValueError("validation requests must not run K-RVEA evolution")
    engine_payload = copy.deepcopy(payload)
    engine_payload["implementation"] = {
        key: payload["implementation"][key]
        for key in ("krvea_source_sha256", "krvea_relay_source_sha256")
    }
    result = _engine_function("run_request_payload")(engine_payload)
    result.diagnostics["tree_provenance"] = copy.deepcopy(payload["tree_provenance"])
    result.diagnostics["implementation"] = _implementation()
    return result


def fit_predict_physical(
    payload: Mapping[str, Any], prediction_x_unit: np.ndarray,
) -> dict[str, Any]:
    """Fit only request training rows and predict held-out X (never its labels).

    S11 uses the mature bounded-logit moment transform, RoI identity, area exact.
    This is an explicit expensive operation; request construction does not fit.
    """

    validate_request_payload(payload)
    space = input_space_from_payload(payload["input_space"])
    contract = payload["objective_contract"]["exact_objective"]
    prediction_x = np.asarray(prediction_x_unit, dtype=np.float64)
    area = exact_normalized_area(prediction_x, space, contract)
    training = payload["training"]
    successful = ~np.asarray(training["is_penalty"], dtype=bool)
    x = np.asarray(training["x_unit"], dtype=np.float64)[successful]
    y = np.asarray(training["y_expensive_minimize"], dtype=np.float64)[successful]
    settings = SurrogateFitSettings(**payload["surrogate_settings"])
    scaler = legacy.SurrogateTargetScaler.fit(y)
    predictor, diagnostics = _fit_surrogate_predictor(
        x, scaler.transform(y), settings=settings,
        device_name=payload["compute"]["device"], seed=payload["krvea_config"]["seed"],
    )
    prediction = predictor(prediction_x)
    mean, std = scaler.inverse_prediction(
        prediction.mean, prediction.std,
        quadrature_order=settings.bounded_moment_quadrature_order,
    )
    return {
        "predicted_mean_minimize": np.column_stack((mean, area)).tolist(),
        "predicted_std": np.column_stack((std, np.zeros(len(area)))).tolist(),
        "diagnostics": {
            "surrogate": diagnostics, "target_scaler": scaler.to_dict(),
            "gp_training_observations": len(x), "validation_observations": len(area),
            "validation_labels_used_in_fit": False,
        },
    }


def response_payload(request_sha256: str, result: ProposalResult) -> dict[str, Any]:
    return _engine_function("response_payload")(request_sha256, result)


def result_from_response(
    payload: Mapping[str, Any], *, expected_request_sha256: str,
    expected_q: int, expected_dimension: int, observed_x_unit: np.ndarray | None = None,
    input_space: InputSpaceLike | Mapping[str, Any] | None = None,
    exact_contract: Mapping[str, Any] | None = None,
    duplicate_tolerance: float = legacy.RESPONSE_DUPLICATE_TOLERANCE,
) -> ProposalResult:
    # JSON cannot retain the second dimension of an empty (0, d) array.
    # Restore only canonical empty lists; malformed nonempty shapes must fail.
    if expected_q == 0:
        payload = copy.deepcopy(payload)
        result_payload = legacy._mapping(payload.get("result"), "result")
        for key, width in (
            ("unit_values", expected_dimension), ("raw_values", expected_dimension),
            ("predicted_mean_minimize", 3), ("predicted_std", 3),
            ("predicted_mean_standardized", 3), ("predicted_std_standardized", 3),
        ):
            values = result_payload.get(key)
            if isinstance(values, list) and not values:
                result_payload[key] = np.empty((0, width), dtype=np.float64)
    result = _engine_function("result_from_response")(
        payload, expected_request_sha256=expected_request_sha256,
        expected_q=expected_q, expected_dimension=expected_dimension,
        observed_x_unit=observed_x_unit, input_space=input_space,
        duplicate_tolerance=duplicate_tolerance,
    )
    if np.any(result.predicted_mean[:, 0] < 0) or np.any(result.predicted_mean[:, 0] > 1):
        raise ValueError("proposal S11 predictions must remain bounded")
    if exact_contract is not None:
        if input_space is None:
            raise ValueError("input_space is required for exact-area response validation")
        area = exact_normalized_area(
            result.unit_values, legacy._wire_input_space(input_space), exact_contract,
        )[:, 0]
        if not np.allclose(result.predicted_mean[:, 2], area, rtol=1e-10, atol=1e-12):
            raise ValueError("proposal area does not match its exact tree geometry")
    return result


def execute_request_file(request_path: str | Path, response_path: str | Path) -> bool:
    """Execute or validate/reuse one content-addressed response, no CST involved."""

    request_path, response_path = Path(request_path), Path(response_path)
    payload = json.loads(request_path.read_text(encoding="utf-8-sig"))
    validate_request_payload(legacy._mapping(payload, "request"))
    request_sha = sha256_file(request_path)
    if payload.get("task") == "validate":
        if response_path.is_file():
            validation_from_response(
                json.loads(response_path.read_text(encoding="utf-8-sig")),
                expected_request_sha256=request_sha, request=payload,
            )
            return True
        prediction = fit_predict_physical(payload, payload["validation"]["x_unit"])
        response = {
            "schema_version": RESPONSE_SCHEMA_VERSION, "status": "completed",
            "task": "validate", "request_sha256": request_sha,
            "validation_result": prediction,
        }
        validation_from_response(response, expected_request_sha256=request_sha, request=payload)
        legacy._atomic_write_json(response_path, response)
        return False
    q = min(payload["krvea_config"]["q"], payload["remaining_expensive_budget"])
    if response_path.is_file():
        result_from_response(
            json.loads(response_path.read_text(encoding="utf-8-sig")),
            expected_request_sha256=request_sha, expected_q=q,
            expected_dimension=len(payload["input_space"]["parameter_names"]),
            observed_x_unit=np.asarray(payload["training"]["x_unit"]),
            input_space=payload["input_space"],
            exact_contract=payload["objective_contract"]["exact_objective"],
        )
        return True
    result = run_request_payload(payload)
    response_payload_value = response_payload(request_sha, result)
    result_from_response(
        response_payload_value, expected_request_sha256=request_sha, expected_q=q,
        expected_dimension=len(payload["input_space"]["parameter_names"]),
        observed_x_unit=np.asarray(payload["training"]["x_unit"]),
        input_space=payload["input_space"],
        exact_contract=payload["objective_contract"]["exact_objective"],
    )
    legacy._atomic_write_json(response_path, response_payload_value)
    return False


def validation_from_response(
    payload: Mapping[str, Any], *, expected_request_sha256: str, request: Mapping[str, Any],
) -> dict[str, Any]:
    validate_request_payload(request)
    if request.get("task") != "validate":
        raise ValueError("validation response requires a validation request")
    if (payload.get("schema_version") != RESPONSE_SCHEMA_VERSION
            or payload.get("task") != "validate" or payload.get("status") != "completed"):
        raise ValueError("not a completed tree validation response")
    if payload.get("request_sha256") != expected_request_sha256:
        raise ValueError("validation response belongs to a different request")
    result = dict(legacy._mapping(payload.get("validation_result"), "validation_result"))
    mean = np.asarray(result.get("predicted_mean_minimize"), dtype=np.float64)
    std = np.asarray(result.get("predicted_std"), dtype=np.float64)
    shape = (len(request["validation"]["x_unit"]), 3)
    if mean.shape != shape or std.shape != shape:
        raise ValueError(f"validation prediction shape must be {shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("validation predictions must be finite with nonnegative uncertainty")
    if np.any(mean[:, 0] < 0) or np.any(mean[:, 0] > 1):
        raise ValueError("validation S11 predictions must remain bounded")
    area = exact_normalized_area(
        np.asarray(request["validation"]["x_unit"]), input_space_from_payload(request["input_space"]),
        request["objective_contract"]["exact_objective"],
    )[:, 0]
    if (not np.allclose(mean[:, 2], area, rtol=1e-10, atol=1e-12)
            or not np.allclose(std[:, 2], 0, rtol=0, atol=1e-12)):
        raise ValueError("validation area must be exact with zero uncertainty")
    diagnostics = legacy._mapping(result.get("diagnostics"), "validation diagnostics")
    successful = sum(not value for value in request["training"]["is_penalty"])
    if (diagnostics.get("validation_labels_used_in_fit") is not False
            or diagnostics.get("gp_training_observations") != successful
            or diagnostics.get("validation_observations") != shape[0]):
        raise ValueError("validation diagnostics disagree with the held-out training contract")
    return result


def relay_remote_proposal(
    *, device: DeviceConfig, remote: RemoteProposalConfig, plan_id: str, batch_index: int,
    local_request_path: Path, local_response_path: Path,
    expected_q: int, expected_dimension: int, observed_x_unit: np.ndarray | None = None,
    input_space: InputSpaceLike | Mapping[str, Any] | None = None,
) -> ProposalResult:
    payload, request_sha, request, response = _transfer_remote_request(
        device=device, remote=remote, plan_id=plan_id, batch_index=batch_index,
        local_request_path=local_request_path, local_response_path=local_response_path,
        expected_task="propose",
    )
    result = result_from_response(
        json.loads(local_response_path.read_text(encoding="utf-8-sig")),
        expected_request_sha256=request_sha, expected_q=expected_q,
        expected_dimension=expected_dimension, observed_x_unit=observed_x_unit,
        input_space=input_space if input_space is not None else payload["input_space"],
        exact_contract=payload["objective_contract"]["exact_objective"],
    )
    result.diagnostics.update(
        proposal_executor="remote_ssh", proposal_device_id=device.id,
        proposal_request_sha256=request_sha, proposal_remote_request=str(request),
        proposal_remote_response=str(response),
    )
    return result


def relay_remote_validation(
    *, device: DeviceConfig, remote: RemoteProposalConfig, plan_id: str, batch_index: int,
    local_request_path: Path, local_response_path: Path,
) -> dict[str, Any]:
    payload, request_sha, _, _ = _transfer_remote_request(
        device=device, remote=remote, plan_id=plan_id, batch_index=batch_index,
        local_request_path=local_request_path, local_response_path=local_response_path,
        expected_task="validate",
    )
    return validation_from_response(
        json.loads(local_response_path.read_text(encoding="utf-8-sig")),
        expected_request_sha256=request_sha, request=payload,
    )


def _transfer_remote_request(
    *, device: DeviceConfig, remote: RemoteProposalConfig, plan_id: str, batch_index: int,
    local_request_path: Path, local_response_path: Path, expected_task: str,
) -> tuple[dict[str, Any], str, PureWindowsPath, PureWindowsPath]:
    if device.id != remote.device_id or not device.is_remote:
        raise ValueError("selected tree GPU device must match the SSH remote configuration")
    if not plan_id or any(char in plan_id for char in "/\\:\r\n") or plan_id in (".", ".."):
        raise ValueError("plan_id must be a single directory name")
    payload = json.loads(local_request_path.read_text(encoding="utf-8-sig"))
    validate_request_payload(payload)
    if payload.get("task", "propose") != expected_task:
        raise ValueError("remote request task does not match the requested operation")
    request_sha = sha256_file(local_request_path)
    stem = f"batch_{batch_index:04d}_{request_sha[:16]}"
    root = PureWindowsPath(device.repo_root) / DEFAULT_REMOTE_WORK_ROOT / plan_id
    request, response = root / f"{stem}.request.json", root / f"{stem}.response.json"
    worker = PureWindowsPath(device.repo_root) / PureWindowsPath(WORKER_RELATIVE_PATH)
    prefix = PureWindowsPath(remote.python_path).parent
    literal = legacy._ps_literal
    push_file_atomic(device, local_request_path, str(request), overwrite=True)
    script = "\n".join((
        "$ErrorActionPreference = 'Stop'", "$ProgressPreference = 'SilentlyContinue'",
        f"$python = {literal(remote.python_path)}", f"$worker = {literal(str(worker))}",
        "if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'bocuda Python missing' }",
        "if (-not (Test-Path -LiteralPath $worker -PathType Leaf)) { throw 'Tree worker missing; pull first' }",
        f"$env:CONDA_PREFIX = {literal(str(prefix))}",
        f"$env:PATH = {literal(';'.join(map(str, (prefix, prefix / 'Library/bin', prefix / 'Scripts'))) + ';')} + $env:PATH",
        f"Set-Location -LiteralPath {literal(str(device.repo_root))}",
        f"& $python -B -u $worker --request {literal(str(request))} --response {literal(str(response))}",
        "if ($LASTEXITCODE -ne 0) { throw ('Tree GPU worker exited with code ' + $LASTEXITCODE) }",
    ))
    completed = run_remote_powershell(
        device, script, timeout=remote.timeout_seconds, action="run tree K-RVEA GPU proposal",
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    pull_file_atomic(device, str(response), local_response_path, overwrite=True)
    return payload, request_sha, request, response
