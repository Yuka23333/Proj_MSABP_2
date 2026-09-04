"""Isolated learned-noise GPU relay for deep-optimization Stage 2.

The baseline K-RVEA relay remains untouched.  This worker adds two contracts:

* move the learned likelihood-noise lower bound onto the requested CUDA device;
* apply frozen physical-unit standard-deviation floors after the bounded target
  transforms have been inverted and before K-RVEA standardizes objectives.

The controller uploads this source beside every hash-addressed request, so a
remote Git pull is not required for the worker itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np


def _repository_root() -> Path:
    for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
        if (candidate / "src" / "msabp_opt").is_dir():
            return candidate
    raise RuntimeError("cannot locate repository root containing src/msabp_opt")


REPOSITORY_ROOT = _repository_root()
SRC_ROOT = REPOSITORY_ROOT / "src"
for _import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from msabp_opt.optimization import krvea_relay  # noqa: E402
from msabp_opt.simulation.distributed.transport import (  # noqa: E402
    pull_file_atomic,
    push_file_atomic,
    run_remote_powershell,
)


POLICY_SCHEMA_VERSION = 1
REMOTE_STAGE2_SUBDIRECTORY = "stage2_learned_noise_guard"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class Stage2Policy:
    physical_std_floor: tuple[float, float, float]
    calibration_source: str
    holdout_source: str
    holdout_catastrophic_optimism_count: int

    def __post_init__(self) -> None:
        floors = tuple(float(value) for value in self.physical_std_floor)
        if len(floors) != len(krvea_relay.EXPENSIVE_OBJECTIVE_NAMES):
            raise ValueError("Stage-2 physical_std_floor requires three values")
        if not np.isfinite(floors).all() or any(value <= 0.0 for value in floors):
            raise ValueError("Stage-2 physical_std_floor values must be finite and positive")
        if not self.calibration_source.strip() or not self.holdout_source.strip():
            raise ValueError("Stage-2 calibration and holdout sources must be non-empty")
        if self.holdout_catastrophic_optimism_count < 0:
            raise ValueError("holdout catastrophic count cannot be negative")
        object.__setattr__(self, "physical_std_floor", floors)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Stage2Policy":
        if int(value.get("schema_version", -1)) != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported Stage-2 policy schema")
        floor_map = _mapping(value.get("physical_std_floor"), "physical_std_floor")
        expected = set(krvea_relay.EXPENSIVE_OBJECTIVE_NAMES)
        unknown = set(floor_map) - expected
        missing = expected - set(floor_map)
        if unknown or missing:
            raise ValueError(
                "Stage-2 physical_std_floor objective mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(
            physical_std_floor=tuple(
                float(floor_map[name])
                for name in krvea_relay.EXPENSIVE_OBJECTIVE_NAMES
            ),
            calibration_source=str(value.get("calibration_source", "")),
            holdout_source=str(value.get("holdout_source", "")),
            holdout_catastrophic_optimism_count=int(
                value.get("holdout_catastrophic_optimism_count", -1)
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "physical_std_floor": dict(
                zip(
                    krvea_relay.EXPENSIVE_OBJECTIVE_NAMES,
                    self.physical_std_floor,
                    strict=True,
                )
            ),
            "calibration_source": self.calibration_source,
            "holdout_source": self.holdout_source,
            "holdout_catastrophic_optimism_count": (
                self.holdout_catastrophic_optimism_count
            ),
            "application_layer": "physical_after_target_inverse_before_objective_scale",
            "worker_source_sha256": krvea_relay.sha256_python_source(__file__),
        }


def attach_policy(
    request_payload: Mapping[str, Any], policy: Stage2Policy
) -> dict[str, Any]:
    request = dict(_mapping(request_payload, "request"))
    if "stage2_policy" in request:
        raise ValueError("proposal request already contains stage2_policy")
    request["stage2_policy"] = policy.to_wire()
    return request


def apply_physical_std_floor(
    std_physical: np.ndarray, floors: Sequence[float]
) -> np.ndarray:
    values = np.asarray(std_physical, dtype=np.float64)
    floor = np.asarray(floors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(floor):
        raise ValueError("physical surrogate std and floor shapes are incompatible")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("physical surrogate std must be finite and non-negative")
    return np.maximum(values, floor[None, :])


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


def _validated_worker_policy(request: Mapping[str, Any]) -> Stage2Policy:
    policy_wire = _mapping(request.get("stage2_policy"), "stage2_policy")
    expected_source = krvea_relay.sha256_python_source(__file__)
    if policy_wire.get("worker_source_sha256") != expected_source:
        raise RuntimeError(
            "Stage-2 worker source differs from the proposal request fingerprint"
        )
    if policy_wire.get("application_layer") != (
        "physical_after_target_inverse_before_objective_scale"
    ):
        raise ValueError("unexpected Stage-2 uncertainty application layer")
    return Stage2Policy.from_mapping(policy_wire)


def execute_request_file(request_path: Path, response_path: Path) -> bool:
    """Run one Stage-2 request after installing process-local CUDA adapters."""

    request_file = request_path.expanduser().resolve()
    response_file = response_path.expanduser().resolve()
    request_sha256 = krvea_relay.sha256_file(request_file)
    if response_file.is_file():
        existing = json.loads(response_file.read_text(encoding="utf-8-sig"))
        if existing.get("request_sha256") != request_sha256:
            raise RuntimeError("existing Stage-2 response has a different request hash")
        if existing.get("status") != "completed":
            raise RuntimeError("existing Stage-2 response is not completed")
        return True

    request = _mapping(
        json.loads(request_file.read_text(encoding="utf-8-sig")), "request"
    )
    policy = _validated_worker_policy(request)
    compute = _mapping(request.get("compute"), "compute")
    device_name = str(compute.get("device", ""))
    if not device_name.lower().startswith("cuda"):
        raise ValueError("Stage-2 learned-noise worker requires a CUDA device")
    settings = _mapping(request.get("surrogate_settings"), "surrogate_settings")
    if settings.get("gp_noise_mode") != "learned":
        raise ValueError("Stage-2 worker requires gp_noise_mode='learned'")

    original_imports = krvea_relay._botorch_imports
    original_inverse = krvea_relay.SurrogateTargetScaler.inverse_prediction

    def cuda_constraint_imports() -> dict[str, Any]:
        runtime = dict(original_imports())
        torch = runtime["torch"]
        greater_than_type = runtime["GreaterThan"]

        def cuda_greater_than(lower_bound: Any, *args: Any, **kwargs: Any) -> Any:
            bound = torch.as_tensor(
                lower_bound,
                dtype=torch.float64,
                device=torch.device(device_name),
            )
            return greater_than_type(bound, *args, **kwargs)

        runtime["GreaterThan"] = cuda_greater_than
        return runtime

    def guarded_inverse(
        scaler: krvea_relay.SurrogateTargetScaler,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        quadrature_order: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        physical_mean, physical_std = original_inverse(
            scaler,
            mean,
            std,
            quadrature_order=quadrature_order,
        )
        return physical_mean, apply_physical_std_floor(
            physical_std, policy.physical_std_floor
        )

    try:
        krvea_relay._botorch_imports = cuda_constraint_imports
        krvea_relay.SurrogateTargetScaler.inverse_prediction = guarded_inverse
        result = krvea_relay.run_request_payload(request)
    finally:
        krvea_relay._botorch_imports = original_imports
        krvea_relay.SurrogateTargetScaler.inverse_prediction = original_inverse

    result.diagnostics["stage2_policy"] = {
        **policy.to_wire(),
        "worker_source_sha256": krvea_relay.sha256_python_source(__file__),
        "learned_noise_cuda_constraint_adapter": True,
    }
    _atomic_write_json(
        response_file, krvea_relay.response_payload(request_sha256, result)
    )
    return False


def _ps_literal(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if any(character in text for character in ("\0", "\r", "\n")):
        raise ValueError("PowerShell literal contains a control character")
    return "'" + text.replace("'", "''") + "'"


def relay_remote_proposal(
    *,
    device: Any,
    remote: Any,
    plan_id: str,
    batch_index: int,
    local_request_path: Path,
    local_response_path: Path,
    expected_q: int,
    expected_dimension: int,
    observed_x_unit: np.ndarray,
    input_space: Any,
) -> krvea_relay.ProposalResult:
    """Upload this isolated worker, execute one proposal, and validate it."""

    if device.id != remote.device_id:
        raise ValueError("selected GPU device does not match Stage-2 remote config")
    if not device.is_remote:
        raise ValueError("Stage-2 GPU proposal device must be SSH-addressable")
    request_sha256 = krvea_relay.sha256_file(local_request_path)
    stem = f"batch_{batch_index:04d}_{request_sha256[:16]}"
    remote_root = (
        PureWindowsPath(device.repo_root)
        / krvea_relay.DEFAULT_REMOTE_WORK_ROOT
        / plan_id
        / REMOTE_STAGE2_SUBDIRECTORY
    )
    remote_request = remote_root / f"{stem}.request.json"
    remote_response = remote_root / f"{stem}.response.json"
    worker_hash = krvea_relay.sha256_python_source(__file__)
    remote_worker = remote_root / f"worker_{worker_hash[:16]}.py"

    push_file_atomic(device, __file__, str(remote_worker), overwrite=True)
    push_file_atomic(device, local_request_path, str(remote_request), overwrite=True)
    script = "\n".join(
        (
            "$ErrorActionPreference = 'Stop'",
            f"$python = {_ps_literal(remote.python_path)}",
            f"$worker = {_ps_literal(str(remote_worker))}",
            "if (-not (Test-Path -LiteralPath $python -PathType Leaf)) "
            "{ throw 'bocuda Python executable does not exist' }",
            f"& $python $worker --worker --request {_ps_literal(str(remote_request))} "
            f"--response {_ps_literal(str(remote_response))}",
            "if ($LASTEXITCODE -ne 0) "
            "{ throw ('Stage-2 GPU worker exited with code ' + $LASTEXITCODE) }",
        )
    )
    completed = run_remote_powershell(
        device,
        script,
        timeout=remote.timeout_seconds,
        action="run Stage-2 learned-noise GPU proposal",
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    pull_file_atomic(device, str(remote_response), local_response_path, overwrite=True)
    response = json.loads(local_response_path.read_text(encoding="utf-8-sig"))
    result = krvea_relay.result_from_response(
        _mapping(response, "response"),
        expected_request_sha256=request_sha256,
        expected_q=expected_q,
        expected_dimension=expected_dimension,
        observed_x_unit=observed_x_unit,
        input_space=input_space,
    )
    result.diagnostics.update(
        {
            "proposal_executor": "remote_ssh_stage2_worker",
            "proposal_device_id": device.id,
            "proposal_request_sha256": request_sha256,
            "proposal_remote_request": str(remote_request),
            "proposal_remote_response": str(remote_response),
            "stage2_worker_source_sha256": worker_hash,
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.worker or args.request is None or args.response is None:
        raise SystemExit("this relay is executed directly only in --worker mode")
    reused = execute_request_file(args.request, args.response)
    action = "reused" if reused else "completed"
    print(f"[Stage-2 GPU] {action}: {args.response}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
