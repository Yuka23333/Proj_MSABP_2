"""File-backed relay between the local campaign controller and a GPU host.

The local controller remains authoritative for observations, campaign state,
geometry preflight, penalties, and the Princess/Maid solver run.  Only compact
normalized training arrays cross this boundary.  A request hash makes retries
idempotent and leaves an auditable request/response pair beside each BO batch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import numpy as np
import pandas as pd

from msabp_opt.simulation.distributed.config import DeviceConfig
from msabp_opt.simulation.distributed.transport import (
    pull_file_atomic,
    push_file_atomic,
    run_remote_powershell,
)

from . import qlogehvi


REQUEST_SCHEMA_VERSION = 1
RESPONSE_SCHEMA_VERSION = 1
DEFAULT_REMOTE_TIMEOUT_SECONDS = 1800.0
DEFAULT_REMOTE_WORK_ROOT = PureWindowsPath(
    "simulations", "runs", "qlogehvi_gpu"
)


@dataclass(frozen=True)
class RemoteProposalConfig:
    """How the local controller reaches one proposal-only GPU worker."""

    device_id: str = "coconutg2"
    python_path: str = (
        r"C:\Users\telecom\miniforge3\envs\bocuda\python.exe"
    )
    compute_device: str = "cuda"
    timeout_seconds: float = DEFAULT_REMOTE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("proposal device_id must be non-empty")
        python_path = PureWindowsPath(self.python_path)
        if not python_path.is_absolute():
            raise ValueError("proposal python_path must be an absolute Windows path")
        if not self.compute_device.strip():
            raise ValueError("proposal compute_device must be non-empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("proposal timeout_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def input_space_from_payload(payload: Mapping[str, Any]) -> qlogehvi.InputSpace:
    names = payload.get("parameter_names")
    lower = payload.get("lower")
    upper = payload.get("upper")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("request input_space.parameter_names must be a string list")
    return qlogehvi.InputSpace(
        names=tuple(names),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
    )


def build_request_payload(
    observations: pd.DataFrame,
    input_space: qlogehvi.InputSpace,
    *,
    settings: qlogehvi.ProposalSettings,
    iteration: int,
    compute_device: str = "cuda",
) -> dict[str, Any]:
    """Compress local observations into the stable GPU-worker wire format."""

    train_x, train_y_rf, train_y_full, aggregate = qlogehvi.training_arrays(
        observations,
        input_space,
    )
    if len(train_x) < 2:
        raise ValueError("at least two distinct observations are required for GP fitting")
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "algorithm": "qLogExpectedHypervolumeImprovement",
        "iteration": int(iteration),
        "compute": {"device": str(compute_device), "dtype": "float64"},
        "input_space": input_space.to_dict(),
        "proposal_settings": asdict(settings),
        "training": {
            "x_unit": train_x.tolist(),
            "y_rf_maximize": train_y_rf.tolist(),
            "y_full_maximize": train_y_full.tolist(),
            "summary": {
                "training_observations_raw": int(len(observations)),
                "training_observations_distinct": int(len(train_x)),
                "penalty_observations": int(observations["is_penalty"].sum()),
                "replicate_groups": int((aggregate["replicate_count"] > 1).sum()),
            },
        },
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def run_request_payload(payload: Mapping[str, Any]) -> qlogehvi.ProposalResult:
    """Execute a validated proposal request in the current Python process."""

    if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported qLogEHVI proposal request schema")
    if payload.get("algorithm") != "qLogExpectedHypervolumeImprovement":
        raise ValueError("unsupported proposal algorithm")
    compute = _mapping(payload.get("compute"), "compute")
    if compute.get("dtype") != "float64":
        raise ValueError("GPU proposal requests must use float64")
    input_space = input_space_from_payload(
        _mapping(payload.get("input_space"), "input_space")
    )
    settings = qlogehvi.ProposalSettings(
        **dict(_mapping(payload.get("proposal_settings"), "proposal_settings"))
    )
    training = _mapping(payload.get("training"), "training")
    summary = {
        str(name): int(value)
        for name, value in _mapping(training.get("summary"), "training.summary").items()
    }
    return qlogehvi.propose_qlogehvi_from_training_arrays(
        np.asarray(training.get("x_unit"), dtype=float),
        np.asarray(training.get("y_rf_maximize"), dtype=float),
        np.asarray(training.get("y_full_maximize"), dtype=float),
        input_space,
        settings=settings,
        iteration=int(payload.get("iteration", 0)),
        device_name=str(compute.get("device", "")),
        training_summary=summary,
    )


def response_payload(
    request_sha256: str,
    result: qlogehvi.ProposalResult,
) -> dict[str, Any]:
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "status": "completed",
        "request_sha256": request_sha256,
        "result": {
            "unit_values": result.unit_values.tolist(),
            "raw_values": result.raw_values.tolist(),
            "acquisition_values": list(result.acquisition_values),
            "diagnostics": result.diagnostics,
        },
    }


def result_from_response(
    payload: Mapping[str, Any],
    *,
    expected_request_sha256: str,
    expected_q: int,
    expected_dimension: int,
) -> qlogehvi.ProposalResult:
    if payload.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise ValueError("unsupported qLogEHVI proposal response schema")
    if payload.get("status") != "completed":
        raise ValueError("qLogEHVI proposal response is not completed")
    if payload.get("request_sha256") != expected_request_sha256:
        raise ValueError("proposal response belongs to a different request")
    result = _mapping(payload.get("result"), "result")
    unit_values = np.asarray(result.get("unit_values"), dtype=float)
    raw_values = np.asarray(result.get("raw_values"), dtype=float)
    expected_shape = (expected_q, expected_dimension)
    if unit_values.shape != expected_shape or raw_values.shape != expected_shape:
        raise ValueError(
            f"proposal response shape must be {expected_shape}, got "
            f"{unit_values.shape} and {raw_values.shape}"
        )
    if not np.isfinite(unit_values).all() or not np.isfinite(raw_values).all():
        raise ValueError("proposal response contains non-finite candidates")
    if np.any(unit_values < -1e-12) or np.any(unit_values > 1.0 + 1e-12):
        raise ValueError("proposal response contains normalized values outside [0, 1]")
    acquisition_values = tuple(
        float(value) for value in result.get("acquisition_values", [])
    )
    diagnostics = dict(_mapping(result.get("diagnostics"), "result.diagnostics"))
    return qlogehvi.ProposalResult(
        unit_values=unit_values,
        raw_values=raw_values,
        acquisition_values=acquisition_values,
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
) -> qlogehvi.ProposalResult:
    """Upload, execute, and retrieve one idempotent coconutg2 proposal."""

    if device.id != remote.device_id:
        raise ValueError("selected GPU device does not match RemoteProposalConfig")
    if not device.is_remote:
        raise ValueError("remote GPU proposal device must be SSH-addressable")
    request_sha = sha256_file(local_request_path)
    stem = f"batch_{batch_index:04d}_{request_sha[:16]}"
    remote_root = (
        PureWindowsPath(device.repo_root)
        / DEFAULT_REMOTE_WORK_ROOT
        / plan_id
    )
    remote_request = remote_root / f"{stem}.request.json"
    remote_response = remote_root / f"{stem}.response.json"
    worker_path = (
        PureWindowsPath(device.repo_root)
        / "scripts"
        / "optimization"
        / "qlogehvi_gpu_worker.py"
    )
    push_file_atomic(
        device,
        local_request_path,
        str(remote_request),
        overwrite=True,
    )
    script = "\n".join(
        (
            "$ErrorActionPreference = 'Stop'",
            f"$python = {_ps_literal(remote.python_path)}",
            f"$worker = {_ps_literal(str(worker_path))}",
            "if (-not (Test-Path -LiteralPath $python -PathType Leaf)) "
            "{ throw 'bocuda Python executable does not exist' }",
            "if (-not (Test-Path -LiteralPath $worker -PathType Leaf)) "
            "{ throw 'qLogEHVI GPU worker does not exist; pull the repository first' }",
            f"& $python $worker --request {_ps_literal(str(remote_request))} "
            f"--response {_ps_literal(str(remote_response))}",
            "if ($LASTEXITCODE -ne 0) "
            "{ throw ('qLogEHVI GPU worker exited with code ' + $LASTEXITCODE) }",
        )
    )
    completed = run_remote_powershell(
        device,
        script,
        timeout=remote.timeout_seconds,
        action="run qLogEHVI GPU proposal",
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    pull_file_atomic(
        device,
        str(remote_response),
        local_response_path,
        overwrite=True,
    )
    payload = json.loads(local_response_path.read_text(encoding="utf-8-sig"))
    result = result_from_response(
        _mapping(payload, "response"),
        expected_request_sha256=request_sha,
        expected_q=expected_q,
        expected_dimension=expected_dimension,
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
