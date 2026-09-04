"""CUDA constraint adapter for Stage-1 learned-noise replay.

BoTorch's learned Gaussian-likelihood noise constraint can retain a CPU lower
bound while the GP lives on CUDA.  This isolated worker converts only that
constraint bound to the requested CUDA device, then delegates all replay,
validation, and response writing to ``深度优化_阶段1代理回放.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Sequence


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

from msabp_opt.simulation.distributed.config import (  # noqa: E402
    DEFAULT_DEVICE_CONFIG_PATH,
    load_device_registry,
)
from msabp_opt.simulation.distributed.transport import (  # noqa: E402
    pull_file_atomic,
    push_file_atomic,
    run_remote_powershell,
)


DEVICE_ID = "coconutg2"
REMOTE_PYTHON = r"C:\Users\telecom\miniforge3\envs\bocuda\python.exe"
REMOTE_TIMEOUT_SECONDS = 3600.0
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "deep_surrogate_replay_round4_round5"
)
VARIANT = "learned_floor_1e-4"


def _ps_literal(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if any(character in text for character in ("\0", "\r", "\n")):
        raise ValueError("PowerShell literal contains a control character")
    return "'" + text.replace("'", "''") + "'"


def _load_replay_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("deep_surrogate_replay_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load replay worker: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_worker(main_script: Path, request_path: Path, response_path: Path) -> None:
    replay = _load_replay_module(main_script)
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    if request.get("variant") != VARIANT:
        raise ValueError(f"adapter accepts only {VARIANT}")

    original_imports = replay.krvea_relay._botorch_imports

    def cuda_constraint_imports() -> dict[str, Any]:
        runtime = dict(original_imports())
        torch = runtime["torch"]
        greater_than_type = runtime["GreaterThan"]

        def cuda_greater_than(lower_bound: Any, *args: Any, **kwargs: Any) -> Any:
            device = torch.device(str(request["device"]))
            bound = torch.as_tensor(lower_bound, dtype=torch.float64, device=device)
            return greater_than_type(bound, *args, **kwargs)

        runtime["GreaterThan"] = cuda_greater_than
        return runtime

    replay.krvea_relay._botorch_imports = cuda_constraint_imports
    replay.run_worker(request_path, response_path)


def relay() -> Path:
    request_path = OUTPUT_DIRECTORY / f"request.{VARIANT}.json"
    response_path = OUTPUT_DIRECTORY / f"response.{VARIANT}.json"
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    request_sha256 = str(request["request_sha256"])
    device = load_device_registry(DEFAULT_DEVICE_CONFIG_PATH).get_device(DEVICE_ID)
    remote_directory = (
        PureWindowsPath(device.repo_root)
        / "simulations"
        / "runs"
        / "deep_surrogate_replay"
        / request_sha256
    )
    remote_adapter = remote_directory / "learned_noise_cuda_worker.py"
    remote_main = remote_directory / "deep_surrogate_replay_worker.py"
    remote_request = remote_directory / f"request.{VARIANT}.json"
    remote_response = remote_directory / f"response.{VARIANT}.json"
    push_file_atomic(device, __file__, str(remote_adapter), transfer_timeout=180.0)
    command = (
        f"Set-Location -LiteralPath {_ps_literal(device.repo_root)}; "
        f"& {_ps_literal(REMOTE_PYTHON)} {_ps_literal(remote_adapter)} "
        f"--worker --main-script {_ps_literal(remote_main)} "
        f"--request {_ps_literal(remote_request)} --response {_ps_literal(remote_response)}"
    )
    completed = run_remote_powershell(
        device,
        command,
        timeout=REMOTE_TIMEOUT_SECONDS,
        action="run learned-noise CUDA replay adapter",
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    pull_file_atomic(
        device, str(remote_response), response_path, transfer_timeout=180.0
    )
    response = json.loads(response_path.read_text(encoding="utf-8-sig"))
    if response.get("request_sha256") != request_sha256:
        raise RuntimeError("learned-noise response/request hash mismatch")
    return response_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--main-script", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.main_script is None or args.request is None or args.response is None:
            raise SystemExit("--worker requires --main-script, --request, and --response")
        run_worker(args.main_script, args.request, args.response)
        return 0
    path = relay()
    print(f"[Replay] learned-noise response: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
