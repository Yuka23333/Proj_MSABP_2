"""Run a small, resumable, real CST solve on the two remote Maid hosts.

Press F5 in an IDE to use the constants below.  Unlike the ordinary Princess
CLI this entrypoint deliberately asks for an explicit ``RUN`` confirmation,
because it starts licensed CST solvers and performs real simulations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence, TextIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from msabp_opt.simulation.distributed.config import (  # noqa: E402
    DEFAULT_DEVICE_CONFIG_PATH,
    LaunchMode,
    load_device_registry,
)
from scripts.automation import antenna_sampler  # noqa: E402
from scripts.automation import cst_build_msabp_geometry  # noqa: E402


# F5 settings.  Keep RUN_ID unchanged to resume an interrupted run.  Use a new
# value if the sampling config or generated CSV is intentionally changed.
RUN_ID = "remote-real-smoke-006"
DEVICE_IDS = ("convallariag5", "coconutg2")
CANDIDATE_COUNT = 4
REQUIRE_CONFIRMATION = True
SMOKE_GLOBAL_RELATIVE_RANGE = (-0.02, 0.02)

SAMPLING_CONFIG = (
    REPOSITORY_ROOT / "configs" / "optimization" / "antenna_sampling.json"
)
DEVICE_CONFIG = DEFAULT_DEVICE_CONFIG_PATH
PROJECT_TEMPLATE = REPOSITORY_ROOT / "simulations" / "models" / "msa-bp.cst"
PRINCESS_SCRIPT = REPOSITORY_ROOT / "scripts" / "simulation" / "princess.py"


@dataclass(frozen=True)
class PreparedSmokeInput:
    """Paths and geometry policy for a generated real-solve input."""

    csv_path: Path
    resolved_path: Path
    row_count: int
    coordinate_quantum_mm: float
    allow_disconnected_conductor: bool


def default_input_path(run_id: str) -> Path:
    """Put generated input beside the durable Princess state for this run."""

    return REPOSITORY_ROOT / "simulations" / "runs" / run_id / "input_samples.csv"


def default_log_path(run_id: str) -> Path:
    return REPOSITORY_ROOT / "logs" / f"princess.{run_id}.real-smoke.log"


def prepare_input(
    output_path: str | Path,
    *,
    candidate_count: int = CANDIDATE_COUNT,
    config_path: str | Path = SAMPLING_CONFIG,
) -> PreparedSmokeInput:
    """Generate Sobol candidates and revalidate their persisted CSV values."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")

    config = copy.deepcopy(antenna_sampler.load_sampling_config(config_path))
    smoke_lower, smoke_upper = SMOKE_GLOBAL_RELATIVE_RANGE
    smoke_range = {
        "mode": "relative",
        "lower": smoke_lower,
        "upper": smoke_upper,
        "reference": "nominal",
    }
    for name, parameter_config in config["sampling"]["parameters"].items():
        nominal = float(
            parameter_config.get(
                "value",
                antenna_sampler.PARAMETER_REGISTRY[name].code_default,
            )
        )
        if nominal == 0.0:
            parameter_config["range"] = {
                "mode": "absolute",
                "min": 0.0,
                "max": smoke_upper,
            }
        else:
            parameter_config["range"] = dict(smoke_range)
    draw_count = candidate_count
    result: antenna_sampler.SamplingResult | None = None
    for _attempt in range(9):
        plan = antenna_sampler.resolve_sampling_plan(config, n_samples=draw_count)
        drawn = antenna_sampler.generate_samples(plan)
        valid_frame = drawn.frame.loc[drawn.frame["geometry_valid"]].head(
            candidate_count
        )
        if len(valid_frame) >= candidate_count:
            valid_frame = valid_frame.copy().reset_index(drop=True)
            valid_frame["sample_id"] = range(candidate_count)
            result = antenna_sampler.SamplingResult(
                frame=valid_frame,
                plan=replace(plan, n_samples=candidate_count),
            )
            plan = result.plan
            break
        draw_count *= 2
    if result is None:
        raise RuntimeError(
            f"could not find {candidate_count} valid candidates after checking "
            f"{draw_count // 2} Sobol points; no real solve was started"
        )

    csv_path, resolved_path = antenna_sampler.save_sampling_result(
        result,
        output_path,
        valid_only=True,
    )
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        persisted_rows = list(csv.DictReader(handle))

    if len(persisted_rows) != candidate_count:
        raise RuntimeError(
            f"expected {candidate_count} persisted rows, found {len(persisted_rows)}"
        )
    for row in persisted_rows:
        parameters = antenna_sampler.parameters_from_csv_row(row)
        cst_build_msabp_geometry.build_sampled_polygon_specs(
            parameters,
            coordinate_quantum_mm=plan.geometry_policy.coordinate_quantum_mm,
        )

    return PreparedSmokeInput(
        csv_path=csv_path.resolve(),
        resolved_path=resolved_path.resolve(),
        row_count=len(persisted_rows),
        coordinate_quantum_mm=plan.geometry_policy.coordinate_quantum_mm,
        allow_disconnected_conductor=(
            plan.geometry_policy.allow_disconnected_conductor
        ),
    )


def validate_devices(
    device_ids: Sequence[str],
    *,
    device_config: str | Path = DEVICE_CONFIG,
) -> None:
    """Fail locally when an F5 device selection cannot use Maid Bell."""

    if not device_ids:
        raise ValueError("at least one Maid device is required")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("Maid device ids must be unique")

    registry = load_device_registry(device_config)
    for device_id in device_ids:
        device = registry.get_device(device_id)
        if device.launch_mode is not LaunchMode.BELL:
            raise ValueError(
                f"device {device_id!r} uses {device.launch_mode.value!r}, not 'bell'"
            )
        if not device.bell_host:
            raise ValueError(f"device {device_id!r} has no bell_host")


def build_princess_command(
    prepared: PreparedSmokeInput,
    *,
    run_id: str,
    device_ids: Sequence[str],
    project_template: str | Path = PROJECT_TEMPLATE,
    device_config: str | Path = DEVICE_CONFIG,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the real Princess command; this function never adds --dry-run."""

    command = [
        python_executable,
        str(PRINCESS_SCRIPT),
        "start",
        "--csv",
        str(prepared.csv_path),
        "--run-id",
        run_id,
        "--project",
        str(Path(project_template).resolve()),
        "--devices-config",
        str(Path(device_config).resolve()),
        "--coordinate-quantum-mm",
        str(prepared.coordinate_quantum_mm),
    ]
    for device_id in device_ids:
        command.extend(("--device", device_id))
    if prepared.allow_disconnected_conductor:
        command.append("--allow-disconnected-conductor")
    return command


def run_and_tee(
    command: Sequence[str],
    log_path: str | Path,
    *,
    output: TextIO = sys.stdout,
) -> int:
    """Run Princess while mirroring its combined output to a durable log."""

    resolved_log = Path(log_path).resolve()
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    with resolved_log.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"\n$ {subprocess.list2cmdline(list(command))}\n")
        process = subprocess.Popen(  # noqa: S603 - fixed local script entrypoint
            list(command),
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            log_handle.write(line)
        return process.wait()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four valid Sobol cases and run a real remote CST smoke."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--candidate-count", type=int, default=CANDIDATE_COUNT)
    parser.add_argument(
        "--device",
        action="append",
        dest="device_ids",
        help="Maid Bell device id; repeat to override the two F5 defaults",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive RUN confirmation",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="generate and validate the CSV without starting Princess or CST",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device_ids = tuple(args.device_ids or DEVICE_IDS)

    if not PROJECT_TEMPLATE.is_file():
        raise FileNotFoundError(f"CST project template not found: {PROJECT_TEMPLATE}")
    validate_devices(device_ids)

    prepared = prepare_input(
        default_input_path(args.run_id),
        candidate_count=args.candidate_count,
    )
    print(
        f"Prepared {prepared.row_count} persisted-and-revalidated cases:\n"
        f"  CSV:      {prepared.csv_path}\n"
        f"  resolved: {prepared.resolved_path}"
    )
    if args.prepare_only:
        print("Prepare-only complete: Princess and CST were not started.")
        return 0

    command = build_princess_command(
        prepared,
        run_id=args.run_id,
        device_ids=device_ids,
    )
    log_path = default_log_path(args.run_id)
    print(
        f"REAL CST solve: run={args.run_id}, devices={','.join(device_ids)}\n"
        f"Live log: {log_path}\n"
        "Reusing this run ID with the same CSV resumes durable Princess state."
    )
    if REQUIRE_CONFIRMATION and not args.yes:
        confirmation = input("Type RUN to start the real CST solve: ").strip()
        if confirmation != "RUN":
            print("Cancelled: Princess and CST were not started.")
            return 0

    return_code = run_and_tee(command, log_path)
    print(
        f"Princess exited with code {return_code}.\n"
        f"Log: {log_path}\n"
        f"Status: {sys.executable} {PRINCESS_SCRIPT} status --run-id {args.run_id}"
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
