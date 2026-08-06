"""Generate and label the redesigned 23-dimensional Sobol antenna dataset.

F5 runs the single ``redesigned_23d`` topology with the constants below.  The
output is a NumPy structured array loadable with ``allow_pickle=False``;
``geometry_valid`` is the Boolean label.
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from shapely.errors import ShapelyError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import antenna_sampler  # noqa: E402
from scripts.automation import cst_build_msabp_geometry  # noqa: E402
from scripts.geometry import shapely_antenna_model as antenna_outline  # noqa: E402


SAMPLE_COUNT = 65_536
WORKER_COUNT = 24
CHUNK_SIZE = 256
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data" / "samples"
OVERWRITE_EXISTING = False

SCENARIO_BRANCH_STATES: dict[str, dict[str, bool]] = {"redesigned_23d": {}}
PARAMETER_NAMES = tuple(antenna_sampler.PARAMETER_REGISTRY)


def configure_scenario(
    config: Mapping[str, Any],
    scenario: str,
) -> dict[str, Any]:
    """Return a copied sampler config with one explicit branch topology."""

    if scenario not in SCENARIO_BRANCH_STATES:
        raise ValueError(f"unknown branch scenario: {scenario}")
    return copy.deepcopy(dict(config))


def _chunk_parameter_rows(
    frame: pd.DataFrame,
    chunk_size: int,
) -> Iterable[tuple[tuple[Any, ...], ...]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    parameter_frame = frame.loc[:, PARAMETER_NAMES]
    for start in range(0, len(parameter_frame), chunk_size):
        stop = min(start + chunk_size, len(parameter_frame))
        yield tuple(
            parameter_frame.iloc[start:stop].itertuples(index=False, name=None)
        )


def _validate_parameter_chunk(
    payload: tuple[tuple[tuple[Any, ...], ...], float, bool],
) -> np.ndarray:
    """Validate one chunk inside a worker process and return Boolean labels."""

    rows, coordinate_quantum_mm, allow_disconnected_conductor = payload
    labels = np.zeros(len(rows), dtype=np.bool_)
    for index, values in enumerate(rows):
        parameters = antenna_outline.ShapelyAntennaParameters(
            **dict(zip(PARAMETER_NAMES, values, strict=True))
        )
        try:
            cst_build_msabp_geometry.build_sampled_polygon_specs(
                parameters,
                coordinate_quantum_mm=coordinate_quantum_mm,
            )
        except (TypeError, ValueError, ShapelyError):
            continue
        labels[index] = True
    return labels


def label_parameter_frame_parallel(
    frame: pd.DataFrame,
    plan: antenna_sampler.SamplingPlan,
    *,
    workers: int = WORKER_COUNT,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Label every parameter row with ordered 24-worker process parallelism."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    labels = np.empty(len(frame), dtype=np.bool_)
    payloads = (
        (
            rows,
            plan.geometry_policy.coordinate_quantum_mm,
            plan.geometry_policy.allow_disconnected_conductor,
        )
        for rows in _chunk_parameter_rows(frame, chunk_size)
    )
    completed = 0
    started_at = time.perf_counter()
    spawn_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=spawn_context,
    ) as executor:
        for chunk_labels in executor.map(
            _validate_parameter_chunk,
            payloads,
            chunksize=1,
        ):
            stop = completed + len(chunk_labels)
            labels[completed:stop] = chunk_labels
            completed = stop
            if completed % 4096 == 0 or completed == len(frame):
                elapsed = time.perf_counter() - started_at
                print(
                    f"[geometry-labels] checked={completed}/{len(frame)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return labels


def build_labeled_array(
    frame: pd.DataFrame,
    labels: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Build the pickle-free structured NPY payload."""

    labels_array = np.asarray(labels, dtype=np.bool_)
    if labels_array.shape != (len(frame),):
        raise ValueError(
            f"labels must have shape ({len(frame)},), got {labels_array.shape}"
        )

    dtype_fields: list[tuple[str, str]] = [("sample_id", "<i8")]
    for name, spec in antenna_sampler.PARAMETER_REGISTRY.items():
        dtype_fields.append((name, "<f8"))
    dtype_fields.append(("geometry_valid", "?"))

    labeled = np.empty(len(frame), dtype=np.dtype(dtype_fields))
    labeled["sample_id"] = frame["sample_id"].to_numpy(dtype=np.int64)
    for name, spec in antenna_sampler.PARAMETER_REGISTRY.items():
        labeled[name] = frame[name].to_numpy(dtype=np.float64)
    labeled["geometry_valid"] = labels_array
    return labeled


def save_labeled_array(
    array: np.ndarray,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save one NPY file without object arrays or pickle."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def run_scenario(
    scenario: str,
    *,
    sample_count: int = SAMPLE_COUNT,
    workers: int = WORKER_COUNT,
    chunk_size: int = CHUNK_SIZE,
    output_directory: str | Path = OUTPUT_DIRECTORY,
    output_tag: str | None = None,
    overwrite: bool = OVERWRITE_EXISTING,
) -> Path:
    """Sample, label, save, reload, and verify one branch scenario."""

    base_config = antenna_sampler.load_sampling_config()
    config = configure_scenario(base_config, scenario)
    plan = antenna_sampler.resolve_sampling_plan(config, n_samples=sample_count)
    active_dimension_count = sum(
        item.effective_sample for item in plan.resolved_parameters
    )
    print(
        f"[geometry-labels] scenario={scenario} samples={sample_count} "
        f"dimensions={active_dimension_count} method={plan.method} workers={workers}",
        flush=True,
    )

    frame = antenna_sampler.generate_parameter_frame(plan)
    labels = label_parameter_frame_parallel(
        frame,
        plan,
        workers=workers,
        chunk_size=chunk_size,
    )
    labeled = build_labeled_array(frame, labels)
    if output_tag is not None and (
        not output_tag
        or not all(character.isalnum() or character in "-_" for character in output_tag)
    ):
        raise ValueError("output_tag may contain only letters, digits, '-' and '_'")
    tag = "" if output_tag is None else f"_{output_tag}"
    filename = f"antenna_geometry_labels_{sample_count}_{scenario}{tag}.npy"
    output = save_labeled_array(
        labeled,
        Path(output_directory) / filename,
        overwrite=overwrite,
    )

    loaded = np.load(output, allow_pickle=False, mmap_mode="r")
    if loaded.shape != (sample_count,):
        raise RuntimeError(f"saved NPY has unexpected shape: {loaded.shape}")
    if loaded.dtype.names != labeled.dtype.names:
        raise RuntimeError("saved NPY field names changed during round trip")
    valid_count = int(np.count_nonzero(loaded["geometry_valid"]))
    invalid_count = sample_count - valid_count
    print(
        f"[geometry-labels] saved={output} valid={valid_count} "
        f"invalid={invalid_count} bytes={output.stat().st_size}",
        flush=True,
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIO_BRANCH_STATES),
        default="all",
    )
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--workers", type=int, default=WORKER_COUNT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--output-tag")
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_EXISTING)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    scenarios = (
        tuple(SCENARIO_BRANCH_STATES)
        if args.scenario == "all"
        else (args.scenario,)
    )
    for scenario in scenarios:
        run_scenario(
            scenario,
            sample_count=args.samples,
            workers=args.workers,
            chunk_size=args.chunk_size,
            output_directory=args.output_directory,
            output_tag=args.output_tag,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
