"""Prepare 10-degree "watermelon latitude" gain slices for every sample.

For each CST far-field source, the sphere is partitioned into the 18 adjacent
belts theta=[10k, 10(k+1)] for k=0..17. Pattern intensity is integrated with
the physical solid-angle measure sin(theta) dtheta dphi. Directivity, gain,
and realized gain are preserved in linear form at every frequency.

The default archive is the frozen 960-row snapshot after the 512-point DoE and
six K-RVEA rounds. Completed, non-penalty cases are included, together with
the current-default and Roblin--Wei reference far fields. Text parsing, not
floating-point integration, dominates runtime, so independent FFS files are
processed with CPU processes rather than CUDA.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing.cap_gain import parse_ffs  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_OBSERVATIONS = (
    REPOSITORY_ROOT
    / "results"
    / "raw"
    / "msabp-krvea-11var-stage2-learned-64-006"
    / "_krvea"
    / "observations.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "processed" / "watermelon_latitude_10deg"
)
DEFAULT_REFERENCES: tuple[tuple[str, Path], ...] = (
    (
        "current_default_reference",
        REPOSITORY_ROOT
        / "results"
        / "raw"
        / "current-default-reference-001"
        / "case_current_default_reference"
        / "Farfield Source [1].ffs",
    ),
    (
        "roblin_wei_2012_reference",
        REPOSITORY_ROOT
        / "results"
        / "raw"
        / "msa-bp-propagation-baseline-001"
        / "Roblin_Wei_Ref.ffs",
    ),
)
DEFAULT_BAND_GHZ = (3.1, 4.8)
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
THETA_EDGES_DEG = np.arange(0.0, 181.0, 10.0, dtype=np.float64)

ARRAY_FILENAME = "watermelon_latitude_gain.npz"
SAMPLE_INDEX_FILENAME = "sample_index.csv"
BROADBAND_FILENAME = "watermelon_latitude_gain_3p1-4p8GHz.csv"
MANIFEST_FILENAME = "manifest.json"

OBJECTIVE_METADATA_COLUMNS = (
    "worst_s11_linear_amplitude",
    "mean_total_efficiency_linear",
    "normalized_substrate_area",
    "cap_realized_gain_linear",
    "cap_realized_gain_dbi",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.casefold().isin(
        {"true", "1", "yes", "y"}
    )


def _declared_farfield_hash(case_directory: Path, ffs_path: Path) -> str:
    manifest_path = case_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        artifacts = manifest.get("artifacts")
        record = (
            artifacts.get("farfield_source")
            if isinstance(artifacts, Mapping)
            else None
        )
        declared = record.get("sha256") if isinstance(record, Mapping) else None
        if declared:
            return str(declared).strip().lower()
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return _sha256(ffs_path)


def load_sample_index(
    observations_path: str | Path = DEFAULT_OBSERVATIONS,
    *,
    include_default_references: bool = True,
) -> pd.DataFrame:
    """Build the deterministic FFS worklist and retain objective metadata."""

    source = Path(observations_path).expanduser().resolve()
    observations = pd.read_csv(source)
    required = {"source", "case_id", "case_directory", "status", "is_penalty"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observation table is missing columns: {sorted(missing)}")
    completed = observations["status"].astype(str).str.casefold().eq("completed")
    usable = observations.loc[
        completed & ~_bool_series(observations["is_penalty"])
    ].copy()
    usable.sort_values(["source", "case_id"], kind="stable", inplace=True)

    rows: list[dict[str, Any]] = []
    if include_default_references:
        for label, reference_path in DEFAULT_REFERENCES:
            ffs_path = reference_path.resolve()
            if not ffs_path.is_file():
                raise FileNotFoundError(f"default reference FFS does not exist: {ffs_path}")
            stat = ffs_path.stat()
            rows.append(
                {
                    "sample_kind": "reference",
                    "source": label,
                    "case_id": label,
                    "case_directory": str(ffs_path.parent),
                    "ffs_path": str(ffs_path),
                    "ffs_sha256": _sha256(ffs_path),
                    "ffs_size_bytes": int(stat.st_size),
                    "ffs_mtime_ns": int(stat.st_mtime_ns),
                }
            )

    for record in usable.to_dict(orient="records"):
        case_directory = Path(str(record["case_directory"])).resolve()
        ffs_path = case_directory / "Farfield Source [1].ffs"
        if not ffs_path.is_file():
            raise FileNotFoundError(f"completed case has no far-field source: {ffs_path}")
        stat = ffs_path.stat()
        row: dict[str, Any] = {
            "sample_kind": "archive",
            "source": str(record["source"]),
            "case_id": str(record["case_id"]),
            "case_directory": str(case_directory),
            "ffs_path": str(ffs_path),
            "ffs_sha256": _declared_farfield_hash(case_directory, ffs_path),
            "ffs_size_bytes": int(stat.st_size),
            "ffs_mtime_ns": int(stat.st_mtime_ns),
        }
        for column in OBJECTIVE_METADATA_COLUMNS:
            if column in record:
                row[column] = float(record[column])
        rows.append(row)

    result = pd.DataFrame.from_records(rows)
    if result.empty:
        raise ValueError("no far-field samples were found")
    result.insert(0, "sample_index", np.arange(len(result), dtype=int))
    keys = result["source"].astype(str) + "/" + result["case_id"].astype(str)
    if keys.duplicated().any():
        duplicates = keys.loc[keys.duplicated(keep=False)].tolist()
        raise ValueError(f"duplicate sample identities: {duplicates[:5]}")
    return result


def latitude_belt_gain(
    ffs: Mapping[str, np.ndarray],
    *,
    theta_edges_deg: np.ndarray = THETA_EDGES_DEG,
) -> dict[str, np.ndarray]:
    """Integrate directivity/gain/realized gain over adjacent theta belts."""

    frequency_hz = np.asarray(ffs["freq"], dtype=np.float64)
    theta_deg = np.asarray(ffs["theta_deg"], dtype=np.float64)
    phi_deg = np.asarray(ffs["phi_deg"], dtype=np.float64)
    p_rad = np.asarray(ffs["p_rad"], dtype=np.float64)
    p_acc = np.asarray(ffs["p_acc"], dtype=np.float64)
    p_stim = np.asarray(ffs["p_stim"], dtype=np.float64)
    e_theta = np.asarray(ffs["E_theta"], dtype=np.complex128)
    e_phi = np.asarray(ffs["E_phi"], dtype=np.complex128)
    edges = np.asarray(theta_edges_deg, dtype=np.float64)

    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("theta edges must be a strictly increasing vector")
    if not np.isclose(edges[0], 0.0) or not np.isclose(edges[-1], 180.0):
        raise ValueError("theta edges must cover [0, 180] degrees")
    if not np.all(np.diff(theta_deg) > 0.0) or not np.all(np.diff(phi_deg) > 0.0):
        raise ValueError("FFS angular grids must be strictly increasing")
    if not np.isclose(theta_deg[0], 0.0) or not np.isclose(theta_deg[-1], 180.0):
        raise ValueError("FFS theta grid must cover [0, 180] degrees")
    if not np.isclose(phi_deg[0], 0.0) or not np.isclose(phi_deg[-1], 360.0):
        raise ValueError("FFS phi grid must cover [0, 360] degrees")
    expected_shape = (len(frequency_hz), len(phi_deg), len(theta_deg))
    if e_theta.shape != expected_shape or e_phi.shape != expected_shape:
        raise ValueError("FFS electric-field arrays do not match their grids")
    if np.any(p_rad <= 0.0) or np.any(p_acc <= 0.0) or np.any(p_stim <= 0.0):
        raise ValueError("FFS power values must be positive")

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    intensity = np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2
    weighted = intensity * np.sin(theta)[None, None, :]
    full_theta_integral = np.trapezoid(weighted, theta, axis=2)
    pattern_power = np.trapezoid(full_theta_integral, phi, axis=1)
    if np.any(pattern_power <= 0.0):
        raise ValueError("FFS pattern integral must be positive")

    belt_count = len(edges) - 1
    directivity = np.empty((len(frequency_hz), belt_count), dtype=np.float64)
    solid_angle = np.empty(belt_count, dtype=np.float64)
    for belt_index, (lower, upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        lower_present = np.any(np.isclose(theta_deg, lower, atol=1e-10, rtol=0.0))
        upper_present = np.any(np.isclose(theta_deg, upper, atol=1e-10, rtol=0.0))
        if not lower_present or not upper_present:
            raise ValueError(
                f"belt boundary [{lower:g}, {upper:g}] is not present on theta grid"
            )
        mask = (theta_deg >= lower - 1e-10) & (theta_deg <= upper + 1e-10)
        theta_integral = np.trapezoid(
            weighted[:, :, mask],
            theta[mask],
            axis=2,
        )
        belt_power = np.trapezoid(theta_integral, phi, axis=1)
        omega = 2.0 * np.pi * (
            np.cos(np.deg2rad(lower)) - np.cos(np.deg2rad(upper))
        )
        solid_angle[belt_index] = omega
        directivity[:, belt_index] = (
            4.0 * np.pi * belt_power / (omega * pattern_power)
        )

    radiation_efficiency = p_rad / p_acc
    total_efficiency = p_rad / p_stim
    gain = directivity * radiation_efficiency[:, None]
    realized_gain = directivity * total_efficiency[:, None]
    for name, values in (
        ("directivity", directivity),
        ("gain", gain),
        ("realized gain", realized_gain),
    ):
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"computed {name} contains invalid values")
    return {
        "frequency_ghz": frequency_hz / 1.0e9,
        "theta_start_deg": edges[:-1].copy(),
        "theta_stop_deg": edges[1:].copy(),
        "solid_angle_sr": solid_angle,
        "directivity_linear": directivity,
        "gain_linear": gain,
        "realized_gain_linear": realized_gain,
        "radiation_efficiency_linear": radiation_efficiency,
        "total_efficiency_linear": total_efficiency,
    }


def _compute_one(task: tuple[int, str]) -> tuple[int, dict[str, np.ndarray]]:
    index, path = task
    try:
        return index, latitude_belt_gain(parse_ffs(Path(path)))
    except Exception as exc:
        raise RuntimeError(f"watermelon computation failed for {path}: {exc}") from exc


def compute_all(
    sample_index: pd.DataFrame,
    *,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, np.ndarray]:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be positive")
    tasks = [
        (int(row.sample_index), str(row.ffs_path))
        for row in sample_index.itertuples(index=False)
    ]
    results: list[dict[str, np.ndarray] | None] = [None] * len(tasks)
    if worker_count == 1:
        for completed, task in enumerate(tasks, start=1):
            index, result = _compute_one(task)
            results[index] = result
            if completed % 25 == 0 or completed == len(tasks):
                print(f"[Watermelon] parsed {completed}/{len(tasks)}")
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(worker_count, len(tasks))
        ) as executor:
            futures = {executor.submit(_compute_one, task): task[0] for task in tasks}
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures),
                start=1,
            ):
                index, result = future.result()
                results[index] = result
                if completed % 25 == 0 or completed == len(tasks):
                    print(f"[Watermelon] parsed {completed}/{len(tasks)}")

    if any(result is None for result in results):
        raise RuntimeError("one or more watermelon results were not returned")
    completed_results = [result for result in results if result is not None]
    reference = completed_results[0]
    shared_names = (
        "frequency_ghz",
        "theta_start_deg",
        "theta_stop_deg",
        "solid_angle_sr",
    )
    for result in completed_results[1:]:
        for name in shared_names:
            if not np.allclose(result[name], reference[name], atol=1e-10, rtol=0.0):
                raise ValueError(f"sample grids disagree for {name}")
    return {
        **{name: reference[name] for name in shared_names},
        "directivity_linear": np.stack(
            [result["directivity_linear"] for result in completed_results]
        ),
        "gain_linear": np.stack(
            [result["gain_linear"] for result in completed_results]
        ),
        "realized_gain_linear": np.stack(
            [result["realized_gain_linear"] for result in completed_results]
        ),
        "radiation_efficiency_linear": np.stack(
            [result["radiation_efficiency_linear"] for result in completed_results]
        ),
        "total_efficiency_linear": np.stack(
            [result["total_efficiency_linear"] for result in completed_results]
        ),
    }


def broadband_table(
    sample_index: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    *,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
) -> pd.DataFrame:
    frequency = np.asarray(arrays["frequency_ghz"], dtype=np.float64)
    low, high = map(float, band_ghz)
    if not 0.0 < low < high:
        raise ValueError("band must satisfy 0 < low < high")
    mask = (frequency >= low - 1e-10) & (frequency <= high + 1e-10)
    if not np.any(mask):
        raise ValueError("requested band contains no FFS frequency samples")

    metric_names = ("directivity_linear", "gain_linear", "realized_gain_linear")
    means = {
        name: np.asarray(arrays[name], dtype=np.float64)[:, mask, :].mean(axis=1)
        for name in metric_names
    }
    mean_rad_eff = np.asarray(
        arrays["radiation_efficiency_linear"], dtype=np.float64
    )[:, mask].mean(axis=1)
    mean_tot_eff = np.asarray(
        arrays["total_efficiency_linear"], dtype=np.float64
    )[:, mask].mean(axis=1)
    rows: list[dict[str, Any]] = []
    for sample_position, sample in sample_index.iterrows():
        for belt_index, (lower, upper) in enumerate(
            zip(
                arrays["theta_start_deg"],
                arrays["theta_stop_deg"],
                strict=True,
            )
        ):
            directivity = float(means["directivity_linear"][sample_position, belt_index])
            gain = float(means["gain_linear"][sample_position, belt_index])
            realized = float(
                means["realized_gain_linear"][sample_position, belt_index]
            )
            rows.append(
                {
                    "sample_index": int(sample["sample_index"]),
                    "sample_kind": str(sample["sample_kind"]),
                    "source": str(sample["source"]),
                    "case_id": str(sample["case_id"]),
                    "band_low_ghz": low,
                    "band_high_ghz": high,
                    "band_sample_count": int(np.count_nonzero(mask)),
                    "theta_start_deg": float(lower),
                    "theta_stop_deg": float(upper),
                    "latitude_north_deg": 90.0 - float(lower),
                    "latitude_south_deg": 90.0 - float(upper),
                    "solid_angle_sr": float(arrays["solid_angle_sr"][belt_index]),
                    "directivity_linear": directivity,
                    "directivity_dbi": 10.0 * math.log10(directivity),
                    "gain_linear": gain,
                    "gain_dbi": 10.0 * math.log10(gain),
                    "realized_gain_linear": realized,
                    "realized_gain_dbi": 10.0 * math.log10(realized),
                    "mean_radiation_efficiency_linear": float(
                        mean_rad_eff[sample_position]
                    ),
                    "mean_total_efficiency_linear": float(
                        mean_tot_eff[sample_position]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _ensure_writable(destinations: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "watermelon outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )


def _write_frame_atomic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_npz_atomic(arrays: Mapping[str, np.ndarray], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare_dataset(
    observations_path: str | Path = DEFAULT_OBSERVATIONS,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
    workers: int = DEFAULT_WORKERS,
    include_default_references: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_directory).expanduser().resolve()
    array_path = output / ARRAY_FILENAME
    index_path = output / SAMPLE_INDEX_FILENAME
    broadband_path = output / BROADBAND_FILENAME
    manifest_path = output / MANIFEST_FILENAME
    _ensure_writable(
        (array_path, index_path, broadband_path, manifest_path),
        overwrite=overwrite,
    )

    sample_index = load_sample_index(
        observations_path,
        include_default_references=include_default_references,
    )
    archive_count = int((sample_index["sample_kind"] == "archive").sum())
    reference_count = int((sample_index["sample_kind"] == "reference").sum())
    print(
        f"[Watermelon] samples={len(sample_index)} "
        f"(archive={archive_count}, references={reference_count}), "
        f"workers={workers}"
    )
    arrays = compute_all(sample_index, workers=workers)
    broadband = broadband_table(sample_index, arrays, band_ghz=band_ghz)

    _write_frame_atomic(sample_index, index_path)
    _write_npz_atomic(arrays, array_path)
    _write_frame_atomic(broadband, broadband_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "source_observations": str(Path(observations_path).expanduser().resolve()),
        "source_observations_sha256": _sha256(observations_path),
        "sample_count": int(len(sample_index)),
        "archive_sample_count": archive_count,
        "reference_sample_count": reference_count,
        "frequency_count": int(len(arrays["frequency_ghz"])),
        "frequency_ghz": np.asarray(arrays["frequency_ghz"]).tolist(),
        "theta_edges_deg": THETA_EDGES_DEG.tolist(),
        "belt_count": int(len(THETA_EDGES_DEG) - 1),
        "band_ghz": [float(band_ghz[0]), float(band_ghz[1])],
        "band_frequency_aggregation": "arithmetic mean in linear power",
        "angular_definition": {
            "coordinate": "CST theta, 0 deg north pole, 180 deg south pole",
            "belts": "theta=[10k, 10(k+1)] deg for k=0..17",
            "integration": "trapezoidal integral of U*sin(theta) over theta and phi",
            "primary_metric": "realized_gain_linear",
            "realized_gain": "belt-average directivity times total efficiency",
        },
        "execution": {
            "backend": "cpu_process_pool",
            "workers": int(workers),
            "cuda_used": False,
            "cuda_note": "FFS text parsing is the dominant cost",
        },
        "array_shapes": {
            name: list(np.asarray(values).shape) for name, values in arrays.items()
        },
        "outputs": {
            ARRAY_FILENAME: {"sha256": _sha256(array_path)},
            SAMPLE_INDEX_FILENAME: {"sha256": _sha256(index_path)},
            BROADBAND_FILENAME: {"sha256": _sha256(broadband_path)},
        },
    }
    _write_json_atomic(manifest, manifest_path)
    print(f"[Watermelon] arrays: {array_path}")
    print(f"[Watermelon] sample index: {index_path}")
    print(f"[Watermelon] broadband table: {broadband_path}")
    print(f"[Watermelon] manifest: {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--band", nargs=2, type=float, default=DEFAULT_BAND_GHZ)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_dataset(
        args.observations,
        args.output_dir,
        band_ghz=(float(args.band[0]), float(args.band[1])),
        workers=args.workers,
        include_default_references=not args.archive_only,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
