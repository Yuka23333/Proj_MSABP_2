"""Extract on-body radiation proxies from CST far-field source files.

Each CST ``.ffs`` file may contain many frequencies. The output therefore has
one row per input file and frequency; no implicit broadband aggregation is
applied. All angular integrals use the physical solid-angle measure
``sin(theta) dtheta dphi`` on CST's exported regular grid.

The proxy definitions intentionally retain their original names. In these
names, "hor" means the horizon/on-body plane, not horizontal polarization:

- ``P_hor``: theta-polarized pattern-power share over
  ``theta_h <= theta <= 90 deg``;
- ``G_theta_hor``: mean theta-polarized, pattern-normalized intensity at
  ``theta = 90 deg``;
- ``chi_TM``: theta-polarized fraction at ``theta = 90 deg``;
- ``P_cap``: total pattern-power share over ``0 <= theta <= theta_cap``;
- ``G_cap_max``: maximum pattern-normalized total intensity in that cap.

``G_theta_hor`` and ``G_cap_max`` are normalized by the integrated radiation
pattern, exactly as in the original script. They are therefore
directivity-like pattern proxies rather than efficiency-weighted gain.

At theta=90 deg, E_theta is normal to the board/body (vertical polarization)
and E_phi is parallel to it (horizontal polarization) for every phi.
G_theta_hor and chi_TM integrate over all phi; they are not forward-endfire
metrics. The explicit G_endfire_phi90_* columns evaluate the project forward
direction at theta=90 deg, phi=90 deg. Inside a finite theta belt below
90 deg, E_theta remains the spherical theta component and is not exactly
vertical until the theta=90 deg boundary.

CST coordinates follow ``theta = 0 deg`` for broadside/off-body normal and
``theta = 90 deg`` for the on-body plane.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing.cap_gain import parse_ffs  # noqa: E402


DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT / "results" / "processed" / "onbody_proxies" / "onbody_proxies.csv"
)
DEFAULT_THETA_H_DEG = 70.0
DEFAULT_THETA_CAP_DEG = 40.0
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)
FIXED_THETA_H80_DEG = 80.0
HORIZON_THETA_DEG = 90.0
FORWARD_ENDFIRE_PHI_DEG = 90.0
GRID_TOLERANCE_DEG = 1.0e-10


def _require_vector(
    ffs: Mapping[str, Any],
    name: str,
    *,
    length: int | None = None,
) -> np.ndarray:
    values = np.asarray(ffs[name], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"FFS {name} must be a one-dimensional vector")
    if length is not None and len(values) != length:
        raise ValueError(f"FFS {name} length does not match the frequency grid")
    if not np.isfinite(values).all():
        raise ValueError(f"FFS {name} contains non-finite values")
    return values


def _validate_angle_grid(theta_deg: np.ndarray, phi_deg: np.ndarray) -> None:
    if len(theta_deg) < 2 or len(phi_deg) < 2:
        raise ValueError("FFS angular grids must each contain at least two samples")
    if not np.all(np.diff(theta_deg) > 0.0):
        raise ValueError("FFS theta grid must be strictly increasing")
    if not np.all(np.diff(phi_deg) > 0.0):
        raise ValueError("FFS phi grid must be strictly increasing")
    if not (np.isclose(theta_deg[0], 0.0) and np.isclose(theta_deg[-1], 180.0)):
        raise ValueError("FFS theta grid must cover [0, 180] degrees")
    if not (np.isclose(phi_deg[0], 0.0) and np.isclose(phi_deg[-1], 360.0)):
        raise ValueError("FFS phi grid must cover [0, 360] degrees")


def _theta_interval_mask(
    theta_deg: np.ndarray,
    lower_deg: float,
    upper_deg: float,
    *,
    label: str,
) -> np.ndarray:
    if not 0.0 <= lower_deg < upper_deg <= 180.0:
        raise ValueError(f"{label} must satisfy 0 <= lower < upper <= 180 degrees")
    lower_present = np.any(
        np.isclose(theta_deg, lower_deg, atol=GRID_TOLERANCE_DEG, rtol=0.0)
    )
    upper_present = np.any(
        np.isclose(theta_deg, upper_deg, atol=GRID_TOLERANCE_DEG, rtol=0.0)
    )
    if not lower_present or not upper_present:
        raise ValueError(
            f"{label} boundaries [{lower_deg:g}, {upper_deg:g}] degrees "
            "must be present on the exported theta grid"
        )
    return (theta_deg >= lower_deg - GRID_TOLERANCE_DEG) & (
        theta_deg <= upper_deg + GRID_TOLERANCE_DEG
    )


def _theta_index(theta_deg: np.ndarray, target_deg: float) -> int:
    matches = np.flatnonzero(
        np.isclose(theta_deg, target_deg, atol=GRID_TOLERANCE_DEG, rtol=0.0)
    )
    if len(matches) != 1:
        raise ValueError(
            f"theta={target_deg:g} degrees must occur exactly once on the grid"
        )
    return int(matches[0])


def _phi_index(phi_deg: np.ndarray, target_deg: float) -> int:
    matches = np.flatnonzero(
        np.isclose(phi_deg, target_deg, atol=GRID_TOLERANCE_DEG, rtol=0.0)
    )
    if len(matches) != 1:
        raise ValueError(
            f"phi={target_deg:g} degrees must occur exactly once on the grid"
        )
    return int(matches[0])


def _solid_angle_integral(
    values: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    *,
    theta_mask: np.ndarray | None = None,
) -> np.ndarray:
    selected_theta = theta_rad
    selected_values = values
    if theta_mask is not None:
        selected_theta = theta_rad[theta_mask]
        selected_values = values[:, :, theta_mask]
    theta_integral = np.trapezoid(
        selected_values * np.sin(selected_theta)[None, None, :],
        selected_theta,
        axis=2,
    )
    return np.trapezoid(theta_integral, phi_rad, axis=1)


def _to_db(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, float("-inf"), dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    result[valid] = 10.0 * np.log10(values[valid])
    result[np.isnan(values)] = float("nan")
    return result


def compute_onbody_proxies(
    ffs: Mapping[str, Any],
    *,
    source_path: str | Path = "<memory>",
    theta_h_deg: float = DEFAULT_THETA_H_DEG,
    theta_cap_deg: float = DEFAULT_THETA_CAP_DEG,
) -> pd.DataFrame:
    """Compute the original on-body proxies at every exported frequency."""

    frequency_hz = _require_vector(ffs, "freq")
    frequency_count = len(frequency_hz)
    if frequency_count == 0:
        raise ValueError("FFS frequency grid is empty")
    theta_deg = _require_vector(ffs, "theta_deg")
    phi_deg = _require_vector(ffs, "phi_deg")
    p_rad = _require_vector(ffs, "p_rad", length=frequency_count)
    p_acc = _require_vector(ffs, "p_acc", length=frequency_count)
    p_stim = _require_vector(ffs, "p_stim", length=frequency_count)
    _validate_angle_grid(theta_deg, phi_deg)

    e_theta = np.asarray(ffs["E_theta"], dtype=np.complex128)
    e_phi = np.asarray(ffs["E_phi"], dtype=np.complex128)
    expected_shape = (frequency_count, len(phi_deg), len(theta_deg))
    if e_theta.shape != expected_shape or e_phi.shape != expected_shape:
        raise ValueError(
            "FFS electric-field arrays do not match frequency and angular grids"
        )
    if not (np.isfinite(e_theta).all() and np.isfinite(e_phi).all()):
        raise ValueError("FFS electric-field arrays contain non-finite values")

    theta_h = float(theta_h_deg)
    theta_cap = float(theta_cap_deg)
    horizon_mask = _theta_interval_mask(
        theta_deg,
        theta_h,
        HORIZON_THETA_DEG,
        label="horizon belt",
    )
    horizon80_mask = _theta_interval_mask(
        theta_deg,
        FIXED_THETA_H80_DEG,
        HORIZON_THETA_DEG,
        label="fixed 80-degree horizon belt",
    )
    cap_mask = _theta_interval_mask(
        theta_deg,
        0.0,
        theta_cap,
        label="broadside cap",
    )
    horizon_index = _theta_index(theta_deg, HORIZON_THETA_DEG)
    forward_endfire_index = _phi_index(phi_deg, FORWARD_ENDFIRE_PHI_DEG)

    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    u_theta = np.abs(e_theta) ** 2
    u_phi = np.abs(e_phi) ** 2
    u_total = u_theta + u_phi
    pattern_power = _solid_angle_integral(u_total, theta_rad, phi_rad)
    if not np.isfinite(pattern_power).all() or np.any(pattern_power <= 0.0):
        raise ValueError("FFS pattern integral must be finite and positive")

    p_hor = (
        _solid_angle_integral(u_theta, theta_rad, phi_rad, theta_mask=horizon_mask)
        / pattern_power
    )
    p_hor80 = (
        _solid_angle_integral(u_theta, theta_rad, phi_rad, theta_mask=horizon80_mask)
        / pattern_power
    )
    p_cap = (
        _solid_angle_integral(u_total, theta_rad, phi_rad, theta_mask=cap_mask)
        / pattern_power
    )

    horizon_theta = u_theta[:, :, horizon_index]
    horizon_total = u_total[:, :, horizon_index]
    phi_span = phi_rad[-1] - phi_rad[0]
    mean_horizon_theta = np.trapezoid(horizon_theta, phi_rad, axis=1) / phi_span
    horizon_theta_power = np.trapezoid(horizon_theta, phi_rad, axis=1)
    horizon_total_power = np.trapezoid(horizon_total, phi_rad, axis=1)
    g_theta_hor = 4.0 * math.pi * mean_horizon_theta / pattern_power
    g_theta_hor_min = 4.0 * math.pi * np.min(horizon_theta, axis=1) / pattern_power
    chi_tm = np.divide(
        horizon_theta_power,
        horizon_total_power,
        out=np.full(frequency_count, float("nan"), dtype=np.float64),
        where=horizon_total_power > 0.0,
    )
    endfire_vertical_intensity = u_theta[
        :,
        forward_endfire_index,
        horizon_index,
    ]
    endfire_horizontal_intensity = u_phi[
        :,
        forward_endfire_index,
        horizon_index,
    ]
    endfire_total_intensity = endfire_vertical_intensity + endfire_horizontal_intensity
    g_endfire_vertical = 4.0 * math.pi * endfire_vertical_intensity / pattern_power
    g_endfire_horizontal = 4.0 * math.pi * endfire_horizontal_intensity / pattern_power
    g_endfire_total = 4.0 * math.pi * endfire_total_intensity / pattern_power
    chi_vertical_endfire = np.divide(
        endfire_vertical_intensity,
        endfire_total_intensity,
        out=np.full(frequency_count, float("nan"), dtype=np.float64),
        where=endfire_total_intensity > 0.0,
    )
    g_cap_max = (
        4.0 * math.pi * np.max(u_total[:, :, cap_mask], axis=(1, 2)) / pattern_power
    )

    theta_component_power = _solid_angle_integral(u_theta, theta_rad, phi_rad)
    theta_moment = _solid_angle_integral(
        u_theta * theta_deg[None, None, :],
        theta_rad,
        phi_rad,
    )
    theta_centroid = np.divide(
        theta_moment,
        theta_component_power,
        out=np.full(frequency_count, float("nan"), dtype=np.float64),
        where=theta_component_power > 0.0,
    )
    radiation_efficiency = np.divide(
        p_rad,
        p_acc,
        out=np.full(frequency_count, float("nan"), dtype=np.float64),
        where=p_acc != 0.0,
    )
    total_efficiency = np.divide(
        p_rad,
        p_stim,
        out=np.full(frequency_count, float("nan"), dtype=np.float64),
        where=p_stim != 0.0,
    )

    path = Path(source_path)
    sample_count = len(theta_deg) * len(phi_deg)
    return pd.DataFrame(
        {
            "file": [path.name] * frequency_count,
            "path": [str(path)] * frequency_count,
            "freq_hz": frequency_hz,
            "freq_ghz": frequency_hz / 1.0e9,
            "n_theta": [len(theta_deg)] * frequency_count,
            "n_phi": [len(phi_deg)] * frequency_count,
            "n_samples": [sample_count] * frequency_count,
            "theta_h_deg": [theta_h] * frequency_count,
            "theta_cap_deg": [theta_cap] * frequency_count,
            "forward_endfire_phi_deg": [FORWARD_ENDFIRE_PHI_DEG] * frequency_count,
            "P_hor": p_hor,
            "P_hor80": p_hor80,
            "P_cap": p_cap,
            "theta_centroid_deg": theta_centroid,
            "G_theta_hor": g_theta_hor,
            "G_theta_hor_dBi": _to_db(g_theta_hor),
            "G_theta_hor_min": g_theta_hor_min,
            "G_theta_hor_min_dBi": _to_db(g_theta_hor_min),
            "chi_TM": chi_tm,
            "G_endfire_phi90_vertical": g_endfire_vertical,
            "G_endfire_phi90_vertical_dBi": _to_db(g_endfire_vertical),
            "G_endfire_phi90_horizontal": g_endfire_horizontal,
            "G_endfire_phi90_horizontal_dBi": _to_db(g_endfire_horizontal),
            "G_endfire_phi90_total": g_endfire_total,
            "G_endfire_phi90_total_dBi": _to_db(g_endfire_total),
            "chi_vertical_endfire_phi90": chi_vertical_endfire,
            "G_cap_max": g_cap_max,
            "G_cap_max_dBi": _to_db(g_cap_max),
            "P_rad_W": p_rad,
            "P_acc_W": p_acc,
            "P_stim_W": p_stim,
            "Rad_Eff_from_ffs": radiation_efficiency,
            "Tot_Eff_from_ffs": total_efficiency,
        }
    )


def collect_paths(inputs: Sequence[str | Path]) -> list[Path]:
    """Resolve unique FFS inputs in deterministic order."""

    candidates: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            candidates.extend(
                sorted(
                    entry
                    for entry in path.rglob("*")
                    if entry.is_file() and entry.suffix.casefold() == ".ffs"
                )
            )
        elif path.is_file():
            if path.suffix.casefold() != ".ffs":
                raise ValueError(f"input file is not an FFS file: {path}")
            candidates.append(path)
        else:
            raise FileNotFoundError(path)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        identity = os.path.normcase(str(resolved))
        if identity not in seen:
            seen.add(identity)
            unique.append(resolved)
    return unique


def compute_all(
    paths: Sequence[Path],
    *,
    theta_h_deg: float = DEFAULT_THETA_H_DEG,
    theta_cap_deg: float = DEFAULT_THETA_CAP_DEG,
    fail_fast: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> tuple[pd.DataFrame, list[tuple[Path, str]]]:
    """Compute independent FFS files in worker processes.

    The returned rows and errors retain input order even though parallel jobs
    may finish out of order. Set ``workers=1`` for the serial reference path.
    """

    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be positive")
    normalized_paths = [Path(path) for path in paths]
    if not normalized_paths:
        raise ValueError("no FFS input paths were provided")

    tasks = [
        (index, str(path), float(theta_h_deg), float(theta_cap_deg))
        for index, path in enumerate(normalized_paths)
    ]
    tables: list[pd.DataFrame | None] = [None] * len(tasks)
    error_messages: list[str | None] = [None] * len(tasks)

    def retain(result: tuple[int, pd.DataFrame | None, str | None]) -> None:
        index, table, error_message = result
        if error_message is not None:
            if fail_fast:
                raise RuntimeError(
                    "on-body proxy computation failed for "
                    f"{normalized_paths[index]}: {error_message}"
                )
            error_messages[index] = error_message
            return
        if table is None:
            raise RuntimeError(
                f"worker returned neither data nor an error for {normalized_paths[index]}"
            )
        tables[index] = table

    if worker_count == 1:
        for completed, task in enumerate(tasks, start=1):
            retain(_compute_one(task))
            _print_progress(completed, len(tasks))
    else:
        max_workers = min(worker_count, len(tasks))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = {executor.submit(_compute_one, task): task[0] for task in tasks}
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures),
                start=1,
            ):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = (index, None, f"{type(exc).__name__}: {exc}")
                retain(result)
                _print_progress(completed, len(tasks))

    errors = [
        (path, error_message)
        for path, error_message in zip(
            normalized_paths,
            error_messages,
            strict=True,
        )
        if error_message is not None
    ]
    for path, error_message in errors:
        print(
            f"[OnBodyProxies] skipped {path}: {error_message}",
            file=sys.stderr,
        )
    successful_tables = [table for table in tables if table is not None]
    if not successful_tables:
        raise ValueError("no FFS files were parsed successfully")
    return pd.concat(successful_tables, ignore_index=True), errors


def _compute_one(
    task: tuple[int, str, float, float],
) -> tuple[int, pd.DataFrame | None, str | None]:
    """Parse and reduce one FFS file; kept top-level for Windows workers."""

    index, path_text, theta_h_deg, theta_cap_deg = task
    path = Path(path_text)
    try:
        table = compute_onbody_proxies(
            parse_ffs(path),
            source_path=path,
            theta_h_deg=theta_h_deg,
            theta_cap_deg=theta_cap_deg,
        )
    except Exception as exc:
        return index, None, f"{type(exc).__name__}: {exc}"
    return index, table, None


def _print_progress(completed: int, total: int) -> None:
    if completed % 25 == 0 or completed == total:
        print(f"[OnBodyProxies] processed {completed}/{total}")


def _write_csv_atomic(frame: pd.DataFrame, path: Path, *, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite to replace it: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _print_distribution(frame: pd.DataFrame) -> None:
    for column in ("P_hor", "P_cap", "chi_TM"):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        p10, p50, p90 = values.quantile([0.1, 0.5, 0.9]).tolist()
        print(
            f"[OnBodyProxies] {column} p10/p50/p90 = {p10:.3f} / {p50:.3f} / {p90:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="FFS files or directories")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--theta-h",
        type=float,
        default=DEFAULT_THETA_H_DEG,
        help="horizon-belt start in degrees",
    )
    parser.add_argument(
        "--theta-cap",
        type=float,
        default=DEFAULT_THETA_CAP_DEG,
        help="broadside-cap limit in degrees",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "independent FFS worker processes "
            f"(default: {DEFAULT_WORKERS}; use 1 for serial execution)"
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = collect_paths(args.inputs)
    if not paths:
        print("[OnBodyProxies] no FFS files found", file=sys.stderr)
        return 2
    table, errors = compute_all(
        paths,
        theta_h_deg=args.theta_h,
        theta_cap_deg=args.theta_cap,
        fail_fast=args.fail_fast,
        workers=args.workers,
    )
    output_path = _write_csv_atomic(
        table,
        args.output.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    print(
        f"[OnBodyProxies] wrote {len(table)} frequency rows from "
        f"{len(paths) - len(errors)} files -> {output_path}"
    )
    if errors:
        print(f"[OnBodyProxies] skipped files: {len(errors)}")
    _print_distribution(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
