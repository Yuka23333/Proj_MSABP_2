"""Find a compact spherical far-field ROI that tracks measured link quality.

The seed search is deliberately narrow: phi=90 deg, theta=60..120 deg, and
frequency=3.6..4.2 GHz. After selecting the best point, every candidate ROI
must contain that seed and may expand in four controls: theta upward, theta
downward, phi left/right together, and frequency outward together.

All ROI averages are taken in linear scale. Angular samples use spherical-cell
solid-angle weights. Correlation remains descriptive until it is confirmed on
an independent propagation batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_TENSOR = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "link_ffs_tensor_13_3p1-4p8GHz.npz"
)
DEFAULT_LINK_TABLE = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "propagation_s21_12_medoids_plus_roblin_wei_reference"
    / "metrics"
    / "september_s21_and_ebn0_lambda0p5.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "processed" / "onbody_proxies" / "spherical_roi_13"
)
REFERENCE_CASE_NAME = "roblin_wei_2012_reference"
COMPONENTS = ("theta", "phi", "total")
FIELD_FAMILIES = ("directivity", "radiation_gain", "realized_gain", "raw_intensity")
GRID_TOLERANCE = 1.0e-10
DEFAULT_SEED_THETA = (60.0, 120.0)
DEFAULT_SEED_FREQUENCY = (3.6, 4.2)
DEFAULT_SEED_PHI_DEG = 90.0
DEFAULT_MIN_COMPONENT_FRACTION = 1.0e-8
TOP_REGION_COUNT = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
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


def spherical_theta_cell_weights(theta_deg: np.ndarray) -> np.ndarray:
    """Return exact solid-angle theta-band weights for grid-cell centres."""

    theta = np.asarray(theta_deg, dtype=np.float64)
    if theta.ndim != 1 or len(theta) < 2 or not np.all(np.diff(theta) > 0.0):
        raise ValueError("theta grid must be a strictly increasing vector")
    if not (np.isclose(theta[0], 0.0) and np.isclose(theta[-1], 180.0)):
        raise ValueError("theta grid must span 0..180 degrees")
    edges = np.empty(len(theta) + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 180.0
    edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    edges_rad = np.deg2rad(edges)
    weights = np.cos(edges_rad[:-1]) - np.cos(edges_rad[1:])
    if not np.all(weights > 0.0) or not np.isclose(weights.sum(), 2.0):
        raise ValueError("invalid theta-cell solid-angle weights")
    return weights


def symmetric_phi_mask(
    phi_deg: np.ndarray,
    center_deg: float,
    half_width_deg: float,
) -> np.ndarray:
    """Select the periodic phi interval centred on center_deg."""

    phi = np.asarray(phi_deg, dtype=np.float64)
    distance = np.abs((phi - center_deg + 180.0) % 360.0 - 180.0)
    return distance <= half_width_deg + GRID_TOLERANCE


def _grid_index(grid: np.ndarray, value: float, label: str) -> int:
    matches = np.flatnonzero(
        np.isclose(grid, float(value), atol=GRID_TOLERANCE, rtol=0.0)
    )
    if len(matches) != 1:
        raise ValueError(f"{label}={value:g} must occur exactly once on the grid")
    return int(matches[0])


def _interval_indices(
    grid: np.ndarray,
    bounds: tuple[float, float],
    label: str,
) -> np.ndarray:
    lower, upper = (float(value) for value in bounds)
    if lower > upper:
        raise ValueError(f"{label} lower bound exceeds upper bound")
    selected = np.flatnonzero(
        (grid >= lower - GRID_TOLERANCE) & (grid <= upper + GRID_TOLERANCE)
    )
    if len(selected) == 0:
        raise ValueError(f"{label} interval contains no grid points")
    if not np.isclose(grid[selected[0]], lower, atol=GRID_TOLERANCE, rtol=0.0):
        raise ValueError(f"{label} lower bound {lower:g} is absent from the grid")
    if not np.isclose(grid[selected[-1]], upper, atol=GRID_TOLERANCE, rtol=0.0):
        raise ValueError(f"{label} upper bound {upper:g} is absent from the grid")
    return selected


def _load_tensor(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "case_name",
        "frequency_ghz",
        "theta_deg",
        "phi_deg",
        "E_theta",
        "E_phi",
        "pattern_power",
        "radiation_efficiency",
        "total_efficiency",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"FFS tensor lacks arrays: {missing}")
    expected = (
        len(arrays["case_name"]),
        len(arrays["frequency_ghz"]),
        len(arrays["phi_deg"]),
        len(arrays["theta_deg"]),
    )
    if arrays["E_theta"].shape != expected or arrays["E_phi"].shape != expected:
        raise ValueError("complex field arrays disagree with stored grids")
    return arrays


def _load_targets(path: Path, case_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {
        "case_name",
        "s21_power_september",
        "ebn0_ber_v2_at_target_db",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"link table lacks columns: {missing}")
    if frame["case_name"].duplicated().any():
        raise ValueError("link table case_name must be unique")
    indexed = frame.set_index("case_name").reindex(case_names.astype(str))
    if indexed[list(required - {"case_name"})].isna().any().any():
        raise ValueError("link table does not cover every tensor case")
    s21_power = indexed["s21_power_september"].to_numpy(dtype=np.float64)
    ebn0_good = -indexed["ebn0_ber_v2_at_target_db"].to_numpy(dtype=np.float64)
    if not np.isfinite(s21_power).all() or not np.isfinite(ebn0_good).all():
        raise ValueError("link targets must be finite")
    return s21_power, ebn0_good


def _component_intensities(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    theta = np.abs(np.asarray(arrays["E_theta"], dtype=np.complex128)) ** 2
    phi = np.abs(np.asarray(arrays["E_phi"], dtype=np.complex128)) ** 2
    return {"theta": theta, "phi": phi, "total": theta + phi}


def make_field_maps(
    arrays: Mapping[str, np.ndarray],
    field_family: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return selected field maps and their raw component fractions."""

    if field_family not in FIELD_FAMILIES:
        raise ValueError(f"unknown field family: {field_family}")
    intensity = _component_intensities(arrays)
    if field_family == "raw_intensity":
        maps = intensity
    else:
        pattern_power = np.asarray(arrays["pattern_power"], dtype=np.float64)
        scale = 4.0 * math.pi / pattern_power
        if field_family == "radiation_gain":
            scale = scale * np.asarray(arrays["radiation_efficiency"], dtype=np.float64)
        elif field_family == "realized_gain":
            scale = scale * np.asarray(arrays["total_efficiency"], dtype=np.float64)
        maps = {
            component: values * scale[:, :, None, None]
            for component, values in intensity.items()
        }
    fractions = {
        component: np.divide(
            values,
            intensity["total"],
            out=np.zeros_like(values),
            where=intensity["total"] > 0.0,
        )
        for component, values in intensity.items()
    }
    return maps, fractions


def _standardized_ranks(values: np.ndarray) -> np.ndarray:
    ranked = stats.rankdata(np.asarray(values, dtype=np.float64), axis=0)
    ranked -= ranked.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(ranked, axis=0, keepdims=True)
    return np.divide(
        ranked,
        norm,
        out=np.full_like(ranked, np.nan),
        where=norm > 0.0,
    )


def score_features(
    features: np.ndarray,
    s21_power: np.ndarray,
    ebn0_good: np.ndarray,
    *,
    reference_index: int,
) -> dict[str, np.ndarray]:
    """Score case-by-candidate features against two higher-is-better targets."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != len(s21_power):
        raise ValueError("feature and target case counts disagree")
    ranked = _standardized_ranks(x)
    target_s21 = _standardized_ranks(np.asarray(s21_power)[:, None])[:, 0]
    target_ebn0 = _standardized_ranks(np.asarray(ebn0_good)[:, None])[:, 0]
    rho_s21 = target_s21 @ ranked
    rho_ebn0 = target_ebn0 @ ranked

    keep = np.arange(x.shape[0]) != int(reference_index)
    ranked_without_reference = _standardized_ranks(x[keep])
    target_s21_without_reference = _standardized_ranks(
        np.asarray(s21_power)[keep, None]
    )[:, 0]
    target_ebn0_without_reference = _standardized_ranks(
        np.asarray(ebn0_good)[keep, None]
    )[:, 0]
    rho_s21_without_reference = target_s21_without_reference @ ranked_without_reference
    rho_ebn0_without_reference = (
        target_ebn0_without_reference @ ranked_without_reference
    )
    joint = np.minimum(rho_s21, rho_ebn0)
    joint_without_reference = np.minimum(
        rho_s21_without_reference,
        rho_ebn0_without_reference,
    )
    return {
        "rho_s21_power": rho_s21,
        "rho_negative_ebn0": rho_ebn0,
        "joint_rho": joint,
        "joint_mean_rho": 0.5 * (rho_s21 + rho_ebn0),
        "rho_s21_power_without_reference": rho_s21_without_reference,
        "rho_negative_ebn0_without_reference": rho_ebn0_without_reference,
        "joint_rho_without_reference": joint_without_reference,
        "robust_joint_rho": np.minimum(joint, joint_without_reference),
    }


SCORE_COLUMNS = [
    "robust_joint_rho",
    "joint_rho",
    "joint_mean_rho",
    "rho_s21_power",
    "rho_negative_ebn0",
    "joint_rho_without_reference",
    "rho_s21_power_without_reference",
    "rho_negative_ebn0_without_reference",
]


def _attach_scores(
    metadata: pd.DataFrame,
    features: np.ndarray,
    s21_power: np.ndarray,
    ebn0_good: np.ndarray,
    reference_index: int,
) -> pd.DataFrame:
    result = metadata.copy()
    scores = score_features(
        features,
        s21_power,
        ebn0_good,
        reference_index=reference_index,
    )
    for name, values in scores.items():
        result[name] = values
    return result


def _sort_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "robust_joint_rho",
            "joint_rho",
            "joint_mean_rho",
            "roi_grid_points",
        ],
        ascending=[False, False, False, False],
        kind="stable",
    )


def search_seed(
    maps: Mapping[str, np.ndarray],
    fractions: Mapping[str, np.ndarray],
    frequency_ghz: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    s21_power: np.ndarray,
    ebn0_good: np.ndarray,
    *,
    reference_index: int,
    theta_bounds: tuple[float, float],
    frequency_bounds: tuple[float, float],
    phi_center_deg: float,
    min_component_fraction: float,
) -> pd.DataFrame:
    theta_indices = _interval_indices(theta_deg, theta_bounds, "seed theta")
    frequency_indices = _interval_indices(
        frequency_ghz,
        frequency_bounds,
        "seed frequency",
    )
    phi_index = _grid_index(phi_deg, phi_center_deg, "seed phi")

    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for component in COMPONENTS:
        field_map = maps[component]
        fraction_map = fractions[component]
        for frequency_index in frequency_indices:
            for theta_index in theta_indices:
                component_fraction = float(
                    np.median(
                        fraction_map[
                            :,
                            frequency_index,
                            phi_index,
                            theta_index,
                        ]
                    )
                )
                rows.append(
                    {
                        "component": component,
                        "frequency_index": int(frequency_index),
                        "frequency_ghz": float(frequency_ghz[frequency_index]),
                        "phi_index": phi_index,
                        "phi_deg": float(phi_deg[phi_index]),
                        "theta_index": int(theta_index),
                        "theta_deg": float(theta_deg[theta_index]),
                        "component_fraction_median": component_fraction,
                        "component_fraction_guard_passed": bool(
                            component == "total"
                            or component_fraction >= min_component_fraction
                        ),
                        "roi_grid_points": 1,
                    }
                )
                features.append(field_map[:, frequency_index, phi_index, theta_index])
    table = _attach_scores(
        pd.DataFrame(rows),
        np.column_stack(features),
        s21_power,
        ebn0_good,
        reference_index,
    )
    table.loc[~table["component_fraction_guard_passed"], SCORE_COLUMNS] = np.nan
    return _sort_candidates(table).reset_index(drop=True)


def _frequency_intervals(seed_index: int, count: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for radius in range(max(seed_index, count - 1 - seed_index) + 1):
        interval = (max(0, seed_index - radius), min(count - 1, seed_index + radius))
        if not intervals or interval != intervals[-1]:
            intervals.append(interval)
    return intervals


def _phi_half_widths(phi_deg: np.ndarray) -> np.ndarray:
    differences = np.diff(phi_deg)
    if not np.allclose(differences, differences[0], atol=GRID_TOLERANCE, rtol=0.0):
        raise ValueError("phi grid must be uniform")
    return np.arange(len(phi_deg) // 2 + 1, dtype=np.float64) * differences[0]


def _region_metadata(
    *,
    theta_low: int,
    theta_high: int,
    phi_radius_index: int,
    frequency_radius_index: int,
    phi_mask: np.ndarray,
    frequency_interval: tuple[int, int],
    theta_weights: np.ndarray,
    theta_deg: np.ndarray,
    phi_half_widths: np.ndarray,
    frequency_ghz: np.ndarray,
) -> dict[str, Any]:
    frequency_low, frequency_high = frequency_interval
    phi_step_rad = 2.0 * math.pi / len(phi_mask)
    solid_angle = (
        float(theta_weights[theta_low : theta_high + 1].sum())
        * int(phi_mask.sum())
        * phi_step_rad
    )
    return {
        "theta_low_index": theta_low,
        "theta_high_index": theta_high,
        "theta_min_deg": float(theta_deg[theta_low]),
        "theta_max_deg": float(theta_deg[theta_high]),
        "phi_radius_index": phi_radius_index,
        "phi_half_width_deg": float(phi_half_widths[phi_radius_index]),
        "phi_grid_count": int(phi_mask.sum()),
        "frequency_radius_index": frequency_radius_index,
        "frequency_low_index": frequency_low,
        "frequency_high_index": frequency_high,
        "frequency_min_ghz": float(frequency_ghz[frequency_low]),
        "frequency_max_ghz": float(frequency_ghz[frequency_high]),
        "frequency_grid_count": frequency_high - frequency_low + 1,
        "solid_angle_sr": solid_angle,
        "roi_grid_points": (
            (theta_high - theta_low + 1)
            * int(phi_mask.sum())
            * (frequency_high - frequency_low + 1)
        ),
    }


def region_feature(
    field_map: np.ndarray,
    *,
    theta_low: int,
    theta_high: int,
    phi_mask: np.ndarray,
    frequency_interval: tuple[int, int],
    theta_weights: np.ndarray,
) -> np.ndarray:
    """Average a case x frequency x phi x theta map over one ROI."""

    selected = field_map[
        :,
        frequency_interval[0] : frequency_interval[1] + 1,
        phi_mask,
        theta_low : theta_high + 1,
    ]
    weights = theta_weights[theta_low : theta_high + 1]
    theta_average = np.average(selected, axis=3, weights=weights)
    return theta_average.mean(axis=(1, 2))


def search_global_expansion(
    field_map: np.ndarray,
    frequency_ghz: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    s21_power: np.ndarray,
    ebn0_good: np.ndarray,
    *,
    reference_index: int,
    seed_frequency_index: int,
    seed_theta_index: int,
    phi_center_deg: float,
    top_count: int = TOP_REGION_COUNT,
) -> tuple[pd.DataFrame, pd.Series]:
    """Exhaustively search all four-control expansions containing the seed."""

    theta_weights = spherical_theta_cell_weights(theta_deg)
    phi_half_widths = _phi_half_widths(phi_deg)
    phi_masks = [
        symmetric_phi_mask(phi_deg, phi_center_deg, half_width)
        for half_width in phi_half_widths
    ]
    frequency_intervals = _frequency_intervals(
        seed_frequency_index,
        len(frequency_ghz),
    )
    weighted_map = field_map * theta_weights[None, None, None, :]
    theta_prefix = np.concatenate(
        [
            np.zeros((*weighted_map.shape[:-1], 1), dtype=np.float64),
            np.cumsum(weighted_map, axis=3),
        ],
        axis=3,
    )

    shortlists: list[pd.DataFrame] = []
    examined = 0
    for theta_low in range(seed_theta_index, -1, -1):
        for theta_high in range(seed_theta_index, len(theta_deg)):
            theta_sum = theta_prefix[..., theta_high + 1] - theta_prefix[..., theta_low]
            theta_average = theta_sum / theta_weights[theta_low : theta_high + 1].sum()
            features: list[np.ndarray] = []
            rows: list[dict[str, Any]] = []
            for phi_radius_index, phi_mask in enumerate(phi_masks):
                angular_average = theta_average[:, :, phi_mask].mean(axis=2)
                for frequency_radius_index, interval in enumerate(frequency_intervals):
                    features.append(
                        angular_average[:, interval[0] : interval[1] + 1].mean(axis=1)
                    )
                    rows.append(
                        _region_metadata(
                            theta_low=theta_low,
                            theta_high=theta_high,
                            phi_radius_index=phi_radius_index,
                            frequency_radius_index=frequency_radius_index,
                            phi_mask=phi_mask,
                            frequency_interval=interval,
                            theta_weights=theta_weights,
                            theta_deg=theta_deg,
                            phi_half_widths=phi_half_widths,
                            frequency_ghz=frequency_ghz,
                        )
                    )
            batch = _attach_scores(
                pd.DataFrame(rows),
                np.column_stack(features),
                s21_power,
                ebn0_good,
                reference_index,
            )
            examined += len(batch)
            shortlists.append(_sort_candidates(batch).head(top_count))
    top = (
        _sort_candidates(pd.concat(shortlists, ignore_index=True))
        .head(top_count)
        .reset_index(drop=True)
    )
    top.insert(0, "regions_examined", examined)
    return top, top.iloc[0]


def _score_one_region(
    field_map: np.ndarray,
    state: tuple[int, int, int, int],
    *,
    frequency_intervals: list[tuple[int, int]],
    phi_masks: list[np.ndarray],
    theta_weights: np.ndarray,
    theta_deg: np.ndarray,
    phi_half_widths: np.ndarray,
    frequency_ghz: np.ndarray,
    s21_power: np.ndarray,
    ebn0_good: np.ndarray,
    reference_index: int,
) -> dict[str, Any]:
    theta_low, theta_high, phi_radius, frequency_radius = state
    metadata = _region_metadata(
        theta_low=theta_low,
        theta_high=theta_high,
        phi_radius_index=phi_radius,
        frequency_radius_index=frequency_radius,
        phi_mask=phi_masks[phi_radius],
        frequency_interval=frequency_intervals[frequency_radius],
        theta_weights=theta_weights,
        theta_deg=theta_deg,
        phi_half_widths=phi_half_widths,
        frequency_ghz=frequency_ghz,
    )
    feature = region_feature(
        field_map,
        theta_low=theta_low,
        theta_high=theta_high,
        phi_mask=phi_masks[phi_radius],
        frequency_interval=frequency_intervals[frequency_radius],
        theta_weights=theta_weights,
    )
    scores = score_features(
        feature,
        s21_power,
        ebn0_good,
        reference_index=reference_index,
    )
    metadata.update({name: float(values[0]) for name, values in scores.items()})
    return metadata


def _row_key(row: Mapping[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["robust_joint_rho"]),
        float(row["joint_rho"]),
        float(row["joint_mean_rho"]),
        int(row["roi_grid_points"]),
    )


def trace_expansion(
    field_map: np.ndarray,
    frequency_ghz: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    s21_power: np.ndarray,
    ebn0_good: np.ndarray,
    *,
    reference_index: int,
    seed_frequency_index: int,
    seed_theta_index: int,
    phi_center_deg: float,
    target_state: tuple[int, int, int, int] | None,
) -> pd.DataFrame:
    """Trace greedy improvement or a best-next path to a target ROI."""

    theta_weights = spherical_theta_cell_weights(theta_deg)
    phi_half_widths = _phi_half_widths(phi_deg)
    phi_masks = [
        symmetric_phi_mask(phi_deg, phi_center_deg, half_width)
        for half_width in phi_half_widths
    ]
    frequency_intervals = _frequency_intervals(
        seed_frequency_index,
        len(frequency_ghz),
    )
    state = (seed_theta_index, seed_theta_index, 0, 0)
    rows: list[dict[str, Any]] = []
    current = _score_one_region(
        field_map,
        state,
        frequency_intervals=frequency_intervals,
        phi_masks=phi_masks,
        theta_weights=theta_weights,
        theta_deg=theta_deg,
        phi_half_widths=phi_half_widths,
        frequency_ghz=frequency_ghz,
        s21_power=s21_power,
        ebn0_good=ebn0_good,
        reference_index=reference_index,
    )
    current.update({"step": 0, "move": "seed"})
    rows.append(current)

    while True:
        low, high, phi_radius, frequency_radius = state
        moves: list[tuple[str, tuple[int, int, int, int]]] = []
        if low > 0 and (target_state is None or low > target_state[0]):
            moves.append(("theta_up", (low - 1, high, phi_radius, frequency_radius)))
        if high < len(theta_deg) - 1 and (
            target_state is None or high < target_state[1]
        ):
            moves.append(("theta_down", (low, high + 1, phi_radius, frequency_radius)))
        if phi_radius < len(phi_masks) - 1 and (
            target_state is None or phi_radius < target_state[2]
        ):
            moves.append(
                ("phi_left_right", (low, high, phi_radius + 1, frequency_radius))
            )
        if frequency_radius < len(frequency_intervals) - 1 and (
            target_state is None or frequency_radius < target_state[3]
        ):
            moves.append(("frequency", (low, high, phi_radius, frequency_radius + 1)))
        if not moves:
            break

        candidates: list[dict[str, Any]] = []
        for move, candidate_state in moves:
            candidate = _score_one_region(
                field_map,
                candidate_state,
                frequency_intervals=frequency_intervals,
                phi_masks=phi_masks,
                theta_weights=theta_weights,
                theta_deg=theta_deg,
                phi_half_widths=phi_half_widths,
                frequency_ghz=frequency_ghz,
                s21_power=s21_power,
                ebn0_good=ebn0_good,
                reference_index=reference_index,
            )
            candidate["_state"] = candidate_state
            candidate["move"] = move
            candidates.append(candidate)
        next_row = max(candidates, key=_row_key)
        if target_state is None and _row_key(next_row) <= _row_key(current):
            break
        state = next_row.pop("_state")
        next_row["step"] = len(rows)
        rows.append(next_row)
        current = next_row
    return pd.DataFrame(rows)


def _final_case_table(
    arrays: Mapping[str, np.ndarray],
    field_map: np.ndarray,
    best: Mapping[str, Any],
    s21_power: np.ndarray,
    ebn0_good: np.ndarray,
    *,
    seed_frequency_index: int,
    phi_center_deg: float,
) -> pd.DataFrame:
    frequency = np.asarray(arrays["frequency_ghz"], dtype=np.float64)
    theta = np.asarray(arrays["theta_deg"], dtype=np.float64)
    phi = np.asarray(arrays["phi_deg"], dtype=np.float64)
    frequency_intervals = _frequency_intervals(seed_frequency_index, len(frequency))
    phi_half_widths = _phi_half_widths(phi)
    phi_mask = symmetric_phi_mask(
        phi,
        phi_center_deg,
        phi_half_widths[int(best["phi_radius_index"])],
    )
    feature = region_feature(
        field_map,
        theta_low=int(best["theta_low_index"]),
        theta_high=int(best["theta_high_index"]),
        phi_mask=phi_mask,
        frequency_interval=frequency_intervals[int(best["frequency_radius_index"])],
        theta_weights=spherical_theta_cell_weights(theta),
    )
    return pd.DataFrame(
        {
            "case_name": np.asarray(arrays["case_name"]).astype(str),
            "candidate_rank": np.asarray(arrays["candidate_rank"], dtype=np.int64),
            "roi_metric_linear": feature,
            "s21_power_september": s21_power,
            "negative_ebn0_ber_v2_db": ebn0_good,
            "ebn0_ber_v2_at_target_db": -ebn0_good,
        }
    )


def _fixed_roi_loo_summary(case_metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    feature = case_metrics["roi_metric_linear"].to_numpy(dtype=np.float64)
    targets = {
        "s21_power": case_metrics["s21_power_september"].to_numpy(dtype=np.float64),
        "negative_ebn0": case_metrics["negative_ebn0_ber_v2_db"].to_numpy(
            dtype=np.float64
        ),
    }
    summary: dict[str, dict[str, float]] = {}
    for name, target in targets.items():
        correlations = np.asarray(
            [
                stats.spearmanr(
                    np.delete(feature, omitted),
                    np.delete(target, omitted),
                ).statistic
                for omitted in range(len(feature))
            ],
            dtype=np.float64,
        )
        summary[name] = {
            "minimum": float(correlations.min()),
            "median": float(np.median(correlations)),
            "maximum": float(correlations.max()),
        }
    return summary


def run_search(
    tensor_path: str | Path = DEFAULT_TENSOR,
    link_table_path: str | Path = DEFAULT_LINK_TABLE,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    field_family: str = "directivity",
    seed_theta_bounds: tuple[float, float] = DEFAULT_SEED_THETA,
    seed_frequency_bounds: tuple[float, float] = DEFAULT_SEED_FREQUENCY,
    seed_phi_deg: float = DEFAULT_SEED_PHI_DEG,
    min_component_fraction: float = DEFAULT_MIN_COMPONENT_FRACTION,
    overwrite: bool = False,
) -> dict[str, Any]:
    tensor_path = Path(tensor_path).expanduser().resolve()
    link_table_path = Path(link_table_path).expanduser().resolve()
    output_directory = Path(output_directory).expanduser().resolve()
    output_paths = {
        "seed_scan": output_directory / "seed_scan.csv",
        "top_regions": output_directory / "top_expanded_regions.csv",
        "greedy_path": output_directory / "greedy_expansion_path.csv",
        "optimal_path": output_directory / "optimal_region_expansion_path.csv",
        "case_metrics": output_directory / "optimal_roi_case_metrics.csv",
        "result": output_directory / "result.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"ROI output exists; pass --overwrite to replace: {existing[0]}"
        )

    arrays = _load_tensor(tensor_path)
    case_names = np.asarray(arrays["case_name"]).astype(str)
    matches = np.flatnonzero(case_names == REFERENCE_CASE_NAME)
    if len(matches) != 1:
        raise ValueError("tensor must contain the Roblin-Wei reference exactly once")
    reference_index = int(matches[0])
    non_reference_count = len(case_names) - 1
    s21_power, ebn0_good = _load_targets(link_table_path, case_names)
    maps, fractions = make_field_maps(arrays, field_family)
    frequency = np.asarray(arrays["frequency_ghz"], dtype=np.float64)
    theta = np.asarray(arrays["theta_deg"], dtype=np.float64)
    phi = np.asarray(arrays["phi_deg"], dtype=np.float64)

    seed_scan = search_seed(
        maps,
        fractions,
        frequency,
        theta,
        phi,
        s21_power,
        ebn0_good,
        reference_index=reference_index,
        theta_bounds=seed_theta_bounds,
        frequency_bounds=seed_frequency_bounds,
        phi_center_deg=seed_phi_deg,
        min_component_fraction=min_component_fraction,
    )
    if seed_scan["robust_joint_rho"].notna().sum() == 0:
        raise RuntimeError("no seed candidate survived the component fraction guard")
    seed = seed_scan.iloc[0]
    component = str(seed["component"])
    print(
        "[SphericalROI] seed "
        f"{component}, f={seed['frequency_ghz']:.1f} GHz, "
        f"phi={seed['phi_deg']:.0f} deg, theta={seed['theta_deg']:.0f} deg, "
        f"robust joint rho={seed['robust_joint_rho']:.4f}"
    )

    top_regions, optimum = search_global_expansion(
        maps[component],
        frequency,
        theta,
        phi,
        s21_power,
        ebn0_good,
        reference_index=reference_index,
        seed_frequency_index=int(seed["frequency_index"]),
        seed_theta_index=int(seed["theta_index"]),
        phi_center_deg=seed_phi_deg,
    )
    target_state = (
        int(optimum["theta_low_index"]),
        int(optimum["theta_high_index"]),
        int(optimum["phi_radius_index"]),
        int(optimum["frequency_radius_index"]),
    )
    common_trace_options = {
        "reference_index": reference_index,
        "seed_frequency_index": int(seed["frequency_index"]),
        "seed_theta_index": int(seed["theta_index"]),
        "phi_center_deg": seed_phi_deg,
    }
    greedy_path = trace_expansion(
        maps[component],
        frequency,
        theta,
        phi,
        s21_power,
        ebn0_good,
        target_state=None,
        **common_trace_options,
    )
    optimal_path = trace_expansion(
        maps[component],
        frequency,
        theta,
        phi,
        s21_power,
        ebn0_good,
        target_state=target_state,
        **common_trace_options,
    )
    case_metrics = _final_case_table(
        arrays,
        maps[component],
        optimum,
        s21_power,
        ebn0_good,
        seed_frequency_index=int(seed["frequency_index"]),
        phi_center_deg=seed_phi_deg,
    )
    fixed_roi_loo = _fixed_roi_loo_summary(case_metrics)

    _write_csv_atomic(seed_scan, output_paths["seed_scan"])
    _write_csv_atomic(top_regions, output_paths["top_regions"])
    _write_csv_atomic(greedy_path, output_paths["greedy_path"])
    _write_csv_atomic(optimal_path, output_paths["optimal_path"])
    _write_csv_atomic(case_metrics, output_paths["case_metrics"])

    payload: dict[str, Any] = {
        "schema": "msabp.spherical_link_roi.v1",
        "status": "exploratory_in_sample",
        "case_count": len(case_names),
        "field_family": field_family,
        "component": component,
        "score_definition": {
            "target_1": "Spearman(ROI metric, S21 power); higher is better",
            "target_2": "Spearman(ROI metric, -Eb/N0); higher is better",
            "joint_rho": "minimum of the two target correlations",
            "robust_joint_rho": (
                f"minimum joint_rho across all {len(case_names)} cases and the "
                f"{non_reference_count} cases excluding the reference"
            ),
            "tie_break": ("joint_rho, joint_mean_rho, then largest equal-ranking ROI"),
        },
        "seed_constraints": {
            "phi_deg": seed_phi_deg,
            "theta_deg_inclusive": list(seed_theta_bounds),
            "frequency_ghz_inclusive": list(seed_frequency_bounds),
            "components": list(COMPONENTS),
            "minimum_median_component_fraction": min_component_fraction,
        },
        "seed": {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in seed.to_dict().items()
        },
        "optimal_roi": {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in optimum.to_dict().items()
        },
        "greedy_local_endpoint": {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in greedy_path.iloc[-1].to_dict().items()
        },
        "fixed_optimal_roi_leave_one_case_out": fixed_roi_loo,
        "inputs": {
            "tensor": {"path": str(tensor_path), "sha256": _sha256(tensor_path)},
            "link_table": {
                "path": str(link_table_path),
                "sha256": _sha256(link_table_path),
            },
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "warning": (
            f"The ROI was selected and scored on the same {len(case_names)} "
            "non-iid cases. Use a later independent propagation batch for "
            "validation before treating it as a physical mechanism or "
            "optimization objective."
        ),
    }
    _write_json_atomic(payload, output_paths["result"])
    print(
        "[SphericalROI] optimum "
        f"theta=[{optimum['theta_min_deg']:.0f},{optimum['theta_max_deg']:.0f}] deg, "
        f"phi=90+/-{optimum['phi_half_width_deg']:.0f} deg, "
        f"f=[{optimum['frequency_min_ghz']:.1f},"
        f"{optimum['frequency_max_ghz']:.1f}] GHz, "
        f"robust joint rho={optimum['robust_joint_rho']:.4f}"
    )
    print(f"[SphericalROI] outputs -> {output_directory}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor", type=Path, default=DEFAULT_TENSOR)
    parser.add_argument("--link-table", type=Path, default=DEFAULT_LINK_TABLE)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--field-family", choices=FIELD_FAMILIES, default="directivity")
    parser.add_argument("--seed-theta", nargs=2, type=float, default=DEFAULT_SEED_THETA)
    parser.add_argument(
        "--seed-frequency",
        nargs=2,
        type=float,
        default=DEFAULT_SEED_FREQUENCY,
    )
    parser.add_argument("--seed-phi", type=float, default=DEFAULT_SEED_PHI_DEG)
    parser.add_argument(
        "--min-component-fraction",
        type=float,
        default=DEFAULT_MIN_COMPONENT_FRACTION,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_search(
        args.tensor,
        args.link_table,
        args.output_directory,
        field_family=args.field_family,
        seed_theta_bounds=(float(args.seed_theta[0]), float(args.seed_theta[1])),
        seed_frequency_bounds=(
            float(args.seed_frequency[0]),
            float(args.seed_frequency[1]),
        ),
        seed_phi_deg=float(args.seed_phi),
        min_component_fraction=float(args.min_component_fraction),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
