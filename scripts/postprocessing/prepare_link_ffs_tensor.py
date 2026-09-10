"""Pack measured-link FFS files into one reusable NumPy tensor archive.

The archive contains complex E-theta/E-phi fields, CST power metadata, the
shared spherical grid, and source hashes. Frequencies are restricted to the
project band before storage. Phi=360 degrees is removed because it duplicates
the periodic phi=0 endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing.cap_gain import parse_ffs  # noqa: E402


DEFAULT_SELECTED_TABLE = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "link_correlation_13"
    / "selected_13_proxy_frequency_table.csv"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "onbody_proxies"
    / "link_ffs_tensor_13_3p1-4p8GHz.npz"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(".manifest.json")
DEFAULT_BAND_GHZ = (3.1, 4.8)
FREQUENCY_TOLERANCE_GHZ = 1.0e-10
RANK_PATTERN = re.compile(r"(?:^|_)rank_(\d+)(?:_|$)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> Path:
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


def _write_npz_atomic(arrays: dict[str, np.ndarray], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _pattern_power(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    intensity = np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2
    theta_integral = np.trapezoid(
        intensity * np.sin(theta_rad)[None, None, :],
        theta_rad,
        axis=2,
    )
    result = np.trapezoid(theta_integral, phi_rad, axis=1)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("FFS pattern power must be finite and positive")
    return result


def prepare_tensor(
    selected_table_path: str | Path = DEFAULT_SELECTED_TABLE,
    output_path: str | Path = DEFAULT_OUTPUT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
    overwrite: bool = False,
) -> dict[str, Any]:
    selected_path = Path(selected_table_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"tensor output exists; pass --overwrite to replace: {existing[0]}"
        )
    lower_ghz, upper_ghz = (float(value) for value in band_ghz)
    if not 0.0 < lower_ghz <= upper_ghz:
        raise ValueError("band must satisfy 0 < lower <= upper GHz")

    selected = pd.read_csv(selected_path)
    required = {"link_case_name", "ffs_path", "ffs_sha256"}
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"selected table lacks columns: {missing}")
    cases = selected.drop_duplicates("link_case_name").sort_values(
        "link_case_name", kind="stable"
    )
    if len(cases) < 2:
        raise ValueError("selected table must contain at least two unique link cases")
    case_count = len(cases)

    case_names: list[str] = []
    candidate_ranks: list[int] = []
    source_paths: list[str] = []
    source_hashes: list[str] = []
    e_theta_cases: list[np.ndarray] = []
    e_phi_cases: list[np.ndarray] = []
    p_rad_cases: list[np.ndarray] = []
    p_acc_cases: list[np.ndarray] = []
    p_stim_cases: list[np.ndarray] = []
    pattern_power_cases: list[np.ndarray] = []
    reference_grids: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for completed, row in enumerate(cases.itertuples(index=False), start=1):
        source = Path(row.ffs_path).expanduser().resolve()
        source_hash = _sha256(source)
        if source_hash.casefold() != str(row.ffs_sha256).casefold():
            raise ValueError(f"FFS hash mismatch: {source}")
        ffs = parse_ffs(source)
        frequency_hz = np.asarray(ffs["freq"], dtype=np.float64)
        theta_deg = np.asarray(ffs["theta_deg"], dtype=np.float64)
        phi_deg_full = np.asarray(ffs["phi_deg"], dtype=np.float64)
        if not (
            np.isclose(phi_deg_full[0], 0.0) and np.isclose(phi_deg_full[-1], 360.0)
        ):
            raise ValueError(f"FFS phi grid does not include 0/360 endpoints: {source}")
        frequency_ghz = frequency_hz / 1.0e9
        frequency_mask = (frequency_ghz >= lower_ghz - FREQUENCY_TOLERANCE_GHZ) & (
            frequency_ghz <= upper_ghz + FREQUENCY_TOLERANCE_GHZ
        )
        if not np.any(frequency_mask):
            raise ValueError(f"FFS contains no selected-band frequencies: {source}")

        e_theta_full = np.asarray(ffs["E_theta"], dtype=np.complex128)
        e_phi_full = np.asarray(ffs["E_phi"], dtype=np.complex128)
        expected = (len(frequency_hz), len(phi_deg_full), len(theta_deg))
        if e_theta_full.shape != expected or e_phi_full.shape != expected:
            raise ValueError(f"FFS field shape disagrees with grids: {source}")
        pattern_power_full = _pattern_power(
            e_theta_full,
            e_phi_full,
            theta_deg,
            phi_deg_full,
        )

        selected_frequency_ghz = frequency_ghz[frequency_mask]
        phi_deg = phi_deg_full[:-1]
        e_theta = e_theta_full[frequency_mask, :-1, :]
        e_phi = e_phi_full[frequency_mask, :-1, :]
        p_rad = np.asarray(ffs["p_rad"], dtype=np.float64)[frequency_mask]
        p_acc = np.asarray(ffs["p_acc"], dtype=np.float64)[frequency_mask]
        p_stim = np.asarray(ffs["p_stim"], dtype=np.float64)[frequency_mask]

        grids = (selected_frequency_ghz, theta_deg, phi_deg)
        if reference_grids is None:
            reference_grids = tuple(grid.copy() for grid in grids)
        else:
            for name, actual, reference in zip(
                ("frequency", "theta", "phi"),
                grids,
                reference_grids,
                strict=True,
            ):
                if not np.allclose(actual, reference, atol=1.0e-10, rtol=0.0):
                    raise ValueError(f"FFS {name} grid mismatch: {source}")

        case_names.append(str(row.link_case_name))
        rank_match = RANK_PATTERN.search(str(row.link_case_name))
        candidate_ranks.append(-1 if rank_match is None else int(rank_match.group(1)))
        source_paths.append(str(source))
        source_hashes.append(source_hash)
        e_theta_cases.append(e_theta)
        e_phi_cases.append(e_phi)
        p_rad_cases.append(p_rad)
        p_acc_cases.append(p_acc)
        p_stim_cases.append(p_stim)
        pattern_power_cases.append(pattern_power_full[frequency_mask])
        print(
            f"[LinkFFSTensor] parsed {completed}/{case_count}: "
            f"{row.link_case_name}"
        )

    if reference_grids is None:
        raise RuntimeError("no FFS grids were loaded")
    frequency_ghz, theta_deg, phi_deg = reference_grids
    e_theta_array = np.stack(e_theta_cases)
    e_phi_array = np.stack(e_phi_cases)
    p_rad_array = np.stack(p_rad_cases)
    p_acc_array = np.stack(p_acc_cases)
    p_stim_array = np.stack(p_stim_cases)
    pattern_power_array = np.stack(pattern_power_cases)
    rad_efficiency = p_rad_array / p_acc_array
    total_efficiency = p_rad_array / p_stim_array

    arrays = {
        "case_name": np.asarray(case_names),
        "candidate_rank": np.asarray(candidate_ranks, dtype=np.int64),
        "source_ffs_path": np.asarray(source_paths),
        "source_ffs_sha256": np.asarray(source_hashes),
        "frequency_ghz": frequency_ghz,
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
        "E_theta": e_theta_array,
        "E_phi": e_phi_array,
        "pattern_power": pattern_power_array,
        "p_rad_W": p_rad_array,
        "p_acc_W": p_acc_array,
        "p_stim_W": p_stim_array,
        "radiation_efficiency": rad_efficiency,
        "total_efficiency": total_efficiency,
    }
    _write_npz_atomic(arrays, output)
    payload: dict[str, Any] = {
        "schema": "msabp.link_ffs_tensor.v1",
        "case_count": len(case_names),
        "band_ghz_inclusive": [lower_ghz, upper_ghz],
        "array_shape_case_frequency_phi_theta": list(e_theta_array.shape),
        "complex_dtype": str(e_theta_array.dtype),
        "phi_periodic_endpoint_removed": True,
        "array_definitions": {
            "E_theta": "complex CST spherical theta field",
            "E_phi": "complex CST spherical phi field",
            "pattern_power": "integral of abs(E_theta)^2+abs(E_phi)^2 over 4pi",
            "radiation_efficiency": "p_rad/p_acc",
            "total_efficiency": "p_rad/p_stim",
        },
        "input": {"path": str(selected_path), "sha256": _sha256(selected_path)},
        "output": {"path": str(output), "sha256": _sha256(output)},
        "sources": [
            {"case_name": name, "path": path, "sha256": source_hash}
            for name, path, source_hash in zip(
                case_names,
                source_paths,
                source_hashes,
                strict=True,
            )
        ],
    }
    _write_json_atomic(payload, manifest)
    print(f"[LinkFFSTensor] tensor shape={e_theta_array.shape} -> {output}")
    print(f"[LinkFFSTensor] manifest -> {manifest}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-table", type=Path, default=DEFAULT_SELECTED_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--band", nargs=2, type=float, default=DEFAULT_BAND_GHZ)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_tensor(
        args.selected_table,
        args.output,
        args.manifest,
        band_ghz=(float(args.band[0]), float(args.band[1])),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
