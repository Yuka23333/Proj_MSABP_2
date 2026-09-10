"""Build the unaggregated, per-frequency on-body proxy analysis table.

The sample index fixes the exact 929-design archive plus two references used
by the current far-field study. Each FFS file is reduced independently, then
the resulting frequency rows are joined back to the sample metadata. No
frequency weighting or broadband reduction is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.postprocessing.ffs_onbody_proxies import (  # noqa: E402
    DEFAULT_THETA_CAP_DEG,
    DEFAULT_THETA_H_DEG,
    DEFAULT_WORKERS,
    _write_csv_atomic,
    compute_all,
)


DEFAULT_SAMPLE_INDEX = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "watermelon_latitude_10deg"
    / "sample_index.csv"
)
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "processed" / "onbody_proxies"
DEFAULT_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIRECTORY / "onbody_proxies_per_frequency_3p1-4p8GHz.csv"
)
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIRECTORY / "per_frequency_manifest.json"
DEFAULT_BAND_GHZ = (3.1, 4.8)
PROXY_METADATA_COLUMNS = {
    "file",
    "path",
    "freq_hz",
    "freq_ghz",
    "n_theta",
    "n_phi",
    "n_samples",
    "theta_h_deg",
    "theta_cap_deg",
    "forward_endfire_phi_deg",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


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


def _load_sample_index(path: Path) -> tuple[pd.DataFrame, list[Path]]:
    metadata = pd.read_csv(path)
    required = {"sample_index", "sample_kind", "case_id", "ffs_path"}
    missing_columns = sorted(required.difference(metadata.columns))
    if missing_columns:
        raise ValueError(f"sample index lacks columns: {missing_columns}")
    if metadata.empty:
        raise ValueError("sample index is empty")

    metadata = metadata.copy()
    metadata["_source_order"] = np.arange(len(metadata), dtype=np.int64)
    metadata["_ffs_identity"] = metadata["ffs_path"].map(_path_identity)
    if metadata["_ffs_identity"].duplicated().any():
        duplicates = metadata.loc[
            metadata["_ffs_identity"].duplicated(keep=False), "ffs_path"
        ].tolist()
        raise ValueError(f"sample index contains duplicate FFS paths: {duplicates[:3]}")

    paths = [Path(value) for value in metadata["_ffs_identity"]]
    missing_files = [path for path in paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"sample index references {len(missing_files)} missing FFS files; "
            f"first: {missing_files[0]}"
        )
    return metadata, paths


def _join_metadata(
    metadata: pd.DataFrame,
    proxy_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    proxies = proxy_rows.copy()
    proxies["_ffs_identity"] = proxies["path"].map(_path_identity)
    proxies = proxies.rename(columns={"path": "parsed_ffs_path"})
    joined = metadata.merge(
        proxies,
        on="_ffs_identity",
        how="inner",
        validate="one_to_many",
        sort=False,
    )
    joined = joined.sort_values(
        ["_source_order", "freq_ghz"],
        kind="stable",
    ).reset_index(drop=True)

    frequency_grid = np.sort(joined["freq_ghz"].unique())
    expected_rows = len(metadata) * len(frequency_grid)
    if len(joined) != expected_rows:
        raise ValueError(
            "samples do not share a complete frequency grid: "
            f"expected {expected_rows} rows, got {len(joined)}"
        )
    for _, group in joined.groupby("_source_order", sort=False):
        if not np.allclose(
            group["freq_ghz"].to_numpy(dtype=np.float64),
            frequency_grid,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("samples do not share the same frequency grid")

    return joined.drop(columns=["_source_order", "_ffs_identity"]), frequency_grid


def prepare_frequency_table(
    sample_index_path: str | Path = DEFAULT_SAMPLE_INDEX,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    band_ghz: tuple[float, float] = DEFAULT_BAND_GHZ,
    theta_h_deg: float = DEFAULT_THETA_H_DEG,
    theta_cap_deg: float = DEFAULT_THETA_CAP_DEG,
    workers: int = DEFAULT_WORKERS,
    overwrite: bool = False,
) -> dict[str, Any]:
    sample_index = Path(sample_index_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if manifest.exists() and not overwrite:
        raise FileExistsError(
            f"manifest already exists; pass --overwrite to replace it: {manifest}"
        )

    metadata, paths = _load_sample_index(sample_index)
    print(
        f"[OnBodyFrequency] samples={len(paths)}, band={band_ghz[0]:g}-"
        f"{band_ghz[1]:g} GHz, workers={workers}"
    )
    proxy_rows, errors = compute_all(
        paths,
        theta_h_deg=theta_h_deg,
        theta_cap_deg=theta_cap_deg,
        band_ghz=band_ghz,
        fail_fast=True,
        workers=workers,
    )
    if errors:
        raise RuntimeError(f"unexpected retained FFS errors: {len(errors)}")
    table, frequency_grid = _join_metadata(metadata, proxy_rows)
    _write_csv_atomic(table, output, overwrite=overwrite)
    proxy_value_columns = [
        column for column in proxy_rows.columns if column not in PROXY_METADATA_COLUMNS
    ]

    payload: dict[str, Any] = {
        "schema": "msabp.onbody_proxies.per_frequency.v1",
        "frequency_aggregation": None,
        "band_ghz_inclusive": [float(band_ghz[0]), float(band_ghz[1])],
        "frequency_grid_ghz": frequency_grid.tolist(),
        "sample_count": int(len(metadata)),
        "archive_sample_count": int((metadata["sample_kind"] == "archive").sum()),
        "reference_sample_count": int((metadata["sample_kind"] == "reference").sum()),
        "frequency_count_per_sample": int(len(frequency_grid)),
        "row_count": int(len(table)),
        "proxy_column_count": int(len(proxy_value_columns)),
        "proxy_value_columns": proxy_value_columns,
        "theta_h_deg": float(theta_h_deg),
        "theta_cap_deg": float(theta_cap_deg),
        "workers": int(workers),
        "inputs": {
            "sample_index": str(sample_index),
            "sample_index_sha256": _sha256(sample_index),
        },
        "outputs": {
            "frequency_table": str(output),
            "frequency_table_sha256": _sha256(output),
        },
    }
    _write_json_atomic(payload, manifest)
    print(
        f"[OnBodyFrequency] wrote {len(table)} rows "
        f"({len(frequency_grid)} frequencies/sample) -> {output}"
    )
    print(f"[OnBodyFrequency] manifest -> {manifest}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--band", nargs=2, type=float, default=DEFAULT_BAND_GHZ)
    parser.add_argument("--theta-h", type=float, default=DEFAULT_THETA_H_DEG)
    parser.add_argument("--theta-cap", type=float, default=DEFAULT_THETA_CAP_DEG)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_frequency_table(
        args.sample_index,
        args.output,
        args.manifest,
        band_ghz=(float(args.band[0]), float(args.band[1])),
        theta_h_deg=args.theta_h,
        theta_cap_deg=args.theta_cap,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
