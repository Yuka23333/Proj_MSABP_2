"""Per-sample broadside cap-gain scalar table for the 11-var DoE runs.

Metric: realized gain averaged (power domain) over
        band  3.1–4.8 GHz  &  cap  θ ∈ [0°, 15°]
Output: two columns   sample_id, cap_gain_dBi
"""
from __future__ import annotations
from pathlib import Path
import sys, time
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cap_gain import parse_ffs, cap_gain  # type: ignore

RAW = Path("D:/Academic/Proj_MSABP_2/results/raw")
OUT_DIR = Path("D:/Academic/Proj_MSABP_2/results/processed")

RUNS = [
    ("doe-11var-branch-up-lhs-512-001",        "cap_gain_doe_11var_lhs_512.csv"),
    ("doe-11var-branch-up-lhs-holdout-64-001", "cap_gain_doe_11var_lhs_holdout_64.csv"),
]

BAND = (3.1, 4.8)     # GHz
THETA_MAX = 15        # deg


def per_case_scalar(ffs_path: Path) -> float:
    ffs = parse_ffs(ffs_path)
    df  = cap_gain(ffs, [THETA_MAX])
    sub = df[df.theta_max_deg == THETA_MAX]
    f   = sub.freq_ghz.to_numpy()
    g_db = sub.G_realized_dBi.to_numpy()
    mask = (f >= BAND[0]) & (f <= BAND[1])
    g_lin_avg = np.mean(10 ** (g_db[mask] / 10))
    return 10 * np.log10(g_lin_avg)


def build_table(run_dir: Path) -> pd.DataFrame:
    cases = sorted(
        (int(sub.name.split("_")[1]), sub / "Farfield Source [1].ffs")
        for sub in run_dir.iterdir()
        if sub.is_dir() and (sub / "Farfield Source [1].ffs").exists()
    )
    print(f"  {len(cases)} cases")
    rows = []
    t0 = time.time()
    for k, (sid, path) in enumerate(cases, 1):
        rows.append((sid, per_case_scalar(path)))
        if k % 50 == 0 or k == len(cases):
            elapsed = time.time() - t0
            print(f"    {k}/{len(cases)}  ({elapsed:.0f}s  "
                  f"~{elapsed/k*len(cases):.0f}s total)")
    return pd.DataFrame(rows, columns=["sample_id", "cap_gain_dBi"])


def main() -> None:
    for run_name, out_name in RUNS:
        print(f"[{run_name}]")
        df = build_table(RAW / run_name)
        out = OUT_DIR / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  wrote {out}  shape={df.shape}")
        print(f"  cap_gain_dBi  min={df.cap_gain_dBi.min():.2f}  "
              f"median={df.cap_gain_dBi.median():.2f}  "
              f"max={df.cap_gain_dBi.max():.2f}")


if __name__ == "__main__":
    main()
