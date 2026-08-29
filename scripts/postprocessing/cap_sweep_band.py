"""3.1–4.8 GHz band-averaged realized gain vs cap half-angle for 20 samples.

Band-average is done in linear (power) domain then converted back to dBi —
this is the physically meaningful "average gain over band".
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cap_gain import parse_ffs, cap_gain  # type: ignore

RAW = Path("D:/Academic/Proj_MSABP_2/results/raw/doe-11var-branch-up-lhs-512-001")
OUT_PNG = Path("D:/Academic/Proj_MSABP_2/results/figures/cap_sweep_band_3p1_4p8.png")

BAND = (3.1, 4.8)                                # GHz
CAPS = list(range(0, 41, 5))                     # 0..40 deg in 5° steps
N_SAMPLES = 20
SEED = 0


def main() -> None:
    ids = sorted(np.random.default_rng(SEED).choice(512, N_SAMPLES, replace=False).tolist())
    print(f"Samples: {ids}")

    # rows[cap] -> list of band-avg gain (linear) per sample
    band_avg = {c: [] for c in CAPS}
    for sid in ids:
        ffs = parse_ffs(RAW / f"case_{sid:04d}" / "Farfield Source [1].ffs")
        df = cap_gain(ffs, CAPS)
        for c in CAPS:
            sub = df[df.theta_max_deg == c]
            f = sub.freq_ghz.to_numpy()
            g_db = sub.G_realized_dBi.to_numpy()
            mask = (f >= BAND[0]) & (f <= BAND[1])
            g_lin = 10 ** (g_db[mask] / 10)
            band_avg[c].append(g_lin.mean())
    print("Computed all sample × cap combinations")

    # shape (N_samples, N_caps) in dBi
    mat_db = 10 * np.log10(np.array([[band_avg[c][i] for c in CAPS]
                                     for i in range(N_SAMPLES)]))

    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = plt.cm.viridis(np.linspace(0, 1, N_SAMPLES))
    for i, sid in enumerate(ids):
        ax.plot(CAPS, mat_db[i], marker="o", ms=3, lw=0.8,
                color=colors[i], alpha=0.7)
    med = np.median(mat_db, axis=0)
    p10, p90 = np.percentile(mat_db, [10, 90], axis=0)
    ax.plot(CAPS, med, color="black", lw=2.2, marker="o", label="median")
    ax.fill_between(CAPS, p10, p90, color="black", alpha=0.12,
                    label="10–90 percentile")
    ax.axhline(0, color="grey", lw=0.5, ls="--")

    # highlight the recommended range
    ax.axvspan(10, 20, color="tab:orange", alpha=0.12, label="recommended 10–20°")

    ax.set_xlabel("Cap half-angle θ_max (deg)")
    ax.set_ylabel(f"Band-avg realized gain, {BAND[0]}–{BAND[1]} GHz (dBi)")
    ax.set_title(f"How the broadside-cap metric depends on θ_max\n"
                 f"({N_SAMPLES} random cases from doe-11var-branch-up-lhs-512-001, seed={SEED})")
    ax.set_xticks(CAPS)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # sample-to-sample spread annotation
    spread = mat_db.max(axis=0) - mat_db.min(axis=0)
    ax2 = ax.twinx()
    ax2.plot(CAPS, spread, color="tab:red", lw=1.4, ls=":",
             marker="s", ms=3, label="spread (max–min, dB)")
    ax2.set_ylabel("Sample spread (dB)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"Wrote {OUT_PNG}")

    # quick numeric summary
    print("\nBand-avg realized gain (dBi) summary vs cap:")
    print(f"{'θ_max':>6} {'median':>8} {'min':>8} {'max':>8} {'spread':>8}")
    for j, c in enumerate(CAPS):
        col = mat_db[:, j]
        print(f"{c:>6d} {np.median(col):>8.2f} {col.min():>8.2f} "
              f"{col.max():>8.2f} {col.max()-col.min():>8.2f}")


if __name__ == "__main__":
    main()
