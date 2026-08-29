"""Plot cap-averaged realized gain for 20 random cases from the 512 DoE run."""
from __future__ import annotations
from pathlib import Path
import sys, time
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cap_gain import parse_ffs, cap_gain  # type: ignore

RAW = Path("D:/Academic/Proj_MSABP_2/results/raw/doe-11var-branch-up-lhs-512-001")
OUT_PNG = Path("D:/Academic/Proj_MSABP_2/results/figures/cap_gain_20samples.png")
THETA_MAX = [0, 5, 10, 15, 20, 25, 30]
N_SAMPLES = 20
SEED = 0


def main() -> None:
    ids = sorted(np.random.default_rng(SEED).choice(512, N_SAMPLES, replace=False).tolist())
    print(f"Selected sample_ids: {ids}")

    curves = {}  # sample_id -> {theta_max: (freq_ghz, G_real_dBi)}
    for sid in ids:
        t0 = time.time()
        ffs = parse_ffs(RAW / f"case_{sid:04d}" / "Farfield Source [1].ffs")
        df = cap_gain(ffs, THETA_MAX)
        curves[sid] = {
            x: (df[df.theta_max_deg == x].freq_ghz.to_numpy(),
                df[df.theta_max_deg == x].G_realized_dBi.to_numpy())
            for x in THETA_MAX
        }
        print(f"  case_{sid:04d} parsed in {time.time()-t0:.1f}s")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    colors = plt.cm.viridis(np.linspace(0, 1, N_SAMPLES))
    for ax, x in zip(axes_flat[:7], THETA_MAX):
        # stack for percentile band
        stack = []
        for i, sid in enumerate(ids):
            f, g = curves[sid][x]
            ax.plot(f, g, color=colors[i], lw=0.8, alpha=0.65)
            stack.append(g)
        stack = np.array(stack)
        median = np.median(stack, axis=0)
        ax.plot(f, median, color="black", lw=1.8, label="median")
        ax.fill_between(f,
                        np.percentile(stack, 10, axis=0),
                        np.percentile(stack, 90, axis=0),
                        color="black", alpha=0.10, label="10–90%")
        ax.set_title(f"θ_max = {x}°   (Ω_cap = {2*np.pi*(1-np.cos(np.deg2rad(x))):.3f} sr)")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
    for ax in axes[:, 0]:
        ax.set_ylabel("Realized gain (dBi)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (GHz)")

    # legend panel
    axes_flat[7].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[7].legend(handles, labels, loc="center", fontsize=11, frameon=False)
    axes_flat[7].text(0.5, 0.85, f"{N_SAMPLES} random cases\n"
                                  f"from doe-11var-branch-up-lhs-512-001\n"
                                  f"seed={SEED}",
                       ha="center", va="top", fontsize=10,
                       transform=axes_flat[7].transAxes)

    fig.suptitle("Cap-averaged realized gain vs frequency", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PNG, dpi=130)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
