"""Cap-averaged (broadband) gain from a CST .ffs farfield source file.

For each frequency in the file, compute the average gain over a spherical
cap centered at the north pole (theta = 0), for a family of cap half-angles
theta_max = 0, 5, 10, ..., 30 degrees.

IEEE-consistent definitions used
--------------------------------
Radiation intensity (up to constant):  U(θ,φ) ∝ |E_θ|² + |E_φ|²
Pattern-integrated radiated power:     P_pat(f) = ∫∫ U sinθ dθ dφ over full sphere
Cap-integrated radiated power:         P_cap(f,θ_m) = ∫∫_{θ≤θ_m} U sinθ dθ dφ

Cap-averaged directivity (IEEE Std 145 average directivity over a solid angle):
    D_cap = (1 / Ω_cap) · ∫∫_cap D(θ,φ) dΩ
          = 4π · P_cap / (Ω_cap · P_pat)

Efficiencies (from the .ffs power block):
    η_rad = P_rad / P_acc       radiation efficiency
    η_tot = P_rad / P_stim      total (incl. mismatch) efficiency

Cap-averaged gain / realized gain:
    G_cap        = η_rad · D_cap
    G_realized   = η_tot · D_cap

Degenerate cap θ_max = 0
    Ω_cap → 0, so we drop the integral and evaluate G on axis:
    D(0) = 4π · U(0,·) / P_pat  (all 73 φ samples encode the same physical field)
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# ---------- FFS parser -------------------------------------------------------

def parse_ffs(path: Path):
    with open(path) as f:
        lines = f.readlines()

    def find(prefix: str, start: int = 0) -> int:
        for i in range(start, len(lines)):
            if lines[i].startswith(prefix):
                return i
        raise ValueError(f"marker not found: {prefix}")

    n_freq = int(lines[find("// #Frequencies") + 1].split()[0])

    # Power block: 4 numeric lines per frequency (rad, acc, stim, freq_hz)
    pwr_start = find("// Radiated/Accepted/Stimulated Power") + 1
    p_rad = np.empty(n_freq); p_acc = np.empty(n_freq)
    p_stim = np.empty(n_freq); freq = np.empty(n_freq)
    idx = pwr_start
    for k in range(n_freq):
        while not lines[idx].strip(): idx += 1
        p_rad[k]  = float(lines[idx].split()[0]); idx += 1
        p_acc[k]  = float(lines[idx].split()[0]); idx += 1
        p_stim[k] = float(lines[idx].split()[0]); idx += 1
        freq[k]   = float(lines[idx].split()[0]); idx += 1

    grid_line = find("// >> Total #phi samples", idx)
    n_phi, n_theta = map(int, lines[grid_line + 1].split())

    # Per-frequency pattern blocks: (phi, theta, ReEt, ImEt, ReEp, ImEp) rows.
    # Layout observed in this file family: theta inner, phi outer.
    E_theta = np.empty((n_freq, n_phi, n_theta), dtype=np.complex128)
    E_phi   = np.empty((n_freq, n_phi, n_theta), dtype=np.complex128)
    theta_vec = None; phi_vec = None
    cursor = grid_line + 2
    for k in range(n_freq):
        hdr = find("// >> Phi, Theta", cursor)
        cursor = hdr + 1
        block = np.array(
            [list(map(float, lines[cursor + r].split()))
             for r in range(n_phi * n_theta)]
        )
        cursor += n_phi * n_theta
        phi_block   = block[:, 0].reshape(n_phi, n_theta)
        theta_block = block[:, 1].reshape(n_phi, n_theta)
        if theta_vec is None:
            theta_vec = theta_block[0]
            phi_vec   = phi_block[:, 0]
        E_theta[k] = block[:, 2].reshape(n_phi, n_theta) + 1j * block[:, 3].reshape(n_phi, n_theta)
        E_phi[k]   = block[:, 4].reshape(n_phi, n_theta) + 1j * block[:, 5].reshape(n_phi, n_theta)

    return dict(freq=freq, p_rad=p_rad, p_acc=p_acc, p_stim=p_stim,
                theta_deg=theta_vec, phi_deg=phi_vec,
                E_theta=E_theta, E_phi=E_phi)


# ---------- Gain computation -------------------------------------------------

def cap_gain(ffs: dict, theta_max_deg_list: list[int]) -> pd.DataFrame:
    theta = np.deg2rad(ffs["theta_deg"])          # (N_theta,)
    phi   = np.deg2rad(ffs["phi_deg"])            # (N_phi,)
    sin_theta = np.sin(theta)                     # (N_theta,)

    U = (np.abs(ffs["E_theta"])**2 + np.abs(ffs["E_phi"])**2)  # (N_f, N_phi, N_theta)

    # Integrate on the exported angular grid. CST includes both phi=0 and
    # phi=360; trapezoidal integration handles that periodic endpoint without
    # counting it twice. It also gives a cap's outer theta ring half weight,
    # which is important for a small cap sampled at 5-degree intervals.
    full_theta_integral = np.trapezoid(
        U * sin_theta[None, None, :], theta, axis=2
    )
    P_pat = np.trapezoid(full_theta_integral, phi, axis=1)

    eta_rad = ffs["p_rad"]  / ffs["p_acc"]
    eta_tot = ffs["p_rad"]  / ffs["p_stim"]

    rows = []
    for x in theta_max_deg_list:
        if x == 0:
            # Degenerate cap: on-axis point. All 73 φ samples share the same
            # |E|², so just take φ=0 (index 0).
            U_pole = U[:, 0, 0]                                # (N_f,)
            D_cap  = 4 * np.pi * U_pole / P_pat
            omega_cap = 0.0
        else:
            theta_max = np.deg2rad(x)
            mask = theta <= theta_max + 1e-12                  # (N_theta,)
            if not np.any(mask) or not np.isclose(theta[mask][-1], theta_max):
                raise ValueError(
                    f"theta_max={x} deg is not present on the exported theta grid"
                )
            cap_theta_integral = np.trapezoid(
                U[:, :, mask] * sin_theta[mask][None, None, :],
                theta[mask],
                axis=2,
            )
            P_cap = np.trapezoid(cap_theta_integral, phi, axis=1)
            omega_cap = 2 * np.pi * (1 - np.cos(np.deg2rad(x)))
            D_cap = 4 * np.pi * P_cap / (omega_cap * P_pat)
        for i, f in enumerate(ffs["freq"]):
            rows.append(dict(
                freq_ghz    = f / 1e9,
                theta_max_deg = x,
                omega_cap_sr = omega_cap,
                D_cap_dBi   = 10 * np.log10(D_cap[i]),
                G_cap_dBi   = 10 * np.log10(D_cap[i] * eta_rad[i]),
                G_realized_dBi = 10 * np.log10(D_cap[i] * eta_tot[i]),
            ))
    return pd.DataFrame(rows)


# ---------- CLI --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ffs", type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path("cap_gain.csv"))
    ap.add_argument("--theta-max", type=int, nargs="+",
                    default=[0, 5, 10, 15, 20, 25, 30])
    args = ap.parse_args()

    ffs = parse_ffs(args.ffs)
    print(f"Parsed {args.ffs.name}: "
          f"{len(ffs['freq'])} freqs, "
          f"{len(ffs['phi_deg'])} phi, {len(ffs['theta_deg'])} theta")

    df = cap_gain(ffs, args.theta_max)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(df)} rows)")

    # Pivot preview: rows = freq, cols = cap angle, values = realized gain
    piv = df.pivot(index="freq_ghz", columns="theta_max_deg",
                   values="G_realized_dBi")
    print("\nRealized-gain preview (dBi), rows = freq [GHz], cols = θ_max [deg]:")
    print(piv.round(2).to_string(max_rows=15))


if __name__ == "__main__":
    main()
