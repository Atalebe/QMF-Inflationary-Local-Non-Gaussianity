#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # qmf_campaign/
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------
# User-set knobs
# --------------------------
C = 1.0                     # O(1) coefficient (keep explicit)
fNL_planck_center = -0.9     # Planck central value for local fNL
fNL_planck_1sigma = 5.1     # set conservative; you can update to exact later
sigma_cut = 2.0             # "allowed" region = |fNL - center| <= sigma_cut*sigma
H_over_mM_min = 1/0.3       # because mM/H in [1e-3, 0.3] => H/mM in [~3.33, 1000]
# --------------------------

def shape_proxy(k1, k2, k3):
    """
    Semi-numerical proxy for a local-ish + folded-supported shape.
    This is NOT the full in-in result, but is a numerical sanity check:
    - Enhances squeezed (k1<<k2~k3)
    - Enhances folded (k1~k2+k3)
    - Suppresses equilateral-ish regions relative to squeezed/folded
    """
    ks = np.array([k1,k2,k3])
    kmin, kmid, kmax = np.sort(ks)

    # Squeezed weight: grows as kmin/kmax -> 0
    w_sq = (kmax / (kmin + 1e-12))**0.8

    # Folded weight: kmax ~ kmin + kmid
    w_fold = 1.0 / (np.abs(kmax - (kmin + kmid)) + 1e-3)

    # Mild suppression for perfect equilateral (optional)
    equi_penalty = 1.0 / (np.std(ks)/np.mean(ks) + 0.15)

    return (0.6*w_sq + 0.4*w_fold) / equi_penalty

def fNL_local(g, mM_over_H):
    # fNL ≈ C * g * H^2/mM^2 = C * g * (H/mM)^2 = C * g / (mM/H)^2
    return C * g / (mM_over_H**2)

def make_shape_scan(tag, tri_list):
    vals = []
    labels = []
    for (k1,k2,k3,label) in tri_list:
        vals.append(shape_proxy(k1,k2,k3))
        labels.append(label)
    vals = np.array(vals, float)

    plt.figure()
    x = np.arange(len(vals))
    plt.plot(x, vals, marker="o")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("S(k1,k2,k3) proxy amplitude")
    plt.title(f"Shape proxy check: {tag}")
    plt.tight_layout()
    plt.savefig(OUT / f"shape_{tag}.png", dpi=200)
    plt.close()

def main():
    # --- shape sanity checks
    make_shape_scan("squeezed", [
        (0.01, 1.0, 1.0, "0.01,1,1"),
        (0.05, 1.0, 1.0, "0.05,1,1"),
        (0.10, 1.0, 1.0, "0.10,1,1"),
    ])
    make_shape_scan("folded", [
        (1.0, 1.0, 2.0, "1,1,2 (folded)"),
        (0.8, 1.2, 2.0, "0.8,1.2,2"),
        (0.9, 1.1, 2.0, "0.9,1.1,2"),
    ])
    make_shape_scan("equilateral", [
        (1.0, 1.0, 1.0, "1,1,1"),
        (1.0, 1.0, 1.05, "1,1,1.05"),
        (1.0, 1.0, 1.10, "1,1,1.10"),
    ])

    # --- parameter grid for fNL
    g_grid = np.logspace(-4, 0, 220)
    mM_over_H_grid = np.logspace(-3, np.log10(0.3), 220)

    G, M = np.meshgrid(g_grid, mM_over_H_grid, indexing="xy")
    F = fNL_local(G, M)

    # Planck-allowed region (placeholder, editable)
    fNL_lo = fNL_planck_center - sigma_cut*fNL_planck_1sigma
    fNL_hi = fNL_planck_center + sigma_cut*fNL_planck_1sigma
    allowed = (F >= fNL_lo) & (F <= fNL_hi)

    plt.figure()
    # Plot log10 |fNL| for dynamic range
    plt.imshow(np.log10(np.abs(F) + 1e-12),
               origin="lower",
               aspect="auto",
               extent=[np.log10(g_grid.min()), np.log10(g_grid.max()),
                       np.log10(mM_over_H_grid.min()), np.log10(mM_over_H_grid.max())])
    plt.colorbar(label="log10(|f_NL^loc|)")

    # Overlay allowed contour
    # show boundary where allowed flips
    plt.contour(np.log10(g_grid), np.log10(mM_over_H_grid), allowed.astype(float),
                levels=[0.5], linewidths=1.5)

    plt.xlabel("log10(g)")
    plt.ylabel("log10(m_M/H)")
    plt.title("f_NL^loc(g, m_M/H) with Planck-allowed contour (placeholder bounds)")
    plt.tight_layout()
    plt.savefig(OUT / "fNL_grid.png", dpi=220)
    plt.close()

    print("Wrote outputs: outputs/shape_*.png and outputs/fNL_grid.png")

if __name__ == "__main__":
    main()
