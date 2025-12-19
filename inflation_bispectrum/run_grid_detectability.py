#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------
# Pull the Planck settings from your existing run_grid.py (so they stay consistent)
# --------------------------
src = Path(__file__).resolve().parent / "run_grid.py"
txt = src.read_text()

def grab_float(name, default):
    m = re.search(rf"^{name}\s*=\s*([0-9eE\.\+\-]+)", txt, flags=re.M)
    return float(m.group(1)) if m else default

C = grab_float("C", 1.0)
fNL_planck_center = grab_float("fNL_planck_center", -0.9)
fNL_planck_1sigma = grab_float("fNL_planck_1sigma", 5.1)
sigma_cut = grab_float("sigma_cut", 2.0)

# --------------------------
# Detectability thresholds (set explicitly; adjust later if you want)
# These are "targeted sensitivity" lines, not claims about official forecasts.
# --------------------------
fNL_SO_thresh  = 2.0   # Simons Observatory rough target scale (order-of-magnitude)
fNL_S4_thresh  = 1.0   # CMB-S4 rough target scale (order-of-magnitude)

# --------------------------
# Grid
# --------------------------
g_grid = np.logspace(-4, 0, 220)
mM_over_H_grid = np.logspace(-3, np.log10(0.3), 220)

G, M = np.meshgrid(g_grid, mM_over_H_grid, indexing="xy")
F = C * G / (M**2)

# Planck allowed band (2σ by default)
lo = fNL_planck_center - sigma_cut*fNL_planck_1sigma
hi = fNL_planck_center + sigma_cut*fNL_planck_1sigma
allowed = (F >= lo) & (F <= hi)

# For contouring detectability: |fNL| >= threshold
det_SO = (np.abs(F) >= fNL_SO_thresh)
det_S4 = (np.abs(F) >= fNL_S4_thresh)

# Plot
plt.figure(figsize=(7.4, 5.6))
img = plt.imshow(
    np.log10(np.abs(F) + 1e-12),
    origin="lower",
    aspect="auto",
    extent=[
        np.log10(g_grid.min()), np.log10(g_grid.max()),
        np.log10(mM_over_H_grid.min()), np.log10(mM_over_H_grid.max())
    ],
)
plt.colorbar(img, label="log10(|f_NL^loc|)")

# Planck allowed boundary
plt.contour(
    np.log10(g_grid), np.log10(mM_over_H_grid),
    allowed.astype(float),
    levels=[0.5], linewidths=2.0
)

# Detectability contours
plt.contour(
    np.log10(g_grid), np.log10(mM_over_H_grid),
    det_SO.astype(float),
    levels=[0.5], linewidths=1.6, linestyles="--"
)
plt.contour(
    np.log10(g_grid), np.log10(mM_over_H_grid),
    det_S4.astype(float),
    levels=[0.5], linewidths=1.6, linestyles=":"
)

plt.xlabel("log10(g)")
plt.ylabel("log10(m_M/H)")
plt.title(
    "f_NL^loc(g, m_M/H): Planck-allowed region and detectability contours\n"
    f"Planck {sigma_cut:.0f}σ band: [{lo:.1f}, {hi:.1f}]   "
    f"SO: |fNL|≥{fNL_SO_thresh:.1f} (--)   CMB-S4: |fNL|≥{fNL_S4_thresh:.1f} (:)"
)

# Add a manual legend proxy
from matplotlib.lines import Line2D
legend_lines = [
    Line2D([0],[0], color="k", linewidth=2.0, label=f"Planck {sigma_cut:.0f}σ allowed contour"),
    Line2D([0],[0], color="k", linewidth=1.6, linestyle="--", label=f"SO threshold |fNL|≥{fNL_SO_thresh:.1f}"),
    Line2D([0],[0], color="k", linewidth=1.6, linestyle=":", label=f"CMB-S4 threshold |fNL|≥{fNL_S4_thresh:.1f}"),
]
plt.legend(handles=legend_lines, loc="upper right", frameon=True)

plt.tight_layout()
outpath = OUT / "fNL_grid_detectability.png"
plt.savefig(outpath, dpi=240)
plt.close()

print("Wrote:", outpath)
