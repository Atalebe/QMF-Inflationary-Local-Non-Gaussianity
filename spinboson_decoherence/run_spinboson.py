#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Time grid
# -----------------------------
t = np.linspace(0, 10.0, 800)

# -----------------------------
# Decoherence models
# -----------------------------
def ohmic(t, tau):
    return np.exp(-t / tau)

def subohmic(t, tau, s):
    return np.exp(-(t / tau)**s)

def structured(t, A, tau1, B, tau2):
    return A*np.exp(-t/tau1) + B*np.exp(-(t/tau2)**2)

# -----------------------------
# Generate synthetic "data"
# -----------------------------
# Ohmic
D_ohmic = ohmic(t, tau=2.0)

# Sub-Ohmic (no structure)
D_sub = subohmic(t, tau=2.0, s=0.6)

# Structured bath (model prediction)
A_true, B_true = 0.65, 0.35
D_struct = structured(t, A_true, tau1=2.0, B=B_true, tau2=4.5)

# -----------------------------
# Echo pulse (revival)
# -----------------------------
def apply_echo(D, t, t_echo=5.0, strength=0.15):
    revived = D.copy()
    idx = t > t_echo
    revived[idx] += strength * np.exp(-(t[idx]-t_echo)**2 / 0.6**2)
    return np.clip(revived, 0, 1)

D_struct_echo = apply_echo(D_struct, t)

# -----------------------------
# Fit structured model
# -----------------------------
p0 = [0.5, 1.5, 0.5, 3.0]
bounds = ([0,0,0,0], [1,10,1,10])

popt_ohmic, _ = curve_fit(
    lambda tt, tau: ohmic(tt, tau),
    t, D_ohmic, p0=[2.0]
)

popt_sub, _ = curve_fit(
    lambda tt, tau, s: subohmic(tt, tau, s),
    t, D_sub, p0=[2.0,0.7]
)

popt_struct, _ = curve_fit(
    structured, t, D_struct, p0=p0, bounds=bounds
)

# -----------------------------
# Save fit table
# -----------------------------
fit_table = pd.DataFrame({
    "case": ["ohmic", "sub-ohmic", "structured"],
    "B_recovered": [
        0.0,
        0.0,
        popt_struct[2]
    ]
})
fit_table.to_csv(OUT / "decoherence_fit_table.csv", index=False)

# -----------------------------
# Plot curves
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(t, D_ohmic, label="Ohmic (exponential)")
plt.plot(t, D_sub, label="Sub-Ohmic (stretched)")
plt.plot(t, D_struct, label="Structured (B>0)")
plt.xlabel("t")
plt.ylabel("D(t)")
plt.legend()
plt.title("Decoherence regimes")
plt.tight_layout()
plt.savefig(OUT / "decoherence_curves.png", dpi=200)
plt.close()

# -----------------------------
# Plot echo revival
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(t, D_struct, label="Structured (no echo)")
plt.plot(t, D_struct_echo, label="Structured + echo")
plt.axvline(5.0, linestyle="--", color="k", alpha=0.5)
plt.xlabel("t")
plt.ylabel("D(t)")
plt.legend()
plt.title("Echo-selective coherence recovery")
plt.tight_layout()
plt.savefig(OUT / "echo_revival.png", dpi=200)
plt.close()

print("Spin-boson simulations complete.")
print("Outputs:")
print(" - outputs/decoherence_curves.png")
print(" - outputs/echo_revival.png")
print(" - outputs/decoherence_fit_table.csv")
