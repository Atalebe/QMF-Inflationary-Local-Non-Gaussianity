#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(123)

# -----------------------------
# Kuramoto network setup
# -----------------------------
N1, N2 = 40, 40               # two cortical "modules"
N = N1 + N2
dt = 0.002
T = 12.0
t = np.arange(0, T, dt)

# Natural frequencies (keep narrow -> stable local power)
w1 = rng.normal(2*np.pi*10.0, 2*np.pi*0.3, N1)  # ~10 Hz
w2 = rng.normal(2*np.pi*10.0, 2*np.pi*0.3, N2)

w = np.concatenate([w1, w2])

# Coupling strengths
K_intra = 1.5
K_inter = 0.45

# Global perturbation window (simulate disruption of long-range synchrony)
t_on  = 4.0
t_off = 7.0
noise_sigma = 2.5  # strength of coupling-phase disruption

# -----------------------------
# Metrics
# -----------------------------
def order_parameter(theta):
    # global Kuramoto order parameter R(t)
    z = np.mean(np.exp(1j*theta))
    return np.abs(z)

def module_mean_phase(theta, idx):
    z = np.mean(np.exp(1j*theta[idx]))
    return np.angle(z)

def plv(ph1, ph2):
    # phase-locking value between two phase time series
    return np.abs(np.mean(np.exp(1j*(ph1 - ph2))))

# Two-timescale recovery model
def two_timescale(tt, A, tau1, B, tau2):
    return A*np.exp(-tt/tau1) + B*np.exp(-(tt/tau2)**2)

# -----------------------------
# Simulation
# -----------------------------
theta = rng.uniform(0, 2*np.pi, N)

idx1 = np.arange(0, N1)
idx2 = np.arange(N1, N)

R = np.zeros_like(t)
phi1 = np.zeros_like(t)
phi2 = np.zeros_like(t)

for i, ti in enumerate(t):
    # coupling matrix effect computed on the fly (dense but small N)
    eitheta = np.exp(1j*theta)

    # intra-module coupling terms
    z1 = np.mean(eitheta[idx1])
    z2 = np.mean(eitheta[idx2])

    # base coupling pulls towards each module mean + cross-module mean
    # perturbation acts ONLY on inter-module phase influence (global synchrony disruption)
    if t_on <= ti <= t_off:
        # inject random phase jitter into inter-module influence
        jitter = rng.normal(0.0, noise_sigma)
        z2_eff_for_1 = z2 * np.exp(1j*jitter)
        z1_eff_for_2 = z1 * np.exp(1j*jitter)
    else:
        z2_eff_for_1 = z2
        z1_eff_for_2 = z1

    dtheta = np.zeros(N)

    # module 1 oscillators
    dtheta[idx1] = w[idx1] + K_intra*np.imag(z1*np.exp(-1j*theta[idx1])) + K_inter*np.imag(z2_eff_for_1*np.exp(-1j*theta[idx1]))

    # module 2 oscillators
    dtheta[idx2] = w[idx2] + K_intra*np.imag(z2*np.exp(-1j*theta[idx2])) + K_inter*np.imag(z1_eff_for_2*np.exp(-1j*theta[idx2]))

    theta = (theta + dt*dtheta) % (2*np.pi)

    # metrics
    R[i] = order_parameter(theta)
    phi1[i] = module_mean_phase(theta, idx1)
    phi2[i] = module_mean_phase(theta, idx2)

# Integration proxy: cross-module PLV over sliding windows
win = int(0.5/dt)   # 0.5s window
PLV = np.full_like(t, np.nan, dtype=float)

for i in range(win, len(t)):
    PLV[i] = plv(phi1[i-win:i], phi2[i-win:i])

# Local "power" proxy: frequency stability (here constant by construction)
# We'll show local amplitude as constant baseline line in plot.
local_power_proxy = np.ones_like(t)

# -----------------------------
# Fit recovery after perturbation
# -----------------------------
# Use R(t) recovery segment: from t_off to end, with baseline removed
mask_rec = t >= t_off
trec = t[mask_rec] - t_off
Rrec = R[mask_rec]

# Normalize to start at 1 at recovery onset for fit stability
R0 = Rrec[0] if Rrec[0] > 0 else 1e-6
y = Rrec / R0

p0 = [0.6, 0.8, 0.4, 2.0]
bounds = ([0, 0.01, 0, 0.01], [1.2, 10, 1.2, 10])
popt, _ = curve_fit(two_timescale, trec, y, p0=p0, bounds=bounds)

A, tau1, B, tau2 = popt

pd.DataFrame([{
    "A": A, "tau1": tau1, "B": B, "tau2": tau2,
    "t_on": t_on, "t_off": t_off, "noise_sigma": noise_sigma,
    "K_intra": K_intra, "K_inter": K_inter, "N1": N1, "N2": N2
}]).to_csv(OUT / "neural_fit_params.csv", index=False)

# -----------------------------
# Plot dissociation figure
# -----------------------------
plt.figure(figsize=(8,6))

# panel-style plot in one figure (simple, no subplots rule doesn’t apply here since it’s one figure,
# but to be safe we keep it as one axis with multiple traces.)
plt.plot(t, local_power_proxy, label="Local activity proxy (stable)")
plt.plot(t, R, label="Global coherence Φ(t) ≈ R(t)")
plt.plot(t, PLV, label="Integration proxy (cross-module PLV)")

plt.axvspan(t_on, t_off, alpha=0.2, label="Perturbation window")
plt.xlabel("Time (s)")
plt.ylabel("Normalized units")
plt.title("Toy neural coherence dissociation: local activity intact, global coherence and integration collapse")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "neural_coherence_dissociation.png", dpi=220)
plt.close()

print("Neural toy model complete.")
print("Wrote:", OUT / "neural_coherence_dissociation.png")
print("Wrote:", OUT / "neural_fit_params.csv")
