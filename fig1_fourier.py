"""
Paper 4 — Figure 1: Fourier Power Spectrum of S(x)
Dimensionless power spectrum k³|S̃(k)|² for seven cluster profiles,
showing the emergent effective scalar field mass.
"""

import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator

# ============================================================
# Physics setup (same as paper4_fourier.py)
# ============================================================
hbar = 1.0546e-34
c = 2.998e8
eV = 1.602e-19
kpc = 3.086e19
Mpc = 3.086e22
H0_si = 70e3 / Mpc

def beta_model(r, r_c, beta=2/3):
    return (1.0 + (r / r_c)**2)**(-3.0 * beta / 2.0)

def hernquist(r, r_s):
    x = r / r_s + 1e-10
    return 1.0 / (x * (1.0 + x)**3)

def nfw(r, r_s):
    x = r / r_s + 1e-10
    return 1.0 / (x * (1.0 + x)**2)

def composite_stellar(r, r_star, r_gas, f_star=0.8):
    stellar = beta_model(r, r_star, beta=1.0)
    gas = beta_model(r, r_gas, beta=2/3)
    return f_star * stellar + (1.0 - f_star) * gas

def hankel_transform(profile_func, k, r_max=5000.0, profile_args=()):
    def integrand(r):
        kr = k * r
        sinc_kr = np.sin(kr) / kr if kr > 1e-10 else 1.0
        return profile_func(r, *profile_args) * r**2 * sinc_kr
    result, _ = quad(integrand, 0, r_max, limit=500, epsrel=1e-8)
    return 4.0 * np.pi * result

def k_to_mass(k_per_kpc):
    k_si = k_per_kpc / kpc
    return hbar * k_si / c * c**2 / eV

# ============================================================
# Compute power spectra
# ============================================================
k_array = np.logspace(np.log10(1e-4), np.log10(0.15), 400)

profiles = [
    (r"$\beta$-model ($r_c$=150 kpc)", beta_model, (150.0, 2/3), "#2196F3", "-"),
    (r"$\beta$-model ($r_c$=250 kpc)", beta_model, (250.0, 2/3), "#64B5F6", "--"),
    ("Hernquist ($r_s$=100 kpc)", hernquist, (100.0,), "#F44336", "-"),
    ("Hernquist ($r_s$=200 kpc)", hernquist, (200.0,), "#EF9A9A", "--"),
    ("NFW ($r_s$=400 kpc)", nfw, (400.0,), "#FF9800", "-"),
    ("Composite (100/300 kpc)", composite_stellar, (100.0, 300.0, 0.8), "#1B5E20", "-"),
    ("Composite (150/400 kpc)", composite_stellar, (150.0, 400.0, 0.8), "#4CAF50", "--"),
]

print("Computing power spectra...")
all_power = []
all_peaks = []

for name, func, args, color, ls in profiles:
    print(f"  {name}...")
    Sk = np.array([hankel_transform(func, k, profile_args=args) for k in k_array])
    power = k_array**3 * Sk**2
    power_norm = power / np.max(power)
    all_power.append(power_norm)
    
    idx = np.argmax(power)
    all_peaks.append(k_array[idx])

# ============================================================
# Create figure
# ============================================================
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6.5))

# Secondary x-axis for effective mass
ax2 = ax1.twiny()

# Plot power spectra
for i, (name, func, args, color, ls) in enumerate(profiles):
    lw = 2.5 if ls == "-" else 1.8
    ax1.plot(k_array, all_power[i], color=color, ls=ls, lw=lw, label=name, zorder=3)

# Shade target mass range
m_low = 1e-29  # eV
m_high = 1e-26  # eV
k_low = m_low * eV / (c * hbar) * kpc  # convert mass to k in 1/kpc
k_high = m_high * eV / (c * hbar) * kpc

ax1.axvspan(k_low, min(k_high, k_array[-1]), alpha=0.08, color='gold', zorder=1)
ax1.text(k_low * 2.5, 0.92, 'Paper 4\ntarget range', fontsize=9, color='#8B7500',
         fontstyle='italic', ha='left', va='top')

# Mark the composite peak
k_comp = all_peaks[5]  # Composite (100/300)
m_comp = k_to_mass(k_comp)
ax1.axvline(k_comp, color='#1B5E20', ls=':', alpha=0.6, lw=1.5, zorder=2)
ax1.annotate(f'$m_{{\\rm eff}}$ = {m_comp:.1e} eV\n$\\lambda_C$ = {2*np.pi/k_comp:.0f} kpc',
             xy=(k_comp, 0.97), xytext=(k_comp * 4, 0.82),
             fontsize=10, color='#1B5E20',
             arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1B5E20', alpha=0.9))

# Mark horizon mass scale
m_horizon = hbar * H0_si / eV
k_horizon = m_horizon * eV / (c * hbar) * kpc
if k_horizon > k_array[0]:
    ax1.axvline(k_horizon, color='gray', ls=':', alpha=0.4, lw=1)
    ax1.text(k_horizon * 1.3, 0.15, '$\\hbar H_0$', fontsize=9, color='gray', fontstyle='italic')

# Formatting
ax1.set_xscale('log')
ax1.set_xlabel(r'Wavenumber $k$ (kpc$^{-1}$)', fontsize=13)
ax1.set_ylabel(r'Normalized power $k^3 |\tilde{S}(k)|^2$', fontsize=13)
ax1.set_xlim(k_array[0], k_array[-1])
ax1.set_ylim(0, 1.05)
ax1.tick_params(labelsize=11)

# Legend
ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.95, ncol=1,
           borderaxespad=0.8)

# Top axis: effective mass in eV
mass_ticks = [1e-30, 1e-29, 1e-28, 1e-27]
k_ticks = [m * eV / (c * hbar) * kpc for m in mass_ticks]
# Filter to those within range
valid = [(k, m) for k, m in zip(k_ticks, mass_ticks) if k_array[0] <= k <= k_array[-1]]

ax2.set_xscale('log')
ax2.set_xlim(ax1.get_xlim())
if valid:
    ax2.set_xticks([v[0] for v in valid])
    ax2.set_xticklabels([f'$10^{{{int(np.log10(v[1]))}}}$' for v in valid], fontsize=10)
ax2.set_xlabel(r'Effective mass $m_{\rm eff} = \hbar k / c$ (eV)', fontsize=13, labelpad=10)
ax2.tick_params(labelsize=10)

# Title
ax1.set_title(r'Fourier Power Spectrum of $S(\mathbf{x})$: Emergent Scalar Field Mass',
              fontsize=14, fontweight='bold', pad=45)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/Fig1_Fourier_spectrum.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/Fig1_Fourier_spectrum.pdf', bbox_inches='tight')
print("\nFigure 1 saved.")
