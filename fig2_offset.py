"""
Paper 4 — Figure 2: Falsifiable Prediction
Lensing-gas offset versus time since merger for the memory kernel,
compared with ΛCDM (time-independent).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ============================================================
# Parameters
# ============================================================
t_formation = 5.0    # Gyr, cluster age before collision
tau_kernel = 14.0    # Gyr, Hubble-time kernel
x_star = 720.0       # kpc, galaxy position (bullet subcluster)
x_gas = 500.0        # kpc, gas position (post-collision)
f_gas = 0.85
f_star = 0.15

# Maximum possible offset (galaxies - gas)
offset_max = x_star - x_gas  # 220 kpc

# ============================================================
# Compute offset vs time for memory kernel
# ============================================================
t_since = np.logspace(-2, 1.3, 500)  # 0.01 to 20 Gyr

def compute_offset(t_post, tau, t_pre=5.0):
    t_tot = t_pre + t_post
    w_pre = tau * (np.exp(-t_post/tau) - np.exp(-t_tot/tau))
    w_post = tau * (1.0 - np.exp(-t_post/tau))
    f_pre = w_pre / (w_pre + w_post)
    
    # Pre-collision: all mass at galaxy position (720 kpc)
    x_pre = x_star
    
    # Post-collision: mass-weighted center
    x_post = f_star * x_star + f_gas * x_gas  # ~533 kpc
    
    # Memory-weighted center
    x_eff = f_pre * x_pre + (1 - f_pre) * x_post
    
    return x_eff - x_gas

offsets_14 = np.array([compute_offset(t, 14.0) for t in t_since])
offsets_5 = np.array([compute_offset(t, 5.0) for t in t_since])
offsets_50 = np.array([compute_offset(t, 50.0) for t in t_since])

# ΛCDM: constant offset (DM halo at galaxy position)
# Offset determined by DM halo position, independent of time
offset_CDM = offset_max  # ~220 kpc, constant

# ============================================================
# Observational data points
# ============================================================
# System, t_since (Gyr), offset (kpc), offset_err (kpc)
observations = [
    ("Bullet Cluster", 0.15, 220, 50),
    ("Abell 56", 0.52, 103, 40),
    # ("Abell 2744 (S core)", 0.55, 70, 50),  # complex multi-merger, less reliable
]

# ============================================================
# Create figure
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

# Shaded region between τ=5 and τ=50 kernels
ax.fill_between(t_since, offsets_5, offsets_50, alpha=0.15, color='#1565C0',
                label=r'Memory kernel range ($\tau$ = 5–50 Gyr)', zorder=2)

# Primary prediction (τ = 14 Gyr = Hubble time)
ax.plot(t_since, offsets_14, color='#1565C0', lw=3, zorder=4,
        label=r'Memory kernel ($\tau = H_0^{-1} \approx$ 14 Gyr)')

# ΛCDM prediction (flat line)
ax.axhline(y=offset_CDM, color='#C62828', lw=2.5, ls='--', zorder=3,
           label=r'$\Lambda$CDM (time-independent)')

# Shade ΛCDM uncertainty band
ax.axhspan(offset_CDM - 40, offset_CDM + 40, alpha=0.08, color='#C62828', zorder=1)

# Plot observations
for name, t_obs, off, err in observations:
    ax.errorbar(t_obs, off, yerr=err, fmt='o', markersize=10, capsize=5,
                color='#2E7D32', markeredgecolor='black', markeredgewidth=1.2,
                ecolor='#2E7D32', elinewidth=2, zorder=5)
    
    # Label
    if name == "Bullet Cluster":
        ax.annotate(name, xy=(t_obs, off), xytext=(t_obs * 2.5, off + 15),
                    fontsize=11, fontweight='bold', color='#2E7D32',
                    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#2E7D32', alpha=0.9))
    else:
        ax.annotate(name, xy=(t_obs, off), xytext=(t_obs * 2.8, off - 5),
                    fontsize=11, fontweight='bold', color='#2E7D32',
                    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#2E7D32', alpha=0.9))

# Annotations
# Mark the "healing zone"
ax.annotate('', xy=(8, 215), xytext=(8, 160),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(9, 187, r'$\Delta$: testable' + '\ndifference', fontsize=9, color='gray',
        ha='left', va='center', fontstyle='italic')

# Add "memory dominates" / "new writes accumulate" labels
ax.text(0.03, 235, 'Boundary memory\ndominates', fontsize=9, color='#1565C0',
        fontstyle='italic', ha='left', va='bottom')
ax.text(12, 150, 'New writes\naccumulate', fontsize=9, color='#1565C0',
        fontstyle='italic', ha='center', va='top')

# Formatting
ax.set_xscale('log')
ax.set_xlabel('Time since merger (Gyr)', fontsize=14)
ax.set_ylabel('Lensing\u2013gas offset (kpc)', fontsize=14)
ax.set_xlim(0.01, 20)
ax.set_ylim(0, 300)
ax.tick_params(labelsize=12)

# Custom x-axis ticks
ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10])
ax.set_xticklabels(['0.01', '0.03', '0.1', '0.3', '1', '3', '10'])

# Legend
ax.legend(loc='lower left', fontsize=11, framealpha=0.95, borderaxespad=1)

# Title
ax.set_title('Falsifiable Prediction: Lensing\u2013Gas Offset Decays with Merger Age',
             fontsize=14, fontweight='bold', pad=15)

# Add text box with key prediction
textstr = ('Key distinction:\n'
           r'$\Lambda$CDM: offset = const (DM halo position)' + '\n'
           'Memory kernel: offset decays as\n'
           'boundary accumulates new writes')
props = dict(boxstyle='round,pad=0.6', facecolor='lightyellow', edgecolor='gray', alpha=0.9)
ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/Fig2_offset_decay.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/Fig2_offset_decay.pdf', bbox_inches='tight')
print("Figure 2 saved.")
