"""
Paper 6 Figure 4: Geometry-Normalized Offset Residual vs TSP
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

with open("/home/claude/geometry_normalized_results.json") as f:
    data = json.load(f)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [1, 1]})

# ---- TOP PANEL: Raw offsets vs TSP (same as Fig 3 top) ----
clusters_ordered = sorted(data["clusters"].items(), key=lambda x: x[1]["TSP"])

for name, c in clusters_ordered:
    color = '0.65' if c["excluded"] else 'C0'
    marker = 's' if c["excluded"] else 'o'
    zorder = 2 if not c["excluded"] else 1
    ax1.errorbar(c["TSP"], c["offset_kpc"], 
                yerr=30 if not c.get("offset_err") else 40,
                fmt=marker, color=color, markersize=8, capsize=4,
                zorder=zorder, markeredgecolor='k', markeredgewidth=0.5)
    
    # Labels
    xoff = 0.03 if name != "El Gordo" else -0.12
    yoff = 15 if name != "Sausage (N)" else -60
    fontcolor = '0.5' if c["excluded"] else 'k'
    ax1.annotate(name.replace(" (N)", ""), 
                (c["TSP"] + xoff, c["offset_kpc"] + yoff),
                fontsize=8, color=fontcolor, style='italic')

ax1.set_ylabel("Observed Offset (kpc)", fontsize=12)
ax1.set_title("(a) Raw Offsets vs Time Since Pericenter", fontsize=12, fontweight='bold')
ax1.set_xlim(-0.05, 1.35)
ax1.set_ylim(-50, 1000)
ax1.axhline(0, color='k', lw=0.5, ls='-')
ax1.text(0.02, 0.95, "Heterogeneous offset definitions\n(lensing-gas, galaxy-gas, relic-center)",
         transform=ax1.transAxes, fontsize=8, va='top', color='0.4', style='italic')

# ---- BOTTOM PANEL: Geometry-normalized residuals ----
for name, c in clusters_ordered:
    color = '0.65' if c["excluded"] else 'C0'
    marker = 's' if c["excluded"] else 'o'
    zorder = 2 if not c["excluded"] else 1
    
    norm_err = 40 / c["geom_scale_norm"]  # approximate
    
    ax2.errorbar(c["TSP"], c["normalized_offset"],
                yerr=norm_err,
                fmt=marker, color=color, markersize=8, capsize=4,
                zorder=zorder, markeredgecolor='k', markeredgewidth=0.5)
    
    xoff = 0.03 if name != "El Gordo" else -0.12
    yoff = 12 if name not in ["Sausage (N)", "Abell 2345"] else -25
    fontcolor = '0.5' if c["excluded"] else 'k'
    ax2.annotate(name.replace(" (N)", ""), 
                (c["TSP"] + xoff, c["normalized_offset"] + yoff),
                fontsize=8, color=fontcolor, style='italic')

# Memory kernel prediction curves
tau_values = [14.0, 7.0, 3.0]  # Gyr
labels = [r"$\tau = H_0^{-1}$", r"$\tau = H_0^{-1}/2$", r"$\tau = 3$ Gyr"]
styles = ['-', '--', ':']
T_form = 5.0
tsp_model = np.linspace(0, 1.4, 100)

# Scale to match the high-offset systems (Bullet, El Gordo)
scale_factor = 220  # kpc at TSP=0

for tau, label, ls in zip(tau_values, labels, styles):
    f_pre = (1 - np.exp(-T_form/tau)) / (1 - np.exp(-(T_form + tsp_model)/tau))
    ax2.plot(tsp_model, scale_factor * f_pre, ls=ls, color='C1', lw=1.5, 
            label=label, alpha=0.8)

# ΛCDM expectation (flat)
ax2.axhline(scale_factor, color='C3', ls='--', lw=1.5, alpha=0.6, 
           label=r"$\Lambda$CDM (time-independent)")

ax2.set_xlabel("Time Since Pericenter (Gyr)", fontsize=12)
ax2.set_ylabel("Geometry-Normalized Offset (kpc)", fontsize=12)
ax2.set_title("(b) Geometry-Normalized Residuals vs TSP", fontsize=12, fontweight='bold')
ax2.set_xlim(-0.05, 1.35)
ax2.set_ylim(-30, 500)
ax2.axhline(0, color='k', lw=0.5, ls='-')
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Statistics annotation
stats = data["statistics"]
ax2.text(0.02, 0.95, 
         f"N = {stats['N_clean']} (excl. Sausage)\n"
         f"Pearson r = {stats['pearson_r']:.2f} (p = {stats['pearson_p']:.2f})\n"
         f"Spearman ρ = {stats['spearman_rho']:.2f} (p = {stats['spearman_p']:.2f})",
         transform=ax2.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Caveat
ax2.text(0.02, 0.55,
         "Geometry normalization: first-order\nram-pressure scaling only",
         transform=ax2.transAxes, fontsize=7, va='top', color='0.4', style='italic')

plt.tight_layout()
plt.savefig("/home/claude/Fig4_geometry_normalized.png", dpi=200, bbox_inches='tight')
plt.savefig("/home/claude/Fig4_geometry_normalized.pdf", bbox_inches='tight')
print("Figure 4 saved.")
