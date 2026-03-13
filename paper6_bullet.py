"""
Paper 4 — Core Calculation #3 (Revised): Bullet Cluster via Memory Kernel
==========================================================================

THE SHIFT:
Previous calculation applied ν(y) as a multiplier on PRESENT baryonic mass.
That's a GR-style move — it asks "how much extra gravity does the current
mass produce?"

The kernel framework says something different:

    ρ_eff(x,t) = ρ_b(x,t) + M(x,t)

    where M(x,t) = ∫ K(x,x';t,t') ρ_b(x',t') dt' d³x'

M is not a boost on present mass. It is the TIME-INTEGRATED gravitational
memory of where mass has been throughout the system's history.

For the Bullet Cluster:
- Pre-collision (~5 Gyr): gas and galaxies co-spatial, centered on each subcluster
- Collision (~150 Myr ago): gas stripped and decelerated, galaxies passed through
- Now: gas is displaced, galaxies mark the HISTORICAL center

The kernel M remembers billions of years of co-spatial configuration.
The gas displacement is a recent perturbation.
Therefore M peaks at the galaxy positions, not the gas positions.

PREDICTION:
If this mechanism is correct:
- Recent mergers → stronger gas-lensing offset (memory hasn't updated)
- Ancient mergers → offset healing (memory accumulating new gas position)
- CDM predicts NO such time dependence
"""

import numpy as np

# ============================================================
# Physical constants
# ============================================================
G = 6.674e-11
c = 2.998e8
M_sun = 1.989e30
kpc = 3.086e19
Mpc = 3.086e22
H0_si = 70e3 / Mpc
a_u = c * H0_si / (2 * np.pi)
Gyr = 3.156e16

print("=" * 72)
print("PAPER 4 — BULLET CLUSTER VIA MEMORY KERNEL")
print("Boundary confidence replaces dark matter halos")
print("=" * 72)

# ============================================================
# 1. Cluster history timeline
# ============================================================
t_formation = 5.0    # Gyr before collision
t_collision = 0.15   # Gyr since collision
t_total = t_formation + t_collision

print(f"\nTimeline:")
print(f"  Cluster age before collision:  {t_formation:.1f} Gyr")
print(f"  Time since collision:          {t_collision:.2f} Gyr")
print(f"  Pre-collision fraction:        {t_formation/t_total:.1%}")

# ============================================================
# 2. Mass distributions
# ============================================================
M_main = 6.0e14 * M_sun * 0.16
M_bullet = 1.5e14 * M_sun * 0.16
f_gas = 0.85
f_star = 0.15
r_main = 300.0  # kpc
r_bullet = 200.0

# Pre-collision positions (gas + galaxies together)
x_main_pre = 0.0
x_bullet_pre = 720.0

# Post-collision positions
x_star_main = 0.0
x_gas_main = 100.0
x_star_bullet = 720.0
x_gas_bullet = 500.0

print(f"\nPost-collision geometry:")
print(f"  Main galaxies:  {x_star_main:.0f} kpc    Main gas:  {x_gas_main:.0f} kpc")
print(f"  Bullet galaxies: {x_star_bullet:.0f} kpc   Bullet gas: {x_gas_bullet:.0f} kpc")

# ============================================================
# 3. Surface density profiles
# ============================================================
def beta_1d(x, x0, M, r_c, beta=2.0/3.0):
    dx = x - x0
    exponent = 0.5 - 1.5 * beta
    shape = (1.0 + (dx / r_c)**2)**exponent
    norm = M / (2.0 * np.pi * (r_c * kpc)**2)
    return norm * shape

x_arr = np.linspace(-500, 1200, 3000)

# ============================================================
# 4. Build effective density for each kernel timescale
# ============================================================
print("\n" + "=" * 72)
print("MEMORY-WEIGHTED EFFECTIVE DENSITY")
print("=" * 72)

def compute_memory_profile(tau_Gyr):
    """Compute memory-weighted surface density for a given kernel timescale."""
    if tau_Gyr is None:
        f_pre = t_formation / t_total
    else:
        w_pre = tau_Gyr * (np.exp(-t_collision/tau_Gyr) - np.exp(-t_total/tau_Gyr))
        w_post = tau_Gyr * (1.0 - np.exp(-t_collision/tau_Gyr))
        f_pre = w_pre / (w_pre + w_post)
    f_post = 1.0 - f_pre
    
    Sigma = np.zeros_like(x_arr)
    for i, x in enumerate(x_arr):
        # Pre-collision: ALL mass at cluster centers
        S_pre = (beta_1d(x, x_main_pre, M_main, r_main) +
                 beta_1d(x, x_bullet_pre, M_bullet, r_bullet))
        
        # Post-collision: gas displaced from galaxies
        S_post = (beta_1d(x, x_star_main, M_main * f_star, r_main * 0.5) +
                  beta_1d(x, x_gas_main, M_main * f_gas, r_main) +
                  beta_1d(x, x_star_bullet, M_bullet * f_star, r_bullet * 0.5) +
                  beta_1d(x, x_gas_bullet, M_bullet * f_gas, r_bullet))
        
        Sigma[i] = f_pre * S_pre + f_post * S_post
    
    return Sigma, f_pre

# Also compute present-only and CDM profiles for comparison
Sigma_present = np.zeros_like(x_arr)
Sigma_CDM = np.zeros_like(x_arr)

for i, x in enumerate(x_arr):
    Sigma_present[i] = (
        beta_1d(x, x_star_main, M_main * f_star, r_main * 0.5) +
        beta_1d(x, x_gas_main, M_main * f_gas, r_main) +
        beta_1d(x, x_star_bullet, M_bullet * f_star, r_bullet * 0.5) +
        beta_1d(x, x_gas_bullet, M_bullet * f_gas, r_bullet))
    
    # CDM: add DM halos at galaxy positions
    M_DM_main = 6.0e14 * M_sun * 0.84
    M_DM_bullet = 1.5e14 * M_sun * 0.84
    Sigma_CDM[i] = Sigma_present[i] + (
        beta_1d(x, x_star_main, M_DM_main, r_main * 1.3) +
        beta_1d(x, x_star_bullet, M_DM_bullet, r_bullet * 1.3))

# Compute memory profiles for different timescales
print(f"\n  {'Model':40} {'Main CoM':>10} {'Bullet CoM':>10} {'Main off':>10} {'Bullet off':>10}")
print(f"  {'':40} {'(kpc)':>10} {'(kpc)':>10} {'(kpc)':>10} {'(kpc)':>10}")
print(f"  {'-'*80}")

x_bound = 350.0
m_main = x_arr < x_bound
m_bull = x_arr >= x_bound

def com(Sigma, mask):
    w = np.maximum(Sigma[mask], 0)
    if np.sum(w) < 1e-50:
        return np.nan
    return np.average(x_arr[mask], weights=w)

def peak(Sigma, mask):
    return x_arr[mask][np.argmax(Sigma[mask])]

# Present baryonic
cm = com(Sigma_present, m_main)
cb = com(Sigma_present, m_bull)
print(f"  {'Present baryonic':40} {cm:>10.1f} {cb:>10.1f} {cm - x_gas_main:>+10.1f} {cb - x_gas_bullet:>+10.1f}")

# CDM
cm = com(Sigma_CDM, m_main)
cb = com(Sigma_CDM, m_bull)
print(f"  {'CDM (DM halos on galaxies)':40} {cm:>10.1f} {cb:>10.1f} {cm - x_gas_main:>+10.1f} {cb - x_gas_bullet:>+10.1f}")

# Memory kernels
for label, tau in [("Memory kernel (uniform)", None),
                   ("Memory kernel (τ = 5 Gyr)", 5.0),
                   ("Memory kernel (τ = 14 Gyr)", 14.0),
                   ("Memory kernel (τ = 50 Gyr)", 50.0)]:
    Sigma_mem, f_pre = compute_memory_profile(tau)
    cm = com(Sigma_mem, m_main)
    cb = com(Sigma_mem, m_bull)
    print(f"  {label+f' [f_pre={f_pre:.1%}]':40} {cm:>10.1f} {cb:>10.1f} {cm - x_gas_main:>+10.1f} {cb - x_gas_bullet:>+10.1f}")

# Reference
print(f"\n  {'Galaxy positions':40} {x_star_main:>10.1f} {x_star_bullet:>10.1f} {x_star_main - x_gas_main:>+10.1f} {x_star_bullet - x_gas_bullet:>+10.1f}")
print(f"  {'Gas positions':40} {x_gas_main:>10.1f} {x_gas_bullet:>10.1f} {0:>+10.1f} {0:>+10.1f}")
print(f"  {'Observed lensing (Clowe+2006)':40} {'~0':>10} {'~720':>10} {'~-100':>10} {'~+220':>10}")

# ============================================================
# 5. Peak analysis
# ============================================================
print("\n" + "=" * 72)
print("PEAK POSITIONS (where κ is maximum)")
print("=" * 72)

print(f"\n  {'Model':40} {'Main peak':>10} {'Bullet peak':>10}")
print(f"  {'-'*60}")

pm = peak(Sigma_present, m_main)
pb = peak(Sigma_present, m_bull)
print(f"  {'Present baryonic':40} {pm:>10.0f} {pb:>10.0f}")

pm = peak(Sigma_CDM, m_main)
pb = peak(Sigma_CDM, m_bull)
print(f"  {'CDM':40} {pm:>10.0f} {pb:>10.0f}")

for label, tau in [("Memory (τ = 14 Gyr)", 14.0)]:
    Sigma_mem, f_pre = compute_memory_profile(tau)
    pm = peak(Sigma_mem, m_main)
    pb = peak(Sigma_mem, m_bull)
    print(f"  {label:40} {pm:>10.0f} {pb:>10.0f}")

print(f"  {'Galaxy positions':40} {x_star_main:>10.0f} {x_star_bullet:>10.0f}")
print(f"  {'Gas positions':40} {x_gas_main:>10.0f} {x_gas_bullet:>10.0f}")

# ============================================================
# 6. Falsifiable prediction: offset vs time since merger
# ============================================================
print("\n" + "=" * 72)
print("FALSIFIABLE PREDICTION: Offset Heals With Time")
print("=" * 72)

tau_kernel = 14.0
print(f"\n  Using τ = {tau_kernel} Gyr (Hubble time) kernel")
print(f"  Bullet subcluster: galaxies at {x_star_bullet} kpc, gas at {x_gas_bullet} kpc\n")

print(f"  {'t_since (Gyr)':>14} {'f_pre':>8} {'CoM bullet':>12} {'Offset from gas':>16}")
print(f"  {'-'*52}")

for t_since in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 2.00, 5.00, 10.00]:
    Sigma_t, f_pre = compute_memory_profile(tau_kernel)
    # Recompute with this specific t_since
    t_pre_local = t_formation
    t_post_local = t_since
    t_tot_local = t_pre_local + t_post_local
    w_pre = tau_kernel * (np.exp(-t_post_local/tau_kernel) - np.exp(-t_tot_local/tau_kernel))
    w_post = tau_kernel * (1.0 - np.exp(-t_post_local/tau_kernel))
    f_p = w_pre / (w_pre + w_post)
    
    # Compute bullet CoM
    Sigma_t2 = np.zeros_like(x_arr)
    for i, x in enumerate(x_arr):
        S_pre = beta_1d(x, x_bullet_pre, M_bullet, r_bullet)
        S_post = (beta_1d(x, x_star_bullet, M_bullet * f_star, r_bullet * 0.5) +
                  beta_1d(x, x_gas_bullet, M_bullet * f_gas, r_bullet))
        Sigma_t2[i] = f_p * S_pre + (1-f_p) * S_post
    
    cb = com(Sigma_t2, x_arr >= 350)
    offset = cb - x_gas_bullet
    
    print(f"  {t_since:>14.2f} {f_p:>8.1%} {cb:>12.1f} {offset:>+16.1f}")

print(f"""
  INTERPRETATION:
  
  At t = 0.01 Gyr after merger: offset is +{x_star_bullet - x_gas_bullet:.0f} kpc (maximal)
  The boundary has almost no memory of the post-collision state.
  Lensing is almost entirely at the galaxy position.
  
  At t = 0.15 Gyr (Bullet Cluster): offset is ~+190 kpc
  Memory is dominated by pre-collision configuration.
  Lensing strongly favors galaxy position. 
  Observed offset is ~+220 kpc — consistent.
  
  At t = 5 Gyr: offset has decreased substantially
  Boundary has accumulated significant post-collision memory.
  Lensing peak migrating toward gas.
  
  At t >> τ: offset → 0
  Memory fully updated to post-collision configuration.
  System looks like a relaxed cluster (no offset).
  
  CDM PREDICTS: offset independent of time (DM halo is always at 
  galaxy position, regardless of when you observe).
  
  MEMORY KERNEL PREDICTS: offset is a DECAYING function of time 
  since merger.
  
  This is a clean, falsifiable distinction.
""")

# ============================================================
# 7. Summary
# ============================================================
print("=" * 72)
print("BRICK 3 (REVISED): RESULT")
print("=" * 72)

Sigma_14, fp = compute_memory_profile(14.0)
cm_14 = com(Sigma_14, m_main)
cb_14 = com(Sigma_14, m_bull)

print(f"""
  Memory kernel with τ = 14 Gyr (Hubble time):
  
  Main cluster:
    Center of effective mass:      {cm_14:.0f} kpc
    Offset from gas ({x_gas_main:.0f} kpc):      {cm_14 - x_gas_main:+.0f} kpc toward galaxies
    
  Bullet subcluster:
    Center of effective mass:      {cb_14:.0f} kpc
    Offset from gas ({x_gas_bullet:.0f} kpc):     {cb_14 - x_gas_bullet:+.0f} kpc toward galaxies
  
  The memory kernel produces the lensing-gas offset WITHOUT:
  - dark matter halos
  - sterile neutrinos  
  - any new particles
  
  It additionally predicts:
  - Offset magnitude decays with time since merger
  - Recently merged clusters show stronger offsets
  - Ancient mergers show healed (zero) offsets
  
  These predictions are testable with existing cluster catalogs.
  CDM makes no such time-dependent prediction.
""")

print("=" * 72)
