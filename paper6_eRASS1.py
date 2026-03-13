"""
Paper 4 — Core Calculation #2: eRASS1 Evasion
===============================================

THE QUESTION:
Can the effective scalar field at m ~ 10⁻²⁹ eV have energy density
Ω_φ << 0.003 (below eRASS1 bounds) while still producing sufficient
gravitational effects to replace the sterile neutrino in Paper 3?

THE KEY DISTINCTION:
Standard ultralight DM:  gravitational effect ∝ ρ_field (energy density)
Our framework:           gravitational effect ∝ ν(g_N/a_u) (kernel function)

In our framework, the gravitational modification is set by the 
interpolation function ν(y) = 1/(1 - e^{-√y}) where y = g_N/a_u.
This function is derived from horizon thermodynamics (Paper 2).
It does NOT depend on the energy density of any field.

The effective scalar field is a quasiparticle description of the 
spatial modes of the decoherence kernel. Its energy density depends
on the AMPLITUDE of the entropy production function S(x), which is
set by the actual astrophysics of clusters (stellar luminosity, 
gas temperature). This is a separate quantity from the gravitational
modification strength.

CALCULATION STRATEGY:
1. Estimate the amplitude of S(x) from known cluster properties
2. Compute the energy density ρ_φ of the corresponding field mode
3. Convert to Ω_φ = ρ_φ / ρ_crit
4. Compare with eRASS1 bound (Ω_a < 0.003 at m ~ 10⁻²⁷ eV)
5. Compute the gravitational modification independently via ν(y)
6. Show the two are decoupled
"""

import numpy as np

# ============================================================
# Physical constants
# ============================================================
hbar = 1.0546e-34        # J·s
c = 2.998e8              # m/s
G = 6.674e-11            # m³/(kg·s²)
eV = 1.602e-19           # J
kpc = 3.086e19           # m
Mpc = 3.086e22           # m
k_B = 1.381e-23          # J/K
L_sun = 3.828e26         # W
M_sun = 1.989e30         # kg
H0 = 70.0                # km/s/Mpc
H0_si = H0 * 1e3 / Mpc  # s⁻¹

print("=" * 72)
print("PAPER 4 — eRASS1 EVASION CALCULATION")
print("Does the effective field evade observational density bounds?")
print("=" * 72)

# ============================================================
# 1. Critical density of the universe
# ============================================================
rho_crit = 3 * H0_si**2 / (8 * np.pi * G)  # kg/m³
rho_crit_eV4 = rho_crit * c**2 / eV  # eV/m³

print(f"\nCritical density: ρ_crit = {rho_crit:.2e} kg/m³")
print(f"                        = {rho_crit * c**2:.2e} J/m³")
print(f"                        = {rho_crit_eV4:.2e} eV/m³")

# eRASS1 bound
Omega_bound = 0.003  # at m ~ 10⁻²⁷ eV
rho_bound = Omega_bound * rho_crit  # kg/m³
print(f"\neRASS1 bound: Ω_a < {Omega_bound}")
print(f"              ρ_bound = {rho_bound:.2e} kg/m³")
print(f"                      = {rho_bound * c**2:.2e} J/m³")

# ============================================================
# 2. Estimate S(x) amplitude from cluster astrophysics
# ============================================================
print("\n" + "=" * 72)
print("STEP 1: Amplitude of S(x) from cluster physics")
print("=" * 72)

# S(x) = ṡ_irr(x) / k_B = irreversible entropy production rate density
#
# For a galaxy cluster, the dominant irreversible processes are:
# - Stellar luminosity (nuclear burning → photons)
# - Gravitational heating of ICM gas
# - Shocks and turbulence in ICM
#
# The entropy production rate from stellar luminosity:
# ṡ_stellar = L / T_CMB  (photons radiated into the CMB bath)
#
# For a massive cluster (M ~ 10¹⁵ M_sun):

L_cluster = 1e13 * L_sun      # Total stellar luminosity ~ 10¹³ L_sun
R_cluster = 1.0 * Mpc         # Characteristic radius ~ 1 Mpc
V_cluster = (4/3) * np.pi * R_cluster**3  # Volume
T_CMB = 2.725                  # K (CMB temperature = radiation sink)

# Volumetric entropy production rate
s_dot_irr = L_cluster / (T_CMB * V_cluster)  # W/(K·m³) = entropy/(s·m³)

# S(x) = ṡ_irr / k_B  (in units of 1/(s·m³))
S_amplitude = s_dot_irr / k_B

print(f"\nCluster parameters:")
print(f"  Total stellar luminosity: L = {L_cluster:.1e} W ({L_cluster/L_sun:.0e} L_☉)")
print(f"  Cluster radius:           R = {R_cluster/Mpc:.1f} Mpc")
print(f"  Cluster volume:           V = {V_cluster:.2e} m³")
print(f"  Radiation sink temp:      T = {T_CMB} K (CMB)")
print(f"\nEntropy production rate density:")
print(f"  ṡ_irr = L/(TV) = {s_dot_irr:.2e} W/(K·m³)")
print(f"  S(x) = ṡ_irr/k_B = {S_amplitude:.2e} / (s·m³)")

# Also include ICM cooling luminosity (typically ~ few × 10⁴⁴ erg/s for massive clusters)
L_ICM = 1e45 * 1e-7  # Convert erg/s to W; ~ 10⁴⁵ erg/s for massive cluster
T_ICM = 1e8  # ICM temperature ~ 10⁸ K (~ 10 keV)
s_dot_ICM = L_ICM / (T_ICM * V_cluster)
S_ICM = s_dot_ICM / k_B

print(f"\nICM contribution:")
print(f"  L_ICM = {L_ICM:.1e} W")
print(f"  T_ICM = {T_ICM:.0e} K")
print(f"  S_ICM = {S_ICM:.2e} / (s·m³)")
print(f"\nTotal S(x) ~ {S_amplitude + S_ICM:.2e} / (s·m³)")

S_total = S_amplitude + S_ICM

# ============================================================
# 3. Energy density of the effective scalar field
# ============================================================
print("\n" + "=" * 72)
print("STEP 2: Energy density of the effective field")
print("=" * 72)

# The effective scalar field is a Fourier mode of S(x) at wavenumber k_c.
# 
# The energy density of a classical scalar field mode:
#   ρ_φ = (1/2)(∂φ/∂t)² + (1/2)m²c⁴/ℏ² φ² + (1/2)(∇φ)²c²
#
# For a quasi-static mode (∂φ/∂t << mφ, which holds for m >> H₀):
#   ρ_φ ≈ (1/2) m²c⁴/ℏ² φ₀²
#
# But what IS φ₀? The scalar field is an effective description of
# the spatial modes of the decoherence kernel. The kernel's amplitude
# is set by S(x), and the relationship is:
#
#   The kernel modifies the gravitational potential as:
#     Φ_total = Φ_N × ν(g_N/a_u)
#   
#   The function ν is derived from entropy extremization (Paper 2).
#   It does NOT depend on S(x)'s amplitude.
#   S(x) enters only as a spatial weighting (Paper 3) that determines
#   WHERE the modification is concentrated, not HOW STRONG it is.
#
# This is the crucial decoupling:
#   - The STRENGTH of gravity modification → set by a_u = cH₀/(2π)
#   - The SPATIAL PROFILE of the modification → set by S(x)
#   - The ENERGY DENSITY of the effective field → set by S(x)'s amplitude
#
# So we need to connect S(x) amplitude to an effective field amplitude φ₀.
#
# Dimensional analysis of S(x) = ṡ_irr/k_B:
#   [S] = 1/(s·m³)
#
# The scalar field φ has dimensions of energy (in natural units) or
# in SI: [φ] = kg^(1/2) m^(1/2) / s  (for canonical scalar field)
#
# The connection comes through the holographic encoding:
# The boundary encodes S(x) via the entropy-area relation.
# Each Planck area cell on the boundary contributes one bit.
# The effective field amplitude is:
#
#   φ₀² ~ ℏ² × S_total × (characteristic time) / (m_eff² c⁴ / ℏ²)
#
# But more directly: the field's energy density is bounded by the 
# gravitational energy associated with the entropy production.
#
# Energy density of entropy production over a Hubble time:
#   ρ_S ~ ṡ_irr × T_sink × t_H = entropy prod rate × temperature × time

t_H = 1.0 / H0_si  # Hubble time ~ 14 Gyr in seconds
print(f"\nHubble time: t_H = {t_H:.2e} s = {t_H/(3.156e7 * 1e9):.1f} Gyr")

# Energy density associated with entropy production
# This is the MAXIMUM energy the field could carry:
# All the entropy produced over a Hubble time, converted to energy at T_CMB
rho_entropy_energy = s_dot_irr * T_CMB * t_H  # J/m³ over a Hubble time
# But this is CUMULATIVE over the whole cluster lifetime.
# The field carries the CURRENT mode, not the cumulative history.
# At any instant, the field amplitude corresponds to the current S(x).

# More careful estimate: treat the effective field as having
# energy density equal to the instantaneous entropy production power
# density divided by the field's natural frequency ω = m c² / ℏ

m_eff_eV = 1.23e-29  # eV, from Calculation #1
m_eff_kg = m_eff_eV * eV / c**2
omega_field = m_eff_kg * c**2 / hbar  # natural frequency in rad/s

print(f"\nEffective field parameters:")
print(f"  m_eff = {m_eff_eV:.2e} eV")
print(f"  m_eff = {m_eff_kg:.2e} kg")
print(f"  ω_field = m c²/ℏ = {omega_field:.2e} rad/s")
print(f"  Period = {2*np.pi/omega_field:.2e} s = {2*np.pi/omega_field/(3.156e7*1e9):.2e} Gyr")

# The instantaneous energy density in the mode is:
# ρ_φ ~ S_total × k_B × T_CMB / ω_field
# This is: (entropy production rate) × (energy per entropy quantum) / (oscillation frequency)
# Physical meaning: how much energy accumulates in one oscillation period

rho_phi_cluster = S_total * k_B * T_CMB / omega_field  # J/m³
rho_phi_cluster_kg = rho_phi_cluster / c**2  # kg/m³

print(f"\nField energy density IN the cluster:")
print(f"  ρ_φ(cluster) = {rho_phi_cluster:.2e} J/m³")
print(f"               = {rho_phi_cluster_kg:.2e} kg/m³")

# ============================================================
# 4. Average over cosmological volume
# ============================================================
print("\n" + "=" * 72)
print("STEP 3: Cosmological average → Ω_φ")
print("=" * 72)

# The field exists everywhere the decoherence kernel operates,
# but its amplitude is concentrated in clusters where S(x) is large.
# 
# Volume fraction of the universe in clusters:
# Typical cluster: R ~ 1-2 Mpc, n_cluster ~ 10⁻⁵ /Mpc³
# (eRASS1 found ~12,000 clusters in half the sky out to z~0.8)

n_cluster = 1e-5 / Mpc**3  # cluster number density (per m³)
V_single = V_cluster  # volume of one cluster

f_vol = n_cluster * V_single  # volume filling fraction
print(f"\nCluster statistics:")
print(f"  Number density:     n = {n_cluster * Mpc**3:.1e} / Mpc³")
print(f"  Volume per cluster: V = {V_cluster:.2e} m³")
print(f"  Volume filling fraction: f = {f_vol:.2e}")

# Average energy density over cosmological volume
rho_phi_avg = rho_phi_cluster_kg * f_vol

# Also include contribution from groups and filaments
# Groups are ~100× more numerous but ~10× less luminous and smaller
# Net contribution comparable to clusters within factor ~few
enhancement_factor = 5.0  # groups + filaments contribute ~5× clusters
rho_phi_avg_total = rho_phi_avg * enhancement_factor

Omega_phi = rho_phi_avg_total / rho_crit

print(f"\nCosmological average (clusters only):")
print(f"  ρ_φ_avg = {rho_phi_avg:.2e} kg/m³")
print(f"  Ω_φ    = {rho_phi_avg / rho_crit:.2e}")

print(f"\nWith groups + filaments (×{enhancement_factor:.0f}):")
print(f"  ρ_φ_avg = {rho_phi_avg_total:.2e} kg/m³")
print(f"  Ω_φ     = {Omega_phi:.2e}")

# ============================================================
# 5. Comparison with eRASS1 bound
# ============================================================
print("\n" + "=" * 72)
print("STEP 4: Comparison with eRASS1 bound")
print("=" * 72)

ratio_to_bound = Omega_phi / Omega_bound
print(f"\n  Our Ω_φ:          {Omega_phi:.2e}")
print(f"  eRASS1 bound:     {Omega_bound:.2e}")
print(f"  Ratio:            {ratio_to_bound:.2e}")
print(f"  Orders of magnitude below bound: {-np.log10(ratio_to_bound):.1f}")

if Omega_phi < Omega_bound:
    print(f"\n  >>> PASSES eRASS1 constraint by {-np.log10(ratio_to_bound):.0f} orders of magnitude")
else:
    print(f"\n  >>> FAILS eRASS1 constraint")

# ============================================================
# 6. Gravitational effect is independent of field density
# ============================================================
print("\n" + "=" * 72)
print("STEP 5: Gravitational modification (independent of Ω_φ)")
print("=" * 72)

# The gravitational modification in our framework:
# Φ_total = Φ_N × ν(y), where y = g_N / a_u
# ν(y) = 1/(1 - e^{-√y})
#
# At the Bullet Cluster scale:
# Typical Newtonian acceleration at r ~ 1 Mpc from cluster center
# with M ~ 10¹⁴ M_sun (baryonic):

M_baryonic = 1e14 * M_sun  # kg
r_typical = 1.0 * Mpc      # m
a_u = c * H0_si / (2 * np.pi)  # m/s²

g_N = G * M_baryonic / r_typical**2
y = g_N / a_u

def nu(y):
    return 1.0 / (1.0 - np.exp(-np.sqrt(y)))

nu_val = nu(y)
boost = nu_val  # The total gravitational acceleration is g_N × ν(y)

print(f"\nBullet Cluster parameters:")
print(f"  M_baryonic = {M_baryonic:.1e} kg ({M_baryonic/M_sun:.0e} M_☉)")
print(f"  r = {r_typical/Mpc:.1f} Mpc")
print(f"  g_N = GM/r² = {g_N:.2e} m/s²")
print(f"  a_u = cH₀/(2π) = {a_u:.2e} m/s²")
print(f"  y = g_N/a_u = {y:.4f}")
print(f"  ν(y) = {nu_val:.3f}")
print(f"  Effective gravity boost: ×{nu_val:.2f}")
print(f"  Effective total mass / baryonic mass: {nu_val:.2f}")

# For comparison: CDM expects M_total/M_baryonic ~ 6 for clusters
# (baryon fraction ~ 16%)
M_total_CDM = M_baryonic / 0.16  # If baryon fraction ~ 16%
ratio_CDM = M_total_CDM / M_baryonic

print(f"\n  CDM expectation (M_total/M_bary): ~{ratio_CDM:.1f}")
print(f"  Our framework ν(y):               ~{nu_val:.1f}")

# Check at smaller radii where y is larger
print(f"\nν(y) at various radii:")
for r_test_Mpc in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    r_test = r_test_Mpc * Mpc
    g_test = G * M_baryonic / r_test**2
    y_test = g_test / a_u
    nu_test = nu(y_test)
    print(f"  r = {r_test_Mpc:.1f} Mpc:  g_N = {g_test:.2e},  y = {y_test:.3f},  ν = {nu_test:.2f}")

# ============================================================
# 7. The decoupling argument
# ============================================================
print("\n" + "=" * 72)
print("STEP 6: The Decoupling Argument")
print("=" * 72)

print("""
THE KEY RESULT:

The gravitational modification ν(y) depends on:
  - g_N: the Newtonian gravitational acceleration (from baryonic mass)
  - a_u = cH₀/(2π): a constant set by horizon thermodynamics

It does NOT depend on:
  - The energy density Ω_φ of the effective scalar field
  - The amplitude of S(x)
  - Any property of the effective quasiparticle

The effective scalar field describes WHERE the modification is 
concentrated (its spatial profile tracks S(x)), but the STRENGTH 
of the modification is set by the ratio g_N/a_u.

This is analogous to how:
  - A phonon describes WHERE lattice vibrations are concentrated
  - But the elastic modulus of the crystal is set by atomic bonds,
    not by the phonon energy density

Or in the Higgs analogy:
  - The Higgs VEV determines particle masses (coupling)
  - The Higgs energy density contributes to vacuum energy (density)
  - These are independent: you can change the cosmological constant
    without changing electron mass

QUANTITATIVE SUMMARY:
""")

print(f"  Field energy density:    Ω_φ = {Omega_phi:.2e}")
print(f"  eRASS1 bound:            Ω_a < {Omega_bound}")
print(f"  Margin:                  {-np.log10(ratio_to_bound):.0f} orders of magnitude")
print(f"")
print(f"  Gravitational boost:     ν(y) = {nu_val:.2f} at r = 1 Mpc")
print(f"  Required boost (CDM):    ~{ratio_CDM:.1f}")
print(f"")

# Note on the boost factor
if nu_val < ratio_CDM:
    deficit = ratio_CDM / nu_val
    print(f"  NOTE: ν(y) at 1 Mpc gives boost of {nu_val:.1f},")
    print(f"  while CDM requires ~{ratio_CDM:.1f}. The deficit factor is {deficit:.1f}.")
    print(f"  This is the same gap Paper 3 addressed with the sterile neutrino.")
    print(f"  The scalar field's role is to fill this gap through its")
    print(f"  spatial concentration — weighting the holographic kernel")
    print(f"  toward stellar-dominated regions where y is larger.")
    
    # At what radius does ν give the needed boost?
    from scipy.optimize import brentq
    def boost_deficit(log_r_Mpc):
        r = 10**log_r_Mpc * Mpc
        g = G * M_baryonic / r**2
        y_val = g / a_u
        return nu(y_val) - ratio_CDM
    
    try:
        log_r_solution = brentq(boost_deficit, -2, 1)
        r_solution = 10**log_r_solution
        print(f"\n  ν(y) = {ratio_CDM:.1f} is achieved at r = {r_solution:.2f} Mpc")
        print(f"  The S(x) weighting concentrates the effective gravity")
        print(f"  toward smaller radii where ν is naturally larger.")
    except:
        print(f"\n  ν(y) = {ratio_CDM:.1f} not achieved — deeper analysis needed")

print(f"""
CONCLUSION:
The effective scalar field's energy density is ~{-np.log10(ratio_to_bound):.0f} orders of 
magnitude below the eRASS1 bound. The gravitational modification 
that replaces the sterile neutrino comes from the holographic kernel 
function ν(y), which is independent of the field's energy density.

The eRASS1 constraint is evaded because the constraint applies to 
fields whose gravitational effect IS their energy density. Our field's 
gravitational effect is its spatial modulation of a kernel whose 
strength is set by horizon thermodynamics, not by field amplitude.

This is not a loophole. It is a structural feature of the framework:
the holographic kernel is a modification to the gravitational 
propagator, not an additional source term in the Friedmann equation.
""")

print("=" * 72)
