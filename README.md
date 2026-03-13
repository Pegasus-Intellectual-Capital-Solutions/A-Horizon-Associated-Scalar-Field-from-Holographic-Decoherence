# Paper 6: A Horizon-Associated Scalar Field from Holographic Decoherence

**Full title:** A Horizon-Associated Scalar Field from Holographic Decoherence: Toward a Self-Contained Alternative to Sterile Neutrinos at Cluster Scales

**Author:** Charles Quarra, Pegasus Intellectual Capital Solutions

**Date:** March 2026 (v3)

## Abstract

Papers 1–5 of this series established a holographic decoherence framework deriving the cosmological acceleration scale a_u = cH₀/(2π) from horizon thermodynamics and fitting 171 SPARC galaxy rotation curves with zero free parameters. This paper demonstrates that the ~11 eV sterile neutrinos imported in Paper 5 are unnecessary: a Fourier analysis of the interaction complexity source function S(x) reveals an effective scalar field at m_eff ~ 1.2 × 10⁻²⁹ eV, emergent from the cluster correlation length. A nonlocal memory kernel K(x,x';t,t') reproduces the Bullet Cluster lensing–gas offset (~220 kpc) and predicts that this offset decays with time since merger — a prediction absent in ΛCDM.

## Key Results

1. **Brick 1:** m_eff = 1.23×10⁻²⁹ eV from composite stellar profile (Figure 1)
2. **Brick 2:** Ω_φ = 4×10⁻¹⁰, seven orders of magnitude below eRASS1 bound
3. **Brick 3:** Memory kernel reproduces Bullet Cluster offset +228 kpc (Figure 2)
4. **Brick 4:** CMB structural compatibility (Paper 7)
5. **Observational test:** Seven-cluster dataset (Table I, Figures 3–4)
6. **Geometry-normalized analysis:** Pearson r = −0.14 (N=6) — consistent with prediction, not yet decisive

## Files

### Manuscript
- `Paper6_Quarra_v3_with_figures.docx` — Complete manuscript with all figures embedded

### Figures
- `Fig1_Fourier_spectrum.{png,pdf}` — Power spectrum of S(x) showing emergent mass scale
- `Fig2_offset_decay.{png,pdf}` — Bullet Cluster offset decay prediction
- `Fig3_offset_vs_TSP.{png,pdf}` — Seven-cluster raw offset vs TSP
- `Fig4_geometry_normalized.{png,pdf}` — Geometry-normalized residual analysis

### Computation Scripts
- `paper6_fourier.py` — Fourier analysis of S(x), effective mass derivation
- `paper6_eRASS1.py` — eRASS1 constraint evasion calculation
- `paper6_bullet.py` — Bullet Cluster memory kernel offset calculation
- `fig1_fourier.py` — Figure 1 generation
- `fig2_offset.py` — Figure 2 generation
- `fig3_real_data.py` — Figure 3 generation (seven-cluster dataset)
- `fig4_geometry_normalized.py` — Figure 4 generation
- `geometry_normalized_analysis.py` — Full geometry-normalized residual analysis

### Data
- `paper6_fourier_results.json` — Fourier analysis numerical results
- `geometry_normalized_results.json` — Geometry-normalized cluster data and statistics

## Observational Dataset (Table I)

| Cluster | TSP (Gyr) | Offset (kpc) | Type | Source |
|---------|-----------|-------------|------|--------|
| Abell 2146 | 0.24 | 40 ± 30 | galaxy-gas | White+ 2015 |
| Bullet | 0.18 | 220 ± 40 | lensing-gas | Springel & Farrar 2007 |
| Abell 2345 | 0.35 | 130 ± 60 | relic-center | Boschin+ 2010 |
| ZwCl 0008 | 0.76 | 100 ± 50 | lensing-gas | Golovich+ 2017 |
| Sausage (N) | 0.90 | 890 ± 130 | galaxy-gas | Dawson+ 2015 |
| El Gordo | 0.91 | 200 ± 60 | lensing-gas | Ng+ 2015 |
| Musket Ball | 1.10 | 150 ± 50 | lensing-gas | Dawson 2013 |

## Dependencies

Python 3.8+, numpy, scipy, matplotlib

## Paper Series

1. Emergent Gravity from Horizon-Induced Decoherence
2. Holographic Lower Bound on Primordial Boundary Entropy
3. Galactic Rotation Curves from the de Sitter Horizon (Zenodo: 10.5281/zenodo.18868563)
4. An Operational Framework for Emergent Gravity (Zenodo: 10.5281/zenodo.18806314)
5. Cluster-Scale Dynamics from de Sitter Entanglement Entropy
6. **This paper**
7. CMB Boltzmann Code Integration (forthcoming)
