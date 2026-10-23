# Skyrmion nucleation bias under non-reciprocal (shear) drive

Code and data for the calculation request [`../docs/compute_request_stageB.md`](../docs/compute_request_stageB.md)
"비상반 구동 아래 스커미온 핵생성 편향 (분기 논문 A, 단계 B)".
Reports: [`REPORT.md`](REPORT.md) (stage B, sections 1-7 of the request) and [`REPORT_B2.md`](REPORT_B2.md)
(stage B', section 8: P10-P15) and [`REPORT_B3.md`](REPORT_B3.md) (stage B'', section 9: P16-P20) and [`REPORT_B4.md`](REPORT_B4.md)
(stage B''', section 10: re-analysis, `scripts/b4_analysis.py`) and [`REPORT_C.md`](REPORT_C.md)
(stage C, section 11: F_pop(T); configs `scripts/make_configs_c.py`, analysis `scripts/c_analysis.py`);
figures are in `results/figures/`.
All commands below are run from this `stageB/` directory.

## Model and conventions (as in the request, unchanged)

- Square lattice, a = 1, periodic, |S_i| = 1, J = γ = μ = k_B = 1.
- E = −J Σ S_i·S_j − Σ D_ij·(S_i × S_j) − B Σ S^z − K Σ (S^z)², interfacial DMI D_ij = D (ẑ × ê_ij).
- Gilbert + Zhang–Li: dS/dt = −S×(H+h) + α S×dS/dt − P(u·∇)S + β S×(u·∇)S, (u·∇)S by central
  differences with u at site i. Integrated in the LL form of the request with Stratonovich Heun
  (renormalisation after predictor and corrector, same noise), ⟨h h⟩ = 2αT δδδ. Sign convention
  dS/dt = −S×H kept.
- Berg–Lüscher charge, one diagonal (i → i+x̂+ŷ) for detection, two-diagonal average for symmetric means.
- With B > 0 the favoured (core −z, Néel) skyrmion has **Q = −1** in this convention (computed, `results/validation/V34.json`).

Implementation: Numba on CPU (no GPU was available; 4 cores). Each run is one single-threaded process;
the 16 replicas of a run are the batch dimension.

| file | content |
| --- | --- |
| `skynuc/model.py` | Hamiltonian, Heun integrator (ziggurat Gaussian noise, splitmix64 streams per lattice row), charge |
| `skynuc/topo.py` | triangle solid angles, topological-jump detection, skyrmion cluster finder |
| `skynuc/nucleation.py` | driven run: events (`events.csv`), tracked objects (`objects.csv`), exposures (`summary.json`) |
| `skynuc/analysis.py` | rates, Poisson-weighted LSQ / Poisson GLM fits, jackknife errors, halves ratio R |
| `skynuc/validate_v12.py`, `skynuc/mc.py` | V1 and V2 (independent Metropolis MC) |
| `skynuc/single.py`, `skynuc/lattice_thiele.py` | isolated skyrmion: relaxation, R, Q, 𝒟, V3/V4, discretisation diagnosis |
| `skynuc/lifetime.py` | seeded-skyrmion lifetimes (calibration) |
| `skynuc/plots.py` | figures |

## Re-running

```bash
pip install numpy scipy numba matplotlib

# V1-V4 (four single-threaded processes)
scripts/run_validation.sh
python3 -m skynuc.single v3scan           # V3 deficit vs skyrmion size
python3 -m skynuc.lattice_thiele          # lattice-consistent Thiele comparison

# configs for P1-P9, dt/2 check, T ladder, second seeds, 1C dependence runs
python3 scripts/make_configs.py
scripts/run_batch.py -j 4 configs/main/P1.json configs/main/P2.json ...   # any configs, N at a time
scripts/run_batch.py -j 4 configs/ladder/*.json
python3 -m skynuc.nucleation configs/main/P1.json --resume                 # resume from state.npz

# stage B' (P10-P15): python3 scripts/make_configs_b2.py prints the config list
scripts/run_batch.py -j 4 $(python3 scripts/make_configs_b2.py)

# stage B'' (P16-P20): configs from scripts/make_configs_b3.py; analysis scripts/b3_analysis.py
scripts/run_batch.py -j 4 $(python3 scripts/make_configs_b3.py)

# analysis and figures
python3 scripts/collect.py                # -> results/summary_table.csv, results/analysis/*.json
python3 scripts/quick_table.py --tau 5,10,20 results/main/P1
python3 -m skynuc.plots validation summary
```

A config is a JSON dict of the keys in `skynuc/nucleation.py:DEFAULTS`; `key=value` arguments override it.

## Output formats

- `events.csv`: `run_id, replica, t, x, y, dQ, kind, flag, dq_w, n_jumps, obj_id, lifetime` — every
  topological jump (triangle solid angle jumping by 4π between samples), grouped within 2.5 sites.
  `kind` ∈ nuc / col / anti_nuc / anti_col / pair (ΔQ = 0) / multi; `flag` = 1 if several jumps
  overlap or the 3×3 window change Δq_w differs from ΔQ by > 0.5.
- `objects.csv`: one row per tracked skyrmion object (birth, death, positions, lifetime, censoring).
  **Primary nucleation/collapse definition** (see REPORT.md §2): nucleation = birth of an object that
  lives ≥ τ_c; collapse = death of an object that lived ≥ τ_c.
- `summary.json`: parameters, band edges and ω per band, raw event counts per band, exposures
  (nucleation area×time with 2R exclusion, skyrmion-time of established objects per τ_c),
  per-(replica, time-block, band) exposures for jackknife, drift velocities and density by band.
- `results/summary_table.csv`: per run and τ_c ∈ {5, 10, 20}: N, rates, slopes s_nuc, s_col
  (WLS with Poisson errors, GLM with jackknife errors), ln R with errors.
