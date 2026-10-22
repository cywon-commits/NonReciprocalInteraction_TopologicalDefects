#!/usr/bin/env python3
"""Configs for stage B'' (section 9 of the request): P16-P20. Output in results/b3/. Prints the run order."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U0 = 0.04
base = dict(out_dir="results/b3", L=64, R=16, T=0.5, t_eq=200.0, t_run=8000.0, dts=0.06, dt=0.02,
            alpha=0.1, beta=0.0, D=0.3, B=0.12, r_sk=2.56, drive="none", u0=0.0)
pts = {}
# P16: d ln Gamma0 / dB by central difference +-0.01 at D = 0.3; common seed (paired comparison)
for B in (0.11, 0.12, 0.13):
    pts[f"P16_B{B:.2f}"] = dict(B=B, seed=8000)
# P18: P10 / P10b with twice the statistics, new seeds
pts["P18_P10"] = dict(drive="rot", u0=U0, rot_sign=1.0, beta=0.1, t_run=16000.0, seed=8101)
pts["P18_P10b"] = dict(drive="rot", u0=U0, rot_sign=-1.0, beta=0.1, t_run=16000.0, seed=8102)
# P19: alpha = beta = 0.05, P5-type shear and P10-type rotation drive, twice the statistics
pts["P19_shear_a0.05"] = dict(drive="shear", u0=U0, alpha=0.05, beta=0.05, t_run=16000.0, seed=8201)
pts["P19_rot_a0.05"] = dict(drive="rot", u0=U0, rot_sign=1.0, alpha=0.05, beta=0.05, t_run=16000.0, seed=8202)
# P20: single sinusoidal shear with period L/3 (k = 6 pi / L), u0 = 0.04/3 (same omega_max); two seeds each
for b, tag in ((0.0, "b0"), (0.1, "ba")):
    for sd in (1, 2):
        pts[f"P20_{tag}_s{sd}"] = dict(drive="shear", u0=U0 / 3, nper=3, beta=b, t_run=16000.0, seed=8300 + 10 * sd + int(b * 10))
# P17: B sensitivity at D = 0.2, 0.4 (B/D^2 = 4/3 centre), T = 0.5 and 0.55; common seed per (D, T)
for D, Bc, dB, rsk in ((0.2, 0.2 ** 2 * 4 / 3, 0.005, 4.0), (0.4, 0.4 ** 2 * 4 / 3, 0.01, 1.9)):
    for T, tr in ((0.5, 8000.0), (0.55, 4000.0)):
        for sgn, tag in ((-1, "m"), (1, "p")):
            pts[f"P17_D{D:.1f}_T{T:.2f}_B{tag}"] = dict(D=D, B=Bc + sgn * dB, r_sk=rsk, T=T, t_run=tr,
                                                      seed=8400 + int(D * 10) * 10 + int(T * 100) % 10)
os.makedirs(os.path.join(ROOT, "configs/b3"), exist_ok=True)
for name, p in pts.items():
    cfg = dict(base, run_id=name)
    cfg.update(p)
    json.dump(cfg, open(os.path.join(ROOT, f"configs/b3/{name}.json"), "w"), indent=1)
print(" ".join(f"configs/b3/{n}.json" for n in pts))
