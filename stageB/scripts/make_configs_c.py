#!/usr/bin/env python3
"""Configs for stage C (request section 11): F_pop(T). Output in results/c/. Prints the run order (C3, C2, C4, C1).

Per temperature: rotation-only drive +-omega (P10/P10b type, beta = alpha = 0.1) and an undriven B = 0.11 / 0.13 pair
(common seed). Undriven B = 0.12 points at T = 0.65, 0.70, 0.75 complete the existing ladder (0.40-0.60) for d/dT.
Run lengths aim at >= 2000 committed nucleations per drive sign (tau_c = 10), from the undriven rates of the ladder.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U0 = 0.04
base = dict(out_dir="results/c", L=64, R=16, t_eq=200.0, dts=0.06, dt=0.02, alpha=0.1, beta=0.1, D=0.3, B=0.12,
            r_sk=2.56, drive="none", u0=0.0)
TRUN = {0.60: 2500.0, 0.55: 4000.0, 0.70: 1500.0, 0.45: 33000.0}
pts = {}
for T in (0.60, 0.55, 0.70, 0.45):  # priority order C3, C2, C4, C1
    tag = f"T{T:.2f}"
    tr = TRUN[T]; s = int(T * 1000)
    pts[f"C_{tag}_rotp"] = dict(T=T, t_run=tr, drive="rot", u0=U0, rot_sign=1.0, seed=9000 + s)
    pts[f"C_{tag}_rotm"] = dict(T=T, t_run=tr, drive="rot", u0=U0, rot_sign=-1.0, seed=9001 + s)
    pts[f"C_{tag}_B0.11"] = dict(T=T, t_run=tr, B=0.11, beta=0.0, seed=9500 + s)
    pts[f"C_{tag}_B0.13"] = dict(T=T, t_run=tr, B=0.13, beta=0.0, seed=9500 + s)
    if T == 0.70:
        for Tb, trb in ((0.65, 2000.0), (0.70, 1500.0), (0.75, 1500.0)):
            pts[f"C_ladder_T{Tb:.2f}_B0.12"] = dict(T=Tb, t_run=trb, beta=0.0, seed=9800 + int(Tb * 100))
# extra statistics at T = 0.55, 0.60 (the predicted signal falls with T; with one +-omega pair F_pop errors were
# 0.3-0.35): three more seed pairs each, run after the C1 (T = 0.45) runs
EXTRA = []
for T in (0.55, 0.60):
    for k in (2, 3, 4):
        for sgn, nm in ((1.0, "rotp"), (-1.0, "rotm")):
            name = f"C_T{T:.2f}_{nm}_s{k}"
            pts[name] = dict(T=T, t_run=TRUN[T], drive="rot", u0=U0, rot_sign=sgn,
                             seed=9000 + int(T * 1000) + 10 * k + (sgn < 0))
            EXTRA.append(name)
os.makedirs(os.path.join(ROOT, "configs/c"), exist_ok=True)
for name, p in pts.items():
    cfg = dict(base, run_id=name)
    cfg.update(p)
    json.dump(cfg, open(os.path.join(ROOT, f"configs/c/{name}.json"), "w"), indent=1)
print(" ".join(f"configs/c/{n}.json" for n in pts))
