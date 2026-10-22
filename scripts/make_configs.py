#!/usr/bin/env python3
"""Write the configs for task 1B (P1-P9), the dt/2 check and the 1A temperature ladder."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_STAR, U0, L, R, T_RUN, DTS = 0.5, 0.04, 64, 16, 8000.0, 0.06
base = dict(out_dir="results/main", L=L, R=R, T=T_STAR, t_eq=200.0, t_run=T_RUN, dts=DTS, dt=0.02)
points = {
    "P1": dict(alpha=0.1, beta=0.0, drive="shear", u0=U0),
    "P2": dict(alpha=0.1, beta=0.0, drive="shear", u0=-U0),
    "P3": dict(alpha=0.1, beta=0.0, drive="shear", u0=U0 / 2),
    "P4": dict(alpha=0.1, beta=0.2, drive="shear", u0=U0),
    "P5": dict(alpha=0.1, beta=0.1, drive="shear", u0=U0),
    "P6": dict(alpha=0.1, beta=0.0, drive="uniform", u0=U0),
    "P7": dict(alpha=0.1, beta=0.0, drive="none", u0=0.0),
    "P8": dict(alpha=0.3, beta=0.0, drive="shear", u0=U0),
    "P9": dict(alpha=0.3, beta=0.6, drive="shear", u0=U0),
}
os.makedirs(os.path.join(ROOT, "configs/main"), exist_ok=True)
for i, (name, p) in enumerate(points.items()):
    cfg = dict(base, run_id=name, seed=1000 + i, **p)
    json.dump(cfg, open(os.path.join(ROOT, f"configs/main/{name}.json"), "w"), indent=1)
# dt/2 check of the main point
cfg = dict(base, run_id="P1_dt0.01", seed=1100, dt=0.01, **points["P1"])
json.dump(cfg, open(os.path.join(ROOT, "configs/main/P1_dt0.01.json"), "w"), indent=1)
# 1A ladder (undriven)
os.makedirs(os.path.join(ROOT, "configs/ladder"), exist_ok=True)
for T, trun in [(0.40, 6000.0), (0.45, 6000.0), (0.55, 3000.0), (0.60, 2000.0)]:
    cfg = dict(base, out_dir="results/ladder", run_id=f"ladder_T{T:.2f}", seed=int(T * 10000), T=T, t_run=trun,
               alpha=0.1, beta=0.0, drive="none", u0=0.0)
    json.dump(cfg, open(os.path.join(ROOT, f"configs/ladder/T{T:.2f}.json"), "w"), indent=1)
