#!/usr/bin/env python3
"""Configs for the follow-up request (stage B', section 8): P10-P15. Output in results/b2/."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U0 = 0.04
base = dict(out_dir="results/b2", L=64, R=16, T=0.5, t_eq=200.0, t_run=8000.0, dts=0.06, dt=0.02,
            alpha=0.1, beta=0.1, D=0.3, B=0.12, r_sk=2.56)
pts = {
    # rotation term only: u = 0, D_t = d_t + (omega/2) z x, omega = profile of the P5 shear
    "P10": dict(drive="rot", u0=U0, rot_sign=1.0),
    "P10b": dict(drive="rot", u0=U0, rot_sign=-1.0),
    # L dependence at fixed omega
    "P13_L128": dict(beta=0.0, drive="shear", u0=2 * U0, L=128, t_run=2500.0),
    "P13_L32": dict(beta=0.0, drive="shear", u0=U0 / 2, L=32, R=64),
    # undriven pilots for the D values of P11 (rates at T = 0.5)
    "pilot_D0.2": dict(drive="none", u0=0.0, D=0.2, B=0.2 ** 2 * 4 / 3, r_sk=4.0, t_run=1500.0),
    "pilot_D0.4": dict(drive="none", u0=0.0, D=0.4, B=0.4 ** 2 * 4 / 3, r_sk=1.9, t_run=1500.0),
    # omega / psi separation: u_x = u1 sin ky + u3 sin 3ky, u3 = -u1/3
    "P14_b0": dict(beta=0.0, drive="shear3", u0=U0, u3=-U0 / 3, t_run=16000.0),
    "P14_ba": dict(beta=0.1, drive="shear3", u0=U0, u3=-U0 / 3, t_run=16000.0),
    # DMI dependence of the beta = alpha residual (B/D^2 = 4/3 as for D = 0.3, B = 0.12)
    "P11_D0.4": dict(drive="shear", u0=U0, D=0.4, B=0.4 ** 2 * 4 / 3, r_sk=1.9),
    "P11_D0.2": dict(drive="shear", u0=U0, D=0.2, B=0.2 ** 2 * 4 / 3, r_sk=4.0),
    # alpha dependence of the beta = alpha residual
    "P12_a0.05": dict(drive="shear", u0=U0, alpha=0.05, beta=0.05),
    "P12_a0.2": dict(drive="shear", u0=U0, alpha=0.2, beta=0.2),
}
# barrier ladders at B = 0.10, 0.14 (undriven)
for B, rsk in ((0.10, 3.43), (0.14, 1.96)):
    for T, tr in ((0.45, 8000.0), (0.50, 4000.0), (0.55, 3000.0)):
        pts[f"P15_B{B:.2f}_T{T:.2f}"] = dict(drive="none", u0=0.0, beta=0.0, B=B, r_sk=rsk, T=T, t_run=tr)
os.makedirs(os.path.join(ROOT, "configs/b2"), exist_ok=True)
for i, (name, p) in enumerate(pts.items()):
    cfg = dict(base, run_id=name, seed=7000 + i)
    cfg.update(p)
    json.dump(cfg, open(os.path.join(ROOT, f"configs/b2/{name}.json"), "w"), indent=1)
print(" ".join(f"configs/b2/{n}.json" for n in pts))
