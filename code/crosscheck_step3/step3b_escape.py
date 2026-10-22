"""Step 3b: pair escape probability.

Relative coordinate of a skyrmion–antiskyrmion pair (doc 5.2–5.3):
  dr = v x_hat dt + sqrt(2 D_r) dW,   v / D_r = 1 / l_u,
absorbed (annihilation) at |r| = a.  Start at |r| = R0 with random angle.
Escape: reaching |r| > R_out = 40 l_u (drift then carries it away).
Exact-increment Gaussian steps with adaptive dt (step ~ 0.05 * distance to absorber).
Predictions:
  leading log      P = ln(R0/a) / ln(l_u/a)
  angle-averaged   P = 1 - K0(R0/2l_u) / K0(a/2l_u)
"""
import numpy as np, json
from scipy.special import k0

def escape(R0, lu, a=1.0, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    Dr = 1.0; v = Dr / lu
    th = rng.uniform(0, 2 * np.pi, n)
    r = np.stack([R0 * np.cos(th), R0 * np.sin(th)], 1)
    alive = np.ones(n, bool); esc = np.zeros(n, bool)
    Rout = 40 * lu
    while alive.any():
        idx = np.where(alive)[0]
        p = r[idx]
        d = np.linalg.norm(p, axis=1)
        h = np.maximum(0.05 * (d - a), 0.002)            # spatial step scale
        dt = h ** 2 / (2 * Dr)
        dt = np.minimum(dt, 0.05 * lu / v)                # keep drift step < 5% of l_u
        p = p + np.stack([v * dt, np.zeros_like(dt)], 1) + rng.normal(size=p.shape) * np.sqrt(2 * Dr * dt)[:, None]
        r[idx] = p
        d = np.linalg.norm(p, axis=1)
        dead = d < a
        out = d > Rout
        alive[idx[dead | out]] = False
        esc[idx[out]] = True
    return esc.mean(), np.sqrt(esc.mean() * (1 - esc.mean()) / n)

if __name__ == "__main__":
    a = 1.0
    for lu in (10.0, 100.0, 1000.0):
        for R0 in (2.0, 4.0, 8.0):
            P, e = escape(R0, lu, a, seed=int(lu + R0))
            ll = np.log(R0 / a) / np.log(lu / a)
            kk = 1 - k0(R0 / (2 * lu)) / k0(a / (2 * lu))
            print(json.dumps({"l_u": lu, "R0": R0, "P_meas": round(P, 4), "err": round(e, 4),
                              "P_leadlog": round(ll, 4), "P_K0": round(kk, 4)}), flush=True)
