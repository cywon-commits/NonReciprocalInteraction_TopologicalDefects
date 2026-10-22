"""Lifetime of a seeded isolated skyrmion at temperature T (calibration aid for 1A).

A replica counts as collapsed at the first sample after which the total (one-diagonal) Q stays 0
for tau_c. Output: mean lifetime estimate (censored exponential MLE) and first-passage times.
"""
import json, sys, time
import numpy as np

from .model import Integrator, charge_map
from .single import relax_skyrmion


def run(B=0.12, D=0.3, T=0.4, L=32, R=16, t_max=2000.0, tau_c=2.0, dts=0.24, alpha=0.1, seed=3):
    S0 = relax_skyrmion(L=L, D=D, B=B, r0=max(2.0, 0.85 / D * (0.12 / B) ** 0.5), t_relax=400.0)
    S = np.repeat(S0, R, axis=0)
    it = Integrator(S, D=D, B=B, alpha=alpha, T=T, seed=seed)
    ns = int(round(dts / it.p["dt"]))
    q = np.empty(S.shape[:3])
    zero_since = np.full(R, np.nan)
    t_die = np.full(R, np.nan)
    while it.t < t_max and np.isnan(t_die).any():
        it.run(ns)
        charge_map(it.S, q, False)
        Q = np.rint(q.sum(axis=(1, 2)))
        for r in range(R):
            if not np.isnan(t_die[r]):
                continue
            if Q[r] == 0:
                if np.isnan(zero_since[r]):
                    zero_since[r] = it.t
                elif it.t - zero_since[r] >= tau_c:
                    t_die[r] = zero_since[r]
            else:
                zero_since[r] = np.nan
    n = int((~np.isnan(t_die)).sum())
    exposure = np.nansum(t_die) + (R - n) * it.t
    tau = exposure / n if n else np.inf
    return dict(B=B, D=D, T=T, L=L, R=R, t_max=it.t, n_collapsed=n, mean_lifetime=tau,
                lifetimes=[None if np.isnan(x) else float(x) for x in t_die])


if __name__ == "__main__":
    kw = json.loads(sys.argv[1])
    t0 = time.time()
    r = run(**kw)
    r["wall_s"] = time.time() - t0
    print(json.dumps(r), flush=True)
