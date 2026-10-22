"""Checks V1 (fluctuation-dissipation) and V2 (equilibrium vs Metropolis MC)."""
import json, sys, time
import numpy as np

from .model import Integrator, uniform_state, energy_per_replica, dmi_vectors, make_rng
from .mc import metropolis_sweeps


def block_err(x, nb=20):
    x = np.asarray(x)
    n = len(x) // nb * nb
    b = x[:n].reshape(nb, -1).mean(axis=1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nb)


def v1(alpha=0.1, dt=0.02, L=32, R=16, t_eq=200.0, t_run=4000.0, seed=1):
    B, T = 0.5, 0.5
    it = Integrator(uniform_state(R, L, L), J=0.0, D=0.0, B=B, K=0.0, alpha=alpha, beta=0.0, T=T, dt=dt, seed=seed)
    it.run(int(t_eq / dt))
    ns = int(1.0 / dt)
    mz = []
    for _ in range(int(t_run)):
        it.run(ns)
        mz.append(it.S[..., 2].mean())
    m, e = block_err(mz)
    exact = 1 / np.tanh(B / T) - T / B
    return dict(check="V1", alpha=alpha, dt=dt, L=L, R=R, t_run=t_run, mz=m, mz_err=e, exact=exact,
                rel_dev=(m - exact) / exact, pass_1pct=bool(abs(m - exact) / exact < 0.01))


def v2(dt=0.02, alpha=0.1, L=64, R=16, t_eq=500.0, t_run=3000.0, mc_eq=4000, mc_run=40000, seed=2):
    J, D, B, T = 1.0, 0.3, 0.12, 0.4
    N = L * L
    # LLG
    it = Integrator(uniform_state(R, L, L), J=J, D=D, B=B, alpha=alpha, T=T, dt=dt, seed=seed)
    it.run(int(t_eq / dt))
    ns = int(1.0 / dt)
    El, Ml = [], []
    for _ in range(int(t_run)):
        it.run(ns)
        El.append(it.energy().mean() / N)
        Ml.append(it.S[..., 2].mean())
    # MC
    Dx, Dy = dmi_vectors(D)
    S = uniform_state(R, L, L)
    rng = make_rng(seed + 100, R * L)
    acc = np.zeros(R * L, np.int64)
    metropolis_sweeps(S, mc_eq, T, J, Dx, Dy, B, 0.0, 0.6, rng, acc)
    acc[:] = 0
    Em, Mm = [], []
    for _ in range(mc_run // 10):
        metropolis_sweeps(S, 10, T, J, Dx, Dy, B, 0.0, 0.6, rng, acc)
        Em.append(energy_per_replica(S, J, Dx, Dy, B, 0.0).mean() / N)
        Mm.append(S[..., 2].mean())
    acc_rate = acc.sum() / (mc_run * R * N / 1.0)
    el, eel = block_err(El); ml, eml = block_err(Ml)
    em, eem = block_err(Em); mm, emm = block_err(Mm)
    return dict(check="V2", dt=dt, alpha=alpha, L=L, R=R, t_run=t_run, mc_sweeps=mc_run, mc_acc=acc_rate,
                E_llg=el, E_llg_err=eel, E_mc=em, E_mc_err=eem, rel_dE=(el - em) / abs(em),
                mz_llg=ml, mz_llg_err=eml, mz_mc=mm, mz_mc_err=emm, rel_dmz=(ml - mm) / mm,
                pass_0p5pct=bool(abs(el - em) / abs(em) < 0.005 and abs(ml - mm) / mm < 0.005))


if __name__ == "__main__":
    which = sys.argv[1]
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    t0 = time.time()
    res = {"V1": v1, "V2": v2}[which](**kw)
    res["wall_s"] = time.time() - t0
    print(json.dumps(res))
