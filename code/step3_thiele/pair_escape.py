"""Step 3b: escape probability of a skyrmion-antiskyrmion pair under current.

Both particles are integrated with the FULL Thiele equation (G and -G, same D):
    G_i zhat x (v_i - u) + D (alpha v_i - beta u) = xi_i
No interaction beyond an absorbing core |r1 - r2| < a (Heisenberg: no log binding).
Pair born at separation R0, random angle. Escape = relative coordinate carried
15 ell_u downstream along the relative drift.

Prediction (relative coordinate = drift-diffusion, ell_u = T / (G |u_eff|)):
    P_esc = ln(R0/a) / [ln(4 ell_u / a) - gamma_E]                    (a, R0 << ell_u)
    full:  1 - I0(R0/2l) K0(R0/2l) / K0(a/2l)                          (monopole, angle avg)
"""
import numpy as np, json, sys
from scipy.special import k0, i0

EPS = np.array([[0.0, -1.0], [1.0, 0.0]])

def run(ell=30.0, R0=3.0, a=1.0, G=1.0, D=1.0, alpha=0.3, beta=0.6, T=1.0,
        Npairs=20000, cfac=0.01, seed=0):
    rng = np.random.default_rng(seed)
    ueff = T / (G * ell)                                  # |u_eff| giving this ell
    u = np.array([alpha / (beta - alpha) * ueff, 0.0])    # current along x
    mu = alpha * D / (alpha**2 * D**2 + G**2)
    Dr = 2 * mu * T
    drift = -2 * mu * G * (EPS @ (np.array([ueff, 0.0])))  # predicted relative drift
    dhat = drift / np.linalg.norm(drift)
    Minv = {s: np.linalg.inv(alpha * D * np.eye(2) + s * G * EPS) for s in (+1, -1)}
    th = rng.uniform(0, 2 * np.pi, Npairs)
    r1 = np.zeros((Npairs, 2))
    r2 = -R0 * np.stack([np.cos(th), np.sin(th)], axis=1)
    alive = np.ones(Npairs, bool); escaped = np.zeros(Npairs, bool)
    nsteps = 0
    while alive.any():
        idx = np.where(alive)[0]
        rel = r1[idx] - r2[idx]
        dist = np.linalg.norm(rel, axis=1)
        dt = np.clip(cfac * (dist - a) ** 2 / Dr, 1e-7 * a**2 / Dr, cfac * ell**2 / Dr)
        sq = np.sqrt(2 * alpha * D * T / dt)[:, None]
        for s, r in ((+1, r1), (-1, r2)):
            xi = rng.normal(size=(len(idx), 2)) * sq
            rhs = xi + s * G * (EPS @ u) + beta * D * u
            r[idx] += (rhs @ Minv[s].T) * dt[:, None]
        rel = r1[idx] - r2[idx]
        dist = np.linalg.norm(rel, axis=1)
        dead = dist < a
        esc = (rel @ dhat > 15 * ell) | (dist > 60 * ell)
        alive[idx[dead | esc]] = False
        escaped[idx[esc & ~dead]] = True
        nsteps += 1
    P = escaped.mean()
    err = np.sqrt(P * (1 - P) / Npairs)
    z0, za = R0 / (2 * ell), a / (2 * ell)
    pred = np.log(R0 / a) / (np.log(4 * ell / a) - np.euler_gamma)
    pred_full = 1 - i0(z0) * k0(z0) / k0(za)
    naive = np.log(R0 / a) / np.log(ell / a)
    return {"ell": ell, "R0": R0, "alpha": alpha, "beta": beta, "P_esc": round(P, 4),
            "err": round(err, 4), "pred": round(pred, 4), "pred_full": round(float(pred_full), 4),
            "naive_doc": round(naive, 4), "steps": nsteps}

if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(run(**kw)))
