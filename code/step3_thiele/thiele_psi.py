"""Step 3a: stationary distribution of a skyrmion in a nonuniform current.

Full Thiele equation with STT, written in the ORIGINAL frame (not the w-frame):
    G zhat x (v - u) + D (alpha v - beta u) = F + xi,   <xi xi> = 2 alpha D T
=>  v = (alpha D 1 + G eps)^{-1} [F + xi + G eps u + beta D u]

The current field u(r) is implemented directly. The prediction uses only the
stream function of u_eff = (beta - alpha) u / alpha:
    zhat x u_eff = grad psi_e,   P(r) ~ exp(-G psi_e(r) / T)        (doc 4.2)
"""
import numpy as np, json, sys

EPS = np.array([[0.0, -1.0], [1.0, 0.0]])          # eps a = zhat x a

def step_velocity(r, u, F, G, D, alpha, beta, T, dt, rng):
    M = alpha * D * np.eye(2) + G * EPS
    Minv = np.linalg.inv(M)
    xi = rng.normal(size=r.shape) * np.sqrt(2 * alpha * D * T / dt)
    rhs = F + xi + G * (u @ EPS.T) + beta * D * u
    return rhs @ Minv.T

# ---------------- case A: cellular flow, periodic box ----------------
def cellular(G=1.0, D=1.0, alpha=0.3, beta=0.6, T=1.0, A=1.5, L=10.0,
             N=4000, dt=0.004, tmax=1500.0, tburn=150.0, nb=20, seed=1):
    rng = np.random.default_rng(seed)
    k = 2 * np.pi / L
    def psi(r):
        return A * np.sin(k * r[:, 0]) * np.sin(k * r[:, 1])
    def ueff(r):                                   # u_eff = (d_y psi, -d_x psi)
        x, y = r[:, 0], r[:, 1]
        return np.stack([A * k * np.sin(k * x) * np.cos(k * y),
                         -A * k * np.cos(k * x) * np.sin(k * y)], axis=1)
    r = rng.uniform(0, L, size=(N, 2))
    H = np.zeros((nb, nb)); nsteps = int(tmax / dt); nburn = int(tburn / dt)
    for n in range(nsteps):
        u = alpha / (beta - alpha) * ueff(r)
        v = step_velocity(r, u, 0.0 * r, G, D, alpha, beta, T, dt, rng)
        r = (r + v * dt) % L
        if n >= nburn and n % 25 == 0:
            H += np.histogram2d(r[:, 0], r[:, 1], bins=nb, range=[[0, L], [0, L]])[0]
    c = (np.arange(nb) + 0.5) * L / nb
    X, Y = np.meshgrid(c, c, indexing="ij")
    ps = A * np.sin(k * X) * np.sin(k * Y)
    P = H / H.sum()
    pred = np.exp(-G * ps / T); pred /= pred.sum()
    # regression ln P = c0 - s * psi  -> s should equal G/T
    m = P > 0
    s = -np.polyfit(ps[m], np.log(P[m]), 1)[0]
    rel = np.abs(P - pred).max() / pred.max()
    return {"case": "cellular", "G": G, "alpha": alpha, "beta": beta,
            "fit_G_over_T": round(s, 4), "pred_G_over_T": G / T,
            "max_dev_rel": round(rel, 4),
            "ratio_vortex": round(P[np.unravel_index(ps.argmin(), ps.shape)] /
                                  P[np.unravel_index(ps.argmax(), ps.shape)], 2),
            "pred_ratio_vortex": round(float(pred.max() / pred.min()), 2)}

# ---------------- case B: strip, edge accumulation ----------------
def strip(G=1.0, D=1.0, alpha=0.3, beta=0.6, T=1.0, u0=0.5, W=10.0, L=10.0,
          mod=0.0, N=4000, dt=0.004, tmax=800.0, tburn=100.0, nb=40, seed=2):
    """u_eff = u0 [1 + mod sin(2 pi y / W)] xhat, reflecting walls at y = 0, W."""
    rng = np.random.default_rng(seed)
    def ueff_x(y):
        return u0 * (1 + mod * np.sin(2 * np.pi * y / W))
    r = np.stack([rng.uniform(0, L, N), rng.uniform(0, W, N)], axis=1)
    H = np.zeros(nb); nsteps = int(tmax / dt); nburn = int(tburn / dt)
    for n in range(nsteps):
        ux = alpha / (beta - alpha) * ueff_x(r[:, 1])
        u = np.stack([ux, 0 * ux], axis=1)
        v = step_velocity(r, u, 0.0 * r, G, D, alpha, beta, T, dt, rng)
        r = r + v * dt
        r[:, 0] %= L
        y = r[:, 1]
        y = np.where(y < 0, -y, y); y = np.where(y > W, 2 * W - y, y)
        r[:, 1] = y
        if n >= nburn and n % 25 == 0:
            H += np.histogram(r[:, 1], bins=nb, range=(0, W))[0]
    c = (np.arange(nb) + 0.5) * W / nb
    # psi(y) = int_0^y u_eff,x
    psi = u0 * (c - mod * W / (2 * np.pi) * (np.cos(2 * np.pi * c / W) - 1))
    P = H / H.sum()
    pred = np.exp(-G * psi / T); pred /= pred.sum()
    rel = np.abs(P - pred).max() / pred.max()
    out = {"case": "strip", "mod": mod, "G": G, "alpha": alpha, "beta": beta,
           "max_dev_rel": round(rel, 4),
           "edge_ratio_meas": round(P[0] / P[-1], 3) if G > 0 else round(P[-1] / P[0], 3),
           "edge_ratio_pred": round(pred[0] / pred[-1], 3) if G > 0 else round(pred[-1] / pred[0], 3)}
    if mod == 0:
        s = -np.polyfit(c, np.log(P), 1)[0]
        out["fit_inv_ell"] = round(s, 4); out["pred_inv_ell"] = G * u0 / T   # 1/ell_u
    return out

if __name__ == "__main__":
    which = sys.argv[1]
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    f = {"cellular": cellular, "strip": strip}[which]
    print(json.dumps(f(**kw)))
