"""High-T check of the vorticity coupling (doc section 3.3).

Stochastic LL dynamics on an L x L square lattice (PBC), Heisenberg J,
FL-channel nonreciprocal exchange delta J_ij = dJ * (e_ij . v(midpoint)),
v = (sin(2 pi y / L), 0)  ->  vorticity omega = -d_y v_x.

  dS_i = [-S_i x (h_i + hN_i) - lam S_i x (S_i x h_i)] dt - sqrt(2 lam T) S_i x o dW_i
  h_i  = J sum_j S_j,   hN_i = sum_j dJ_ij S_j   (dJ_ji = -dJ_ij)

First-order high-T prediction (E_1 = -dE_N/dt / (6 lam T)):
  <chi_ijk> = J (dJ_ij - dJ_ik) / (27 lam T^2)    for chi_ijk = S_i.(S_j x S_k)
"""
import numpy as np, time, sys, json

def roll(a, s, ax):
    return np.roll(a, s, axis=ax)

def cross(a, b):
    return np.cross(a, b)

def fields(S, J, dx, dy):
    # S: (R, L, L, 3), axes 1 = y, 2 = x
    Sxp = roll(S, -1, 2); Sxm = roll(S, 1, 2)
    Syp = roll(S, -1, 1); Sym = roll(S, 1, 1)
    h = J * (Sxp + Sxm + Syp + Sym)
    # dx[y,x] = dJ on bond (x,y)->(x+1,y); dJ_{i,i-x} = -dx[i-x]
    dxm = roll(dx, 1, 1)  # dx at i-x  (dx has axes y,x)
    dym = roll(dy, 1, 0)
    hN = (dx[..., None] * Sxp - dxm[..., None] * Sxm
          + dy[..., None] * Syp - dym[..., None] * Sym)
    return h, hN

def drift(S, J, lam, dx, dy):
    h, hN = fields(S, J, dx, dy)
    return -cross(S, h + hN) - lam * cross(S, cross(S, h))

def run(L=32, R=16, J=1.0, T=2.5, lam=0.5, dJ=0.4, dt=None, nsteps=20000,
        neq=2000, every=5, seed=0):
    rng = np.random.default_rng(seed)
    if dt is None:
        dt = min(0.01, 0.005 / (lam * T))
    y = np.arange(L)
    vx = np.sin(2 * np.pi * y / L)                     # v at x-bond midpoints (same y)
    dx = np.repeat((dJ * vx)[:, None], L, axis=1)      # (y, x)
    dy = np.zeros((L, L))
    S = rng.normal(size=(R, L, L, 3)); S /= np.linalg.norm(S, axis=-1, keepdims=True)
    amp = np.sqrt(2 * lam * T * dt)
    acc = {k: np.zeros(L) for k in ("UR", "UL", "LL", "LR", "chain")}
    nmeas = 0
    for n in range(neq + nsteps):
        dW = rng.normal(size=S.shape) * amp
        f0 = drift(S, J, lam, dx, dy); g0 = -cross(S, dW)
        St = S + f0 * dt + g0; St /= np.linalg.norm(St, axis=-1, keepdims=True)
        f1 = drift(St, J, lam, dx, dy); g1 = -cross(St, dW)
        S = S + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1)
        S /= np.linalg.norm(S, axis=-1, keepdims=True)
        if n >= neq and (n - neq) % every == 0:
            Sxp = roll(S, -1, 2); Sxm = roll(S, 1, 2)
            Syp = roll(S, -1, 1); Sym = roll(S, 1, 1)
            def chi(a, b, c):
                return np.einsum("...k,...k->...", a, cross(b, c))
            vals = {"UR": chi(S, Sxp, Syp), "UL": chi(S, Syp, Sxm),
                    "LL": chi(S, Sxm, Sym), "LR": chi(S, Sym, Sxp),
                    "chain": chi(S, Sxp, Sxm)}           # chi(i,+x,-x)
            for k, v in vals.items():
                acc[k] += v.mean(axis=(0, 2))           # average over replicas and x
            nmeas += 1
    prof = {k: v / nmeas for k, v in acc.items()}
    return prof, vx, dt

def project(prof, basis):
    return 2.0 / len(basis) * np.sum(prof * basis)

if __name__ == "__main__":
    T = float(sys.argv[1]); lam = float(sys.argv[2])
    dJ = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    L = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    R = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    J = 1.0
    t0 = time.time()
    out = {}
    for sgn in (+1, -1):
        prof, vx, dt = run(L=L, R=R, T=T, lam=lam, dJ=sgn * dJ, nsteps=nsteps, seed=7 + sgn)
        out[sgn] = prof
    y = np.arange(L)
    s = np.sin(2 * np.pi * y / L)
    pred_unit = J * dJ / (27 * lam * T ** 2)   # <chi> per unit coefficient (dJ_ij - dJ_ik)/dJ
    # coefficient amplitudes in units of dJ * sin: UR +1, UL +1, LL -1, LR -1, chain +2
    sign = {"UR": 1, "UL": 1, "LL": -1, "LR": -1, "chain": 2}
    res = {"T": T, "lam": lam, "dJ": dJ, "dt": dt, "J_over_lamT": J / (lam * T)}
    for k in sign:
        anti = 0.5 * (out[1][k] - out[-1][k])
        res[k] = project(anti, s) / (sign[k] * pred_unit)
        res[k + "_cos"] = project(anti, np.cos(2 * np.pi * y / L)) / (sign[k] * pred_unit)
    # plaquette sum (4 corner triangles of each plaquette) vs discrete curl
    # plaquette between rows y, y+1: UR(y)+UL(y) [at x, x+1] + LL(y+1)+LR(y+1)
    anti = {k: 0.5 * (out[1][k] - out[-1][k]) for k in sign}
    plaq = anti["UR"] + anti["UL"] + np.roll(anti["LL"] + anti["LR"], -1)
    curl_pred = pred_unit * 2 * (s - np.roll(s, -1))      # 2[d(y) - d(y+1)] / dJ
    res["plaquette"] = float(np.sum(plaq * curl_pred) / np.sum(curl_pred ** 2))
    res["time_s"] = round(time.time() - t0, 1)
    print(json.dumps(res))
