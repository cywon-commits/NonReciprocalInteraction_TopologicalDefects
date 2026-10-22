"""Stage A: stochastic LL on an L x L square lattice with arbitrary FL delta J on bonds.

  dS = [-S x (h + hN) - lam S x (S x h)] dt - sqrt(2 lam T) S x o dW
  E = -J sum S_i.S_j over present bonds, h = -dE/dS, hN_i = sum_j dJ_ij S_j, dJ_ji = -dJ_ij
dxb[y, x]: dJ on bond (x,y)->(x+1,y);  dyb[y, x]: dJ on bond (x,y)->(x,y+1).
Periodic in x; periodic or open in y.
Observable: Q_chi = (1/8pi) sum over all corner triangles (i, i+a, i+b) present.
"""
import numpy as np, json, sys, time

def rollx(a, s): return np.roll(a, s, axis=2)
def rolly(a, s): return np.roll(a, s, axis=1)

class Model:
    def __init__(s, L, J, lam, T, dxb, dyb, open_y=False):
        s.L, s.J, s.lam, s.T = L, J, lam, T
        s.dxb, s.dyb = dxb, dyb
        # bond masks: x bonds always present; y bond (y)->(y+1) absent at y = L-1 if open
        s.my = np.ones((L, L));
        if open_y: s.my[L - 1, :] = 0.0
        s.open_y = open_y
    def fields(s, S):
        J = s.J; my = s.my[None, :, :, None]; mym = np.roll(s.my, 1, 0)[None, :, :, None]
        Sxp, Sxm = rollx(S, -1), rollx(S, 1); Syp, Sym = rolly(S, -1), rolly(S, 1)
        h = J * (Sxp + Sxm + my * Syp + mym * Sym)
        dx = s.dxb[None, :, :, None]; dxm = np.roll(s.dxb, 1, 1)[None, :, :, None]
        dy = (s.dyb * s.my)[None, :, :, None]; dym = np.roll(s.dyb * s.my, 1, 0)[None, :, :, None]
        hN = dx * Sxp - dxm * Sxm + dy * Syp - dym * Sym
        return h, hN
    def drift(s, S):
        h, hN = s.fields(S)
        return -np.cross(S, h + hN) - s.lam * np.cross(S, np.cross(S, h))
    def corners(s, S):
        """sum over sites of the 4 corner chiralities that exist; returns per-replica total."""
        Sxp, Sxm = rollx(S, -1), rollx(S, 1); Syp, Sym = rolly(S, -1), rolly(S, 1)
        c = lambda a, b, d: np.einsum("...k,...k->...", a, np.cross(b, d))
        up = s.my[None]; dn = np.roll(s.my, 1, 0)[None]     # +y neighbour exists / -y neighbour exists
        tot = (c(S, Sxp, Syp) * up + c(S, Syp, Sxm) * up + c(S, Sxm, Sym) * dn + c(S, Sym, Sxp) * dn)
        return tot                                       # (R, L, L)

def run(L=32, R=16, J=1.0, T=1.5, lam=0.5, mode="uniform", dJ=1.0, theta=0.0, slope=0.0,
        open_y=False, tmax=200.0, tburn=20.0, dt=None, every=10, seed=0):
    rng = np.random.default_rng(seed)
    if dt is None: dt = min(0.01, 0.005 / (lam * T))
    yy = np.arange(L)[:, None] * np.ones((1, L))
    if mode == "uniform":
        dxb = dJ * np.cos(theta) * np.ones((L, L)); dyb = dJ * np.sin(theta) * np.ones((L, L))
    elif mode == "linear":          # v_x = slope * (y - y0): omega = -slope (uniform vorticity)
        y0 = (L - 1) / 2
        dxb = slope * (yy - y0); dyb = np.zeros((L, L))
    m = Model(L, J, lam, T, dxb, dyb, open_y)
    S = rng.normal(size=(R, L, L, 3)); S /= np.linalg.norm(S, axis=-1, keepdims=True)
    amp = np.sqrt(2 * lam * T * dt)
    nsteps = int(tmax / dt); nburn = int(tburn / dt)
    vals = []; prof = np.zeros(L); nprof = 0
    for n in range(nsteps):
        dW = rng.normal(size=S.shape) * amp
        f0 = m.drift(S); g0 = -np.cross(S, dW)
        St = S + f0 * dt + g0; St /= np.linalg.norm(St, axis=-1, keepdims=True)
        f1 = m.drift(St); g1 = -np.cross(St, dW)
        S = S + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1)
        S /= np.linalg.norm(S, axis=-1, keepdims=True)
        if n >= nburn and (n - nburn) % every == 0:
            cs = m.corners(S)
            vals.append(cs.sum(axis=(1, 2)) / (8 * np.pi))     # Q per replica
            prof += cs.mean(axis=(0, 2)); nprof += 1
    V = np.array(vals)                                         # (nsamp, R)
    nb = 20
    blk = V[: len(V) // nb * nb].reshape(nb, -1, V.shape[1]).mean(1).mean(1)
    Qm = V.mean(); Qe = blk.std(ddof=1) / np.sqrt(nb)
    out = {"mode": mode, "L": L, "T": T, "lam": lam, "J": J, "dJ": dJ, "theta": theta,
           "slope": slope, "open_y": open_y, "Q_mean": Qm, "Q_err": Qe,
           "q_per_site": Qm / L**2, "q_err": Qe / L**2, "dt": dt,
           "corner_profile": (prof / nprof).tolist()}
    # predictions
    if mode == "linear":
        # leading + (8/9) J/T correction: total Q = kappa * sum_p omega_p, plaquette sum = 2(a-b) omega
        # per plaquette <sum of 4 corners> = 2 (1 + 8/9 J/T) * omega * J/(27 lam T^2), omega = -slope
        npl = L * (L - 1) if open_y else L * L
        omega = -slope
        pred0 = npl * 2 * omega * J / (27 * lam * T**2) / (8 * np.pi)
        out["Q_pred_lead"] = pred0; out["Q_pred_2nd"] = pred0 * (1 + 8 / 9 * J / T)
    return out

if __name__ == "__main__":
    kw = json.loads(sys.argv[1]); t0 = time.time()
    r = run(**kw); r["time_s"] = round(time.time() - t0, 1)
    r.pop("corner_profile") if "keep_prof" not in sys.argv else None
    print(json.dumps(r), flush=True)
