"""Step 3a: single skyrmion in a vortical flow.

Thiele with STT:  G eps (v-u) + alpha D (v - (beta/alpha) u) = F + noise
=> with w = v - (beta/alpha) u:
   (alpha D + G eps) w = -G eps u_eff + xi,   v = w + (beta/alpha) u,
   u_eff = ((beta-alpha)/alpha) u,  eps = rotation by +90 deg.
Cellular flow u = (d_y phi, -d_x phi), phi = A sin x sin y  =>  eps u = grad phi,
so psi_e = c phi with c = (beta-alpha)/alpha.  Claim: P(r) ∝ exp(-G c phi / T).
Control: extra uniform advection (U0, 0) not parallel to u_eff -> claim should fail.
"""
import numpy as np, sys, json

def run(G=2.0, D=1.0, alpha=0.5, ratio=3.0, A=1.0, T=2.0, N=20000,
        dt=0.005, tmax=600.0, tburn=200.0, extra=0.0, advect=True, seed=1):
    rng = np.random.default_rng(seed)
    c = ratio - 1.0                      # (beta - alpha)/alpha
    L = 2 * np.pi
    den = (alpha * D) ** 2 + G ** 2
    Msym = alpha * D / den               # mobility, symmetric part
    Mant = G / den                       # (alpha D + G eps)^-1 = (alpha D - G eps)/den
    r = rng.uniform(0, L, size=(N, 2))
    nsteps = int(tmax / dt); nburn = int(tburn / dt)
    acc_phi = 0.0; acc_n = 0
    H = np.zeros((40, 40))
    for n in range(nsteps):
        x, y = r[:, 0], r[:, 1]
        # grad phi
        gx = A * np.cos(x) * np.sin(y); gy = A * np.sin(x) * np.cos(y)
        # u = (d_y phi, -d_x phi)
        ux, uy = gy, -gx
        # force F = -G eps u_eff = -G c grad phi
        Fx, Fy = -G * c * gx, -G * c * gy
        # w = M F,  M = Msym*1 - Mant*eps,  eps(Fx,Fy) = (-Fy, Fx)
        wx = Msym * Fx + Mant * Fy
        wy = Msym * Fy - Mant * Fx
        vx = wx + (ratio * ux if advect else 0.0) + extra
        vy = wy + (ratio * uy if advect else 0.0)
        noise = rng.normal(size=(N, 2)) * np.sqrt(2 * T * Msym * dt)
        r[:, 0] = (x + vx * dt + noise[:, 0]) % L
        r[:, 1] = (y + vy * dt + noise[:, 1]) % L
        if n >= nburn and n % 20 == 0:
            phi = A * np.sin(r[:, 0]) * np.sin(r[:, 1])
            acc_phi += phi.mean(); acc_n += 1
            H += np.histogram2d(r[:, 0], r[:, 1], bins=40, range=[[0, L], [0, L]])[0]
    # prediction on a fine grid
    g = np.linspace(0, L, 801)[:-1] + L / 1600
    X, Y = np.meshgrid(g, g)
    PHI = A * np.sin(X) * np.sin(Y)
    w = np.exp(-G * c * PHI / T)
    phi_pred = float((PHI * w).sum() / w.sum())
    # histogram comparison: predicted bin probabilities
    Xb, Yb = np.meshgrid(np.linspace(0, L, 41)[:-1] + L / 80, np.linspace(0, L, 41)[:-1] + L / 80, indexing="ij")
    Pb = np.exp(-G * c * A * np.sin(Xb) * np.sin(Yb) / T); Pb /= Pb.sum()
    Hn = H / H.sum()
    tv = 0.5 * np.abs(Hn - Pb).sum()     # total-variation distance
    return {"G": G, "alpha": alpha, "ratio": ratio, "T": T, "extra": extra,
            "advect": advect, "phi_meas": acc_phi / acc_n, "phi_pred": phi_pred,
            "phi_uniform": 0.0, "TV_dist": float(tv)}

if __name__ == "__main__":
    cases = [dict(), dict(advect=False), dict(alpha=0.2), dict(G=-2.0),
             dict(ratio=0.0), dict(extra=1.0)]
    for i, kw in enumerate(cases):
        print(json.dumps(run(seed=10 + i, **kw)), flush=True)
