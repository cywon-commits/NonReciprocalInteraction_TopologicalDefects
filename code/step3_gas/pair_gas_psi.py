"""Items 1-2: skyrmion/antiskyrmion gas in a cellular stream function.

Reduced Thiele velocities (step 3b), units mu = D = T = 1, G = 1:
    v_pm = -/+ G grad psi + c_adv eps grad psi        (common level-set advection)
psi = A sin(kx) sin(ky), one cell per periodic box, b = G A / T.
Processes: pair creation (rate Gamma per area, separation R0), pair annihilation
at a (Brownian bridge), optional single creation g per species per area and
single decay nu per particle (Q-changing lattice events), unbiased here.

Mean-field predictions
  nu = 0:  n_pm(r) = A0 exp(-/+ G psi / T),  n_+ n_- uniform,
           nbar / n0 = < exp(-G psi/T) >   (exact steady state for any transport speed)
  nu > 0 (linear in b): rho = n_+ - n_- = -(G s / T) * k_c^2 / (k_c^2 + nu/D) * psi,
           k_c^2 = 2 k^2  (crossover length l_nu = sqrt(D/nu))
"""
import numpy as np, json, sys
from scipy.spatial import cKDTree

def run(b=1.5, L=750.0, Gamma=1.76e-6, R0=3.0, a=1.0, dt=0.1, c_adv=-1.083,
        g=0.0, nu=0.0, tmax=1.2e5, tburn=3e4, nb=12, seed=0):
    rng = np.random.default_rng(seed)
    D = 1.0; Dr = 2.0; G = 1.0; T = 1.0
    A = b * T / G; k = 2 * np.pi / L
    sig = np.sqrt(2 * D * dt); sb = np.sqrt(Dr * dt)
    def gradpsi(r):
        x, y = r[:, 0], r[:, 1]
        return np.stack([A * k * np.cos(k * x) * np.sin(k * y),
                         A * k * np.sin(k * x) * np.cos(k * y)], 1)
    def vel(r, s):
        gp = gradpsi(r)
        egp = np.stack([-gp[:, 1], gp[:, 0]], 1)
        return -s * G * gp + c_adv * egp
    n0 = np.sqrt(Gamma * np.log(R0 / a) / (2 * np.pi * Dr))
    N0 = int(n0 * L * L)
    P = rng.uniform(0, L, (N0, 2)); M = rng.uniform(0, L, (N0, 2))
    HP = np.zeros((nb, nb)); HM = np.zeros((nb, nb)); HPM = np.zeros((nb, nb))
    nP = []; nM = []
    nsteps = int(tmax / dt); nburn = int(tburn / dt)
    for s in range(nsteps):
        kp = rng.poisson(Gamma * L * L * dt)
        if kp:
            c = rng.uniform(0, L, (kp, 2)); th = rng.uniform(0, 2 * np.pi, kp)
            e = np.stack([np.cos(th), np.sin(th)], 1) * (R0 / 2)
            P = np.vstack([P, (c + e) % L]); M = np.vstack([M, (c - e) % L])
        if g > 0:
            P = np.vstack([P, rng.uniform(0, L, (rng.poisson(g * L * L * dt), 2))])
            M = np.vstack([M, rng.uniform(0, L, (rng.poisson(g * L * L * dt), 2))])
        if nu > 0:
            P = P[rng.random(len(P)) > nu * dt]; M = M[rng.random(len(M)) > nu * dt]
        P0, M0 = P.copy(), M.copy()
        if len(P): P = (P + vel(P, +1) * dt + sig * rng.normal(size=P.shape)) % L
        if len(M): M = (M + vel(M, -1) * dt + sig * rng.normal(size=M.shape)) % L
        if len(P) and len(M):
            tP = cKDTree(P, boxsize=L); tM = cKDTree(M, boxsize=L)
            sp = tP.sparse_distance_matrix(tM, a + 6 * sb, output_type="ndarray")
            if len(sp):
                ii, jj, dd1 = sp["i"], sp["j"], sp["v"]
                dv = P0[ii] - M0[jj]; dv -= L * np.round(dv / L)
                dd0 = np.sqrt((dv ** 2).sum(1))
                pr = np.where(dd1 < a, 1.0,
                              np.exp(-np.clip(dd0 - a, 0, None) * (dd1 - a) / (Dr * dt)))
                hit = rng.random(len(pr)) < pr
                ii, jj, dd1 = ii[hit], jj[hit], dd1[hit]
                order = np.argsort(dd1); uP, uM = set(), set()
                for o in order:
                    i, j = ii[o], jj[o]
                    if i in uP or j in uM: continue
                    uP.add(i); uM.add(j)
                if uP:
                    kP = np.ones(len(P), bool); kP[list(uP)] = False
                    kM = np.ones(len(M), bool); kM[list(uM)] = False
                    P, M = P[kP], M[kM]
        if s >= nburn and s % 10 == 0:
            hp = np.histogram2d(P[:, 0], P[:, 1], bins=nb, range=[[0, L], [0, L]])[0] if len(P) else 0
            hm = np.histogram2d(M[:, 0], M[:, 1], bins=nb, range=[[0, L], [0, L]])[0] if len(M) else 0
            HP += hp; HM += hm; nP.append(len(P)); nM.append(len(M))
    ns = len(nP); cell = (L / nb) ** 2
    nPm = HP / ns / cell; nMm = HM / ns / cell
    # bin-averaged psi~ = sin sin
    sub = 16
    xs = (np.arange(nb * sub) + 0.5) * L / (nb * sub)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    pt = (np.sin(k * X) * np.sin(k * Y))
    pt_b = pt.reshape(nb, sub, nb, sub).mean((1, 3))
    ebp = np.exp(-b * pt).reshape(nb, sub, nb, sub).mean((1, 3))
    ebm = np.exp(+b * pt).reshape(nb, sub, nb, sub).mean((1, 3))
    nbar = 0.5 * (np.mean(nP) + np.mean(nM)) / L ** 2
    rho = nPm - nMm; s_ = nPm + nMm
    # amplitude of rho along psi~ (least squares)
    a_rho = float((rho * pt_b).sum() / (pt_b ** 2).sum())
    # Boltzmann shape test for nu = 0: fit n_+ = A0 * ebp
    A0p = float((nPm * ebp).sum() / (ebp ** 2).sum()); A0m = float((nMm * ebm).sum() / (ebm ** 2).sum())
    resid = float(np.sqrt((((nPm - A0p * ebp) ** 2).mean() + ((nMm - A0m * ebm) ** 2).mean()) / 2) / nbar)
    prod = nPm * nMm
    blocks = np.array(nP[: len(nP) // 10 * 10], float).reshape(10, -1).mean(1) / L ** 2
    out = {"b": b, "nu": nu, "g": g, "nbar": nbar,
           "nbar_err": float(blocks.std(ddof=1) / np.sqrt(10)),
           "n0_formula": n0, "mean_exp": float(np.exp(-b * pt).mean()),
           "Nplus": float(np.mean(nP)), "Nminus": float(np.mean(nM)),
           "a_rho_over_nbar": a_rho / nbar,
           "Boltz_resid_rel": resid,
           "cv_prod": float(prod.std() / prod.mean()), "cv_nplus": float(nPm.std() / nPm.mean()),
           "A0p_over_n0": A0p / n0, "A0m_over_n0": A0m / n0,
           "kc2": 2 * k * k}
    return out

if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(run(**kw)))
