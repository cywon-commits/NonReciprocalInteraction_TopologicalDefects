"""Step 3c: many-body skyrmion/antiskyrmion gas with pair creation and annihilation.

Reduced Thiele dynamics (verified in step 3b): in the centre-of-mass frame,
+ particles drift +v/2 dhat, - particles -v/2 dhat, each diffuses with D.
Relative drift v = D_r / ell_u, D_r = 2D. No interaction except annihilation at a.
Pairs are created uniformly at rate Gamma per area, separation R0, random angle.

Prediction (doc 5.3): n^2 = Gamma ln(R0/a) / (2 pi D_r), independent of ell_u,
valid for a, R0 << ell_u << n^{-1/2}.
Crossings within a step are caught with the Brownian-bridge probability
exp[-(d0-a)(d1-a)/(D_r dt)].
"""
import numpy as np, json, sys
from scipy.spatial import cKDTree

def mind(A, B, L):
    d = A[:, None, :] - B[None, :, :]
    d -= L * np.round(d / L)
    return np.sqrt((d ** 2).sum(-1))

def run(ell=30.0, Gamma=1e-6, L=1000.0, D=1.0, a=1.0, R0=3.0, dt=0.1,
        tmax=40000.0, tburn=10000.0, seed=0):
    rng = np.random.default_rng(seed)
    Dr = 2 * D
    v = 0.0 if not np.isfinite(ell) else Dr / ell
    vp = np.array([v / 2, 0.0])
    sig = np.sqrt(2 * D * dt)
    sb = np.sqrt(Dr * dt)
    n_pred = np.sqrt(Gamma * np.log(R0 / a) / (2 * np.pi * Dr))
    # start near predicted density
    N0 = int(n_pred * L * L)
    P = rng.uniform(0, L, (N0, 2)); M = rng.uniform(0, L, (N0, 2))
    idP = -np.arange(1, N0 + 1); idM = -np.arange(N0 + 1, 2 * N0 + 1)  # unpaired ids
    nextid = 0
    acc_n = 0.0; nacc = 0; created = 0; gem = 0; nongem = 0
    nsteps = int(tmax / dt); nburn = int(tburn / dt); series = []
    for s in range(nsteps):
        k = rng.poisson(Gamma * L * L * dt)
        if k:
            c = rng.uniform(0, L, (k, 2)); th = rng.uniform(0, 2 * np.pi, k)
            e = np.stack([np.cos(th), np.sin(th)], 1) * (R0 / 2)
            P = np.vstack([P, (c + e) % L]); M = np.vstack([M, (c - e) % L])
            ids = np.arange(nextid, nextid + k); nextid += k
            idP = np.concatenate([idP, ids]); idM = np.concatenate([idM, ids])
            if s >= nburn: created += k
        P0, M0 = P.copy(), M.copy()
        P = (P + vp * dt + sig * rng.normal(size=P.shape)) % L
        M = (M - vp * dt + sig * rng.normal(size=M.shape)) % L
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
                order = np.argsort(dd1)
                usedP, usedM = set(), set()
                for o in order:
                    i, j = ii[o], jj[o]
                    if i in usedP or j in usedM: continue
                    usedP.add(i); usedM.add(j)
                    if s >= nburn:
                        if idP[i] == idM[j] and idP[i] >= 0: gem += 1
                        else: nongem += 1
                if usedP:
                    kp = np.ones(len(P), bool); kp[list(usedP)] = False
                    km = np.ones(len(M), bool); km[list(usedM)] = False
                    P, idP = P[kp], idP[kp]; M, idM = M[km], idM[km]
        if s >= nburn and s % 10 == 0:
            acc_n += len(P) / L ** 2; nacc += 1; series.append(len(P))
    n = acc_n / nacc
    ser = np.array(series, float); nb = 20
    blocks = ser[: len(ser) // nb * nb].reshape(nb, -1).mean(1)
    n_err = blocks.std(ddof=1) / np.sqrt(nb) / L ** 2
    return {"ell": ell, "Gamma": Gamma, "L": L, "n": n, "n_pred": n_pred,
            "ratio": round(n / n_pred, 4), "ratio_err": round(n_err / n_pred, 4), "spacing": round(n ** -0.5, 1),
            "N_avg": round(n * L * L, 1),
            "frac_escape": round(1 - gem / max(created, 1), 4),
            "P_esc_pred": round(np.log(R0 / a) / (np.log(4 * ell / a) - np.euler_gamma), 4)
                          if np.isfinite(ell) else None,
            "created": created, "gem": gem, "nongem": nongem}

if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    if "ell" in kw: kw["ell"] = float(kw["ell"])
    print(json.dumps(run(**kw)))
