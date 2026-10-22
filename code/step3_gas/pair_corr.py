"""Step 3d: directional asymmetry of the +- pair correlation g(r, theta).

Same many-body gas as pair_gas.py. theta is measured from the relative drift
direction of r = r_+ - r_- (physically: -sgn(G_+) zhat x u_eff).

Two-body theory (dilute gas; absorbing core imposed exactly in every harmonic m):
  rho = r/2l, z0 = R0/2l, za = a/2l,  N = K0(za) - I0(z0) K0(z0)
  g - 1 = e^{rho cos th} [ S(rho, th) - K0(rho) ] / N
  S = sum_m eps_m (-1)^m I_m(z0) cos(m th) * { I_m(z0) K_m(rho)  (rho > z0)
                                              { K_m(z0) I_m(rho)  (rho < z0)
  n^2 = Gamma N / (2 pi D_r)
The geminate cloud (excess downstream) and the non-geminate wake (depletion
downstream) cancel at leading order; the remaining anisotropy is O((R0/l)^2).
"""
import numpy as np, json, sys
from scipy.spatial import cKDTree
from scipy.special import iv, kv, k0, i0

def _coef(ell, R0, a, Gamma=1.0, D=1.0, mmax=30):
    """Exact two-body solution: c = e^{rho cos th} phi,
    phi = n^2 e^{-rho cos th} + (Gamma/2pi Dr) S + sum_m B_m K_m(rho) cos m th,
    phi(rho_a, th) = 0 for every m; n^2 fixed by total absorbed flux = Gamma."""
    Dr = 2 * D
    z0, za = R0 / (2 * ell), a / (2 * ell)
    m = np.arange(mmax + 1); eps = np.where(m == 0, 1.0, 2.0); sgn = (-1.0) ** m
    gam = Gamma / (2 * np.pi * Dr) * iv(m, z0) * kv(m, z0)
    Rm = iv(m, za) / kv(m, za)
    n2 = (Gamma / (2 * np.pi * Dr) - np.sum(eps * sgn * Rm * gam)) / np.sum(eps * sgn * Rm)
    return dict(z0=z0, za=za, m=m, eps=eps, sgn=sgn, gam=gam, n2=n2, Gs=Gamma / (2 * np.pi * Dr))

def g_theory(r, th, ell, R0, a, mmax=30):
    r = np.asarray(r, float); th = np.asarray(th, float)
    if not np.isfinite(ell):
        g = np.where(r < R0, np.log(np.maximum(r, a) / a) / np.log(R0 / a), 1.0)
        return g + 0 * th
    c = _coef(ell, R0, a, mmax=mmax)
    rho = r / (2 * ell); z0, za, n2, Gs = c["z0"], c["za"], c["n2"], c["Gs"]
    phi = n2 * np.exp(-rho * np.cos(th))
    for m in range(mmax + 1):
        eps = 1.0 if m == 0 else 2.0; sg = (-1.0) ** m
        src = np.where(rho > z0, iv(m, z0) * kv(m, rho), kv(m, z0) * iv(m, rho)) * iv(m, z0)
        # B_m K_m(rho) = -eps sg I_m(za) (n2 + gam_m) K_m(rho)/K_m(za)
        hom = -eps * sg * iv(m, za) * (n2 + c["gam"][m]) * (kv(m, rho) / kv(m, za))
        phi = phi + (Gs * eps * sg * src + hom) * np.cos(m * th)
    return np.exp(rho * np.cos(th)) * phi / n2

def n_theory(Gamma, ell, R0, a, D=1.0):
    Dr = 2 * D
    if not np.isfinite(ell):
        return np.sqrt(Gamma * np.log(R0 / a) / (2 * np.pi * Dr))
    return np.sqrt(_coef(ell, R0, a, Gamma, D)["n2"])

def run(ell=5.0, Gamma=1.1e-7, L=1500.0, D=1.0, a=1.0, R0=5.0, dt=0.1,
        tmax=40000.0, tburn=8000.0, Rmax=None, nr=30, nth=24, nblk=10, seed=0):
    rng = np.random.default_rng(seed)
    if Rmax is None:
        Rmax = 8 * ell if np.isfinite(ell) else 4 * R0
    Dr = 2 * D
    v = 0.0 if not np.isfinite(ell) else Dr / ell
    vp = np.array([v / 2, 0.0])
    sig = np.sqrt(2 * D * dt); sb = np.sqrt(Dr * dt)
    N0 = int(n_theory(Gamma, ell, R0, a, D) * L * L)
    P = rng.uniform(0, L, (N0, 2)); M = rng.uniform(0, L, (N0, 2))
    redges = np.linspace(a, Rmax, nr + 1); tedges = np.linspace(-np.pi, np.pi, nth + 1)
    H = np.zeros((nblk, nr, nth)); W = np.zeros(nblk)
    nsteps = int(tmax / dt); nburn = int(tburn / dt)
    blk_len = (nsteps - nburn) / nblk
    acc_n = 0.0; nacc = 0
    for s in range(nsteps):
        k = rng.poisson(Gamma * L * L * dt)
        if k:
            c = rng.uniform(0, L, (k, 2)); th = rng.uniform(0, 2 * np.pi, k)
            e = np.stack([np.cos(th), np.sin(th)], 1) * (R0 / 2)
            P = np.vstack([P, (c + e) % L]); M = np.vstack([M, (c - e) % L])
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
                order = np.argsort(dd1); usedP, usedM = set(), set()
                for o in order:
                    i, j = ii[o], jj[o]
                    if i in usedP or j in usedM: continue
                    usedP.add(i); usedM.add(j)
                if usedP:
                    kp = np.ones(len(P), bool); kp[list(usedP)] = False
                    km = np.ones(len(M), bool); km[list(usedM)] = False
                    P, M = P[kp], M[km]
        if s >= nburn and s % 10 == 0 and len(P) and len(M):
            b = min(int((s - nburn) / blk_len), nblk - 1)
            tP = cKDTree(P, boxsize=L); tM = cKDTree(M, boxsize=L)
            sp = tP.sparse_distance_matrix(tM, Rmax, output_type="ndarray")
            if len(sp):
                dv = P[sp["i"]] - M[sp["j"]]; dv -= L * np.round(dv / L)
                rr = np.sqrt((dv ** 2).sum(1)); tt = np.arctan2(dv[:, 1], dv[:, 0])
                H[b] += np.histogram2d(rr, tt, bins=[redges, tedges])[0]
            W[b] += len(P) * len(M) / L ** 2
            acc_n += len(P) / L ** 2; nacc += 1
    area = 0.5 * (redges[1:] ** 2 - redges[:-1] ** 2)[:, None] * np.diff(tedges)[None, :]
    gb = H / (W[:, None, None] * area[None])                      # per-block g(r, th)
    tc = 0.5 * (tedges[1:] + tedges[:-1]); rc = 0.5 * (redges[1:] + redges[:-1])
    g0b = gb.mean(2); g1b = 2 * (gb * np.cos(tc)).mean(2); g2b = 2 * (gb * np.cos(2 * tc)).mean(2)
    s1b = 2 * (gb * np.sin(tc)).mean(2)                            # must vanish (mirror)
    def ms(x): return x.mean(0), x.std(0, ddof=1) / np.sqrt(nblk)
    # theory averaged over each (r, th) bin with sub-sampling
    rs = np.linspace(redges[0], redges[-1], nr * 8 + 1); rs = 0.5 * (rs[1:] + rs[:-1])
    ts = np.linspace(-np.pi, np.pi, nth * 8 + 1); ts = 0.5 * (ts[1:] + ts[:-1])
    RS, TS = np.meshgrid(rs, ts, indexing="ij")
    gt = g_theory(RS, TS, ell, R0, a)
    wt = RS                                                        # area weight
    gt_b = (gt * wt).reshape(nr, 8, nth, 8).sum((1, 3)) / wt.reshape(nr, 8, nth, 8).sum((1, 3))
    g0t = gt_b.mean(1); g1t = 2 * (gt_b * np.cos(tc)).mean(1); g2t = 2 * (gt_b * np.cos(2 * tc)).mean(1)
    # pair-weighted mean cos(theta) within r < R_c (R_c = 4 ell or Rmax)
    Rc = min(4 * ell, Rmax) if np.isfinite(ell) else Rmax
    sel = rc < Rc
    Hs = H.sum(0)
    cos_meas = float((Hs[sel] * np.cos(tc)).sum() / Hs[sel].sum())
    cos_blk = [(H[b][sel] * np.cos(tc)).sum() / H[b][sel].sum() for b in range(nblk)]
    exp_cnt = gt_b * area                                           # theory pair weight
    cos_th = float((exp_cnt[sel] * np.cos(tc)).sum() / exp_cnt[sel].sum())
    out = {"ell": ell, "R0": R0, "Gamma": Gamma, "Rmax": Rmax, "Rc": Rc,
           "n": acc_n / nacc, "n_pred": float(n_theory(Gamma, ell, R0, a, D)),
           "cos_meas": round(cos_meas, 5), "cos_err": round(float(np.std(cos_blk, ddof=1) / np.sqrt(nblk)), 5),
           "cos_theory": round(cos_th, 5),
           "rc": rc.tolist(),
           "g0": ms(g0b)[0].tolist(), "g0_err": ms(g0b)[1].tolist(), "g0_th": g0t.tolist(),
           "g1": ms(g1b)[0].tolist(), "g1_err": ms(g1b)[1].tolist(), "g1_th": g1t.tolist(),
           "g2": ms(g2b)[0].tolist(), "g2_err": ms(g2b)[1].tolist(), "g2_th": g2t.tolist(),
           "s1": ms(s1b)[0].tolist(), "s1_err": ms(s1b)[1].tolist()}
    return out

if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    if "ell" in kw: kw["ell"] = float(kw["ell"])
    tag = sys.argv[2] if len(sys.argv) > 2 else "out"
    res = run(**kw)
    json.dump(res, open(f"corr_{tag}.json", "w"))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}))
