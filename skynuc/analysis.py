"""Rate estimates and fits of ln Gamma(y) = ln Gamma0 + s * omega(y) from a run directory."""
import csv, json, os, sys
import numpy as np


def wls_fit(N, expo, om):
    """Poisson-weighted least squares of ln(N/expo) vs om, weights N (var ln N ~ 1/N).

    Bands with N = 0 are dropped. Returns slope, its error, intercept, chi2/dof."""
    N = np.asarray(N, float); expo = np.asarray(expo, float); om = np.asarray(om, float)
    m = (N > 0) & (expo > 0)
    if m.sum() < 3:
        return dict(s=np.nan, ds=np.nan, a=np.nan, chi2dof=np.nan, nb=int(m.sum()))
    y = np.log(N[m] / expo[m]); w = N[m]; X = np.vstack([np.ones(m.sum()), om[m]]).T
    A = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(A)
    beta = cov @ (X.T @ (w * y))
    res = y - X @ beta
    chi2 = float((w * res ** 2).sum())
    dof = max(m.sum() - 2, 1)
    return dict(s=float(beta[1]), ds=float(np.sqrt(cov[1, 1])), a=float(beta[0]), chi2dof=chi2 / dof, nb=int(m.sum()))


def glm_fit(N, expo, om, iters=50):
    """Poisson maximum likelihood: N_b ~ Poisson(expo_b exp(a + s om_b))."""
    N = np.asarray(N, float); expo = np.asarray(expo, float); om = np.asarray(om, float)
    m = expo > 0
    N, expo, om = N[m], expo[m], om[m]
    if N.sum() < 3:
        return dict(s=np.nan, ds=np.nan, a=np.nan)
    X = np.vstack([np.ones(len(N)), om]).T
    b = np.array([np.log(N.sum() / expo.sum()), 0.0])
    for _ in range(iters):
        mu = expo * np.exp(X @ b)
        g = X.T @ (N - mu)
        H = X.T @ (mu[:, None] * X)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-12:
            break
    mu = expo * np.exp(X @ b)
    cov = np.linalg.inv(X.T @ (mu[:, None] * X))
    dev = 2 * np.sum(np.where(N > 0, N * np.log(np.where(N > 0, N, 1) / mu), 0) - (N - mu))
    return dict(s=float(b[1]), ds=float(np.sqrt(cov[1, 1])), a=float(b[0]), deviance_dof=float(dev / max(len(N) - 2, 1)))


def halves(N, expo, om):
    N = np.asarray(N, float); expo = np.asarray(expo, float); om = np.asarray(om, float)
    p, n = om > 0, om < 0
    Np, Nn = N[p].sum(), N[n].sum()
    if Np == 0 or Nn == 0 or not p.any():
        return dict(lnR=np.nan, dlnR=np.nan, Np=float(Np), Nn=float(Nn))
    lnR = np.log(Np / expo[p].sum()) - np.log(Nn / expo[n].sum())
    return dict(lnR=float(lnR), dlnR=float(np.sqrt(1 / Np + 1 / Nn)), Np=float(Np), Nn=float(Nn))


def halves_y(N, expo, nb):
    """Same split as the shear halves (bands nb/4..3nb/4 vs the rest); used for undriven runs."""
    N = np.asarray(N, float); expo = np.asarray(expo, float)
    k = np.arange(nb)
    p = (k >= nb // 4) & (k < 3 * nb // 4)
    om = np.where(p, 1.0, -1.0)
    return halves(N, expo, om)


def _open(path):
    import gzip
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    return open(path)


def read_events(path):
    ev = []
    with _open(path) as f:
        for row in csv.DictReader(f):
            ev.append(row)
    return ev


def committed_counts(ev, nb, bw, t_end, tau_c):
    """Band counts of favoured nucleations whose object lived > tau_c (or survived to the end
    with t_end - t > tau_c), and of collapses of objects older than tau_c (unknown age counts)."""
    life = {}
    for e in ev:
        if e["kind"] == "col":
            life[e["obj_id"]] = float(e["lifetime"])
    nuc = np.zeros(nb); col = np.zeros(nb)
    for e in ev:
        if e["kind"] == "nuc":
            t = float(e["t"])
            L_ = life.get(e["obj_id"], None)
            ok = (L_ > tau_c) if L_ is not None else (t_end - t > tau_c)
            if ok:
                nuc[int(float(e["y"])) // bw % nb] += 1
        elif e["kind"] == "col":
            L_ = float(e["lifetime"])
            if not (L_ <= tau_c):  # NaN (unknown age) counts as committed
                col[int(float(e["y"])) // bw % nb] += 1
    return nuc, col


def lifetimes(ev):
    return np.array([float(e["lifetime"]) for e in ev if e["kind"] == "col" and e["lifetime"] != "nan"])


def analyse(run_dir, tau_c=(0.0, 2.0, 10.0), write=True):
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    c = s["config"]
    nb = c["nbands"]; bw = c["L"] // nb
    om = np.array(s["omega_band"])
    cnt = {k: np.array(v) for k, v in s["counts"].items()}
    fl = {k: np.array(v) for k, v in s["flagged"].items()}
    A = np.array(s["area_time"]); SK = np.array(s["sk_time"])
    out = dict(run_id=c["run_id"], T=c["T"], alpha=c["alpha"], beta=c["beta"], drive=c["drive"], u0=c["u0"],
               L=c["L"], R=c["R"], dt=c["dt"], dts=c["dts"], t_meas=s["t_meas"],
               ueff_factor=(c["beta"] - c["alpha"]) / c["alpha"],
               n_events={k: int(v.sum()) for k, v in cnt.items()},
               n_flagged={k: int(v.sum()) for k, v in fl.items()},
               jump_total=s["jump_total"], dQ_mismatch=s["dQ_mismatch"],
               mean_n_sk_per_replica=float(SK.sum() / s["t_meas"] / c["R"]) if s["t_meas"] else np.nan)
    tot = sum(v.sum() for v in cnt.values())
    out["flag_fraction"] = float(sum(v.sum() for v in fl.values()) / tot) if tot else np.nan
    out["Gamma0_nuc_per_site_time"] = float(cnt["nuc"].sum() / A.sum()) if A.sum() else np.nan
    out["nu_col_per_sk_time"] = float(cnt["col"].sum() / SK.sum()) if SK.sum() else np.nan
    if c["drive"] == "shear":
        out["fit_nuc"] = wls_fit(cnt["nuc"], A, om)
        out["fit_nuc_glm"] = glm_fit(cnt["nuc"], A, om)
        out["fit_col"] = wls_fit(cnt["col"], SK, om)
        out["fit_col_glm"] = glm_fit(cnt["col"], SK, om)
        out["R_nuc"] = halves(cnt["nuc"], A, om)
        out["R_col"] = halves(cnt["col"], SK, om)
        out["fit_anti_nuc"] = wls_fit(cnt["anti_nuc"], A, om)
    else:
        out["R_nuc"] = halves_y(cnt["nuc"], A, nb)
        out["R_col"] = halves_y(cnt["col"], SK, nb)
        # pseudo-fits against the omega profile a shear of the same u0 (or 1) would have
        L = c["L"]; edges = np.arange(nb + 1) * bw
        om_ref = -(1.0 / bw) * (np.sin(2 * np.pi * edges[1:] / L) - np.sin(2 * np.pi * edges[:-1] / L))
        out["fit_nuc_vs_cos_profile"] = wls_fit(cnt["nuc"], A, om_ref)
    # committed (recrossing-filtered) variants
    evp = os.path.join(run_dir, "events.csv")
    if os.path.exists(evp) or os.path.exists(evp + ".gz"):
        ev = read_events(evp)
        t_end = c["t_eq"] + c["t_run"]
        lt = lifetimes(ev)
        out["lifetime_quantiles"] = (np.quantile(lt, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if len(lt) else [])
        out["committed"] = {}
        for tc in tau_c:
            nn, cc = committed_counts(ev, nb, bw, t_end, tc)
            d = dict(N_nuc=int(nn.sum()), N_col=int(cc.sum()))
            if c["drive"] == "shear":
                d["fit_nuc"] = wls_fit(nn, A, om); d["fit_col"] = wls_fit(cc, SK, om)
                d["R_nuc"] = halves(nn, A, om); d["R_col"] = halves(cc, SK, om)
            else:
                d["R_nuc"] = halves_y(nn, A, nb); d["R_col"] = halves_y(cc, SK, nb)
            out["committed"][f"tau_{tc:g}"] = d
    # drift and density profiles
    ddt = np.array(s["drift_dt"])
    with np.errstate(invalid="ignore", divide="ignore"):
        out["drift_vx_band"] = (np.array(s["drift_dx"]) / ddt).tolist()
        out["drift_vy_band"] = (np.array(s["drift_dy"]) / ddt).tolist()
    out["sk_density_band"] = (SK / s["t_meas"] / c["R"] / (bw * c["L"])).tolist() if s["t_meas"] else []
    out["omega_band"] = om.tolist()
    out["counts_band"] = {k: v.tolist() for k, v in cnt.items()}
    out["area_time_band"] = A.tolist(); out["sk_time_band"] = SK.tolist()
    if write:
        json.dump(out, open(os.path.join(run_dir, "analysis.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        o = analyse(d)
        print(json.dumps({k: o[k] for k in o if not k.endswith("_band") and k != "counts_band"}, indent=1, default=float))


# ---------------- space-time chains (recrossing removal) ----------------
# chains_all(): events of one replica that follow each other within tau_c and r_c (periodic) are merged.
# A chain with fav_net = +1 is one net nucleation of the favoured skyrmion, -1 one net collapse,
# 0 a transient (failed nucleation, or collapse followed by renucleation). (t, x, y) = first event.
FAV_SIGN = {"nuc": +1, "col": -1}           # favoured-skyrmion number change
ANTI_SIGN = {"anti_nuc": +1, "anti_col": -1}


def chains_all(ev, tau_c=2.0, r_c=4.0, L=64):
    by_r = {}
    for e in ev:
        by_r.setdefault(int(e["replica"]), []).append(e)
    out = []
    for r, lst in by_r.items():
        lst.sort(key=lambda e: float(e["t"]))
        live = []
        for e in lst:
            t, x, y = float(e["t"]), float(e["x"]), float(e["y"])
            live = [c for c in live if t - c["t_last"] <= tau_c]
            best = None
            for ch in live:
                dx = (x - ch["x_last"] + L / 2) % L - L / 2
                dy = (y - ch["y_last"] + L / 2) % L - L / 2
                if dx * dx + dy * dy <= r_c * r_c:
                    best = ch
                    break
            if best is None:
                best = dict(replica=r, t=t, x=x, y=y, fav_net=0, anti_net=0, n=0, flagged=0)
                live.append(best)
                out.append(best)
            best["t_last"], best["x_last"], best["y_last"] = t, x, y
            best["fav_net"] += FAV_SIGN.get(e["kind"], 0)
            best["anti_net"] += ANTI_SIGN.get(e["kind"], 0)
            best["n"] += 1
            best["flagged"] |= int(e["flag"]) | int(e["kind"] in ("pair", "multi"))
    return out


def chain_counts(run_dir, tau_c=2.0, r_c=4.0):
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    c = s["config"]
    nb = c["nbands"]; L = c["L"]; bw = L // nb; R = c["R"]; nk = c.get("n_tblocks", 1)
    ev = read_events(os.path.join(run_dir, "events.csv"))
    ch = chains_all(ev, tau_c=tau_c, r_c=r_c, L=L)
    shp = (R, nk, nb)
    cnt = {k: np.zeros(shp) for k in ("nuc", "col", "anti_nuc", "anti_col", "transient")}
    nflag = {k: 0 for k in cnt}
    for h in ch:
        k = min(int((h["t"] - c["t_eq"]) / c["t_run"] * nk), nk - 1)
        idx = (h["replica"], k, int(np.floor(h["y"])) // bw % nb)
        kinds = []
        if h["fav_net"] == 1: kinds.append("nuc")
        elif h["fav_net"] == -1: kinds.append("col")
        if h["anti_net"] == 1: kinds.append("anti_nuc")
        elif h["anti_net"] == -1: kinds.append("anti_col")
        if h["fav_net"] == 0 and h["anti_net"] == 0: kinds.append("transient")
        for kd in kinds:
            cnt[kd][idx] += 1
            nflag[kd] += h["flagged"]
    area = np.array(s["area_rkb"]) if "area_rkb" in s else None
    sk = np.array(s["sk_rkb"]) if "sk_rkb" in s else None
    return s, cnt, nflag, area, sk


def jackknife(fun, N, E, om):
    """Delete-one-block jackknife over the (replica, time-block) axes of N[r,k,b], E[r,k,b]."""
    R, K, _ = N.shape
    full = fun(N.sum((0, 1)), E.sum((0, 1)), om)
    vals = []
    for r in range(R):
        for k in range(K):
            Nm = N.sum((0, 1)) - N[r, k]; Em = E.sum((0, 1)) - E[r, k]
            vals.append(fun(Nm, Em, om))
    vals = np.array(vals); n = len(vals)
    err = np.sqrt((n - 1) / n * ((vals - vals.mean()) ** 2).sum()) if n > 1 else np.nan
    return full, err


def _glm_s(N, E, om):
    return glm_fit(N, E, om)["s"]


def _lnR(N, E, om):
    return halves(N, E, om)["lnR"]


def analyse_chains(run_dir, tau_c=2.0, r_c=4.0):
    s, cnt, nflag, area, sk = chain_counts(run_dir, tau_c, r_c)
    c = s["config"]; nb = c["nbands"]
    om = np.array(s["omega_band"])
    if c["drive"] != "shear":
        L = c["L"]; bw = L // nb; edges = np.arange(nb + 1) * bw
        # reference profile -cos(2 pi y / L) (band average), to test for spurious y dependence
        om = -(1.0 / bw) * (np.sin(2 * np.pi * edges[1:] / L) - np.sin(2 * np.pi * edges[:-1] / L)) * 2 * np.pi / L
        om_kind = "reference -2pi/L cos(2pi y/L) (u0=1 shear profile)"
    else:
        om_kind = "shear"
    out = dict(run_id=c["run_id"], tau_c=tau_c, r_c=r_c, omega_kind=om_kind, T=c["T"], alpha=c["alpha"],
               beta=c["beta"], drive=c["drive"], u0=c["u0"], L=c["L"], R=c["R"], t_run=c["t_run"], dts=c["dts"],
               N={k: int(v.sum()) for k, v in cnt.items()}, N_flagged=nflag,
               ueff_factor=(c["beta"] - c["alpha"]) / c["alpha"])
    for kind, E in (("nuc", area), ("col", sk), ("anti_nuc", area)):
        N = cnt[kind]
        if E is None or N.sum() < 3:
            continue
        Nb, Eb = N.sum((0, 1)), E.sum((0, 1))
        w = wls_fit(Nb, Eb, om); g = glm_fit(Nb, Eb, om); h = halves(Nb, Eb, om)
        _, s_jk = jackknife(_glm_s, N, E, om)
        _, r_jk = jackknife(_lnR, N, E, om)
        out[kind] = dict(s_wls=w["s"], ds_wls_poisson=w["ds"], chi2dof=w["chi2dof"], s_glm=g["s"],
                         ds_glm_poisson=g["ds"], ds_glm_jackknife=s_jk, lnR=h["lnR"], dlnR_poisson=h["dlnR"],
                         dlnR_jackknife=r_jk, Np=h["Np"], Nn=h["Nn"],
                         rate=float(Nb.sum() / Eb.sum()),
                         lnrate_band=np.log(np.where(Nb > 0, Nb, np.nan) / Eb).tolist(),
                         N_band=Nb.tolist(), E_band=Eb.tolist())
    out["omega_band"] = om.tolist()
    return out


# ---------------- object-based rates (primary definition) ----------------
def read_objects(path):
    with _open(path) as f:
        return list(csv.DictReader(f))


def object_counts(run_dir, tau_c=5.0):
    """Band counts of nucleations (births of objects that live >= tau_c) and collapses (deaths of objects
    that lived >= tau_c), per (replica, time block, band). Returns summary, counts, exposures."""
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    c = s["config"]
    nb = c["nbands"]; L = c["L"]; bw = L // nb; R = c["R"]; nk = c["n_tblocks"]
    taus = list(c["tau_est"])
    if tau_c not in taus:
        raise ValueError(f"tau_c must be one of {taus}")
    j = taus.index(tau_c)
    t0, t1 = c["t_eq"], c["t_eq"] + c["t_run"]
    cnt = {k: np.zeros((R, nk, nb)) for k in ("nuc", "col", "anti_nuc")}
    lifes = []
    for o in read_objects(os.path.join(run_dir, "objects.csv")):
        sign = int(o["sign"]); life = float(o["lifetime"]); cens = int(o["censored"])
        tb = float(o["t_birth"]); td = float(o["t_death"])
        fav = sign == c["Q_fav"]
        if not np.isnan(tb) and life >= tau_c:
            kind = "nuc" if fav else "anti_nuc"
            k = min(int((tb - t0) / c["t_run"] * nk), nk - 1)
            cnt[kind][int(o["replica"]), k, int(np.floor(float(o["y_birth"]))) // bw % nb] += 1
        if fav and not np.isnan(tb):
            lifes.append((life, cens))
        if fav and not cens and (np.isnan(tb) or life >= tau_c):
            k = min(int((td - t0) / c["t_run"] * nk), nk - 1)
            cnt["col"][int(o["replica"]), k, int(np.floor(float(o["y_death"]))) // bw % nb] += 1
    E = dict(nuc=np.array(s["area_rkb"]), anti_nuc=np.array(s["area_rkb"]), col=np.array(s["est_rkb"])[j])
    return s, cnt, E, lifes


def object_counts_multi(run_dirs, tau_c):
    """Pool independent runs (same parameters, different seeds) by stacking their replica axes."""
    parts = [object_counts(d, tau_c) for d in run_dirs]
    s = parts[0][0]
    cnt = {k: np.concatenate([p[1][k] for p in parts], axis=0) for k in parts[0][1]}
    E = {k: np.concatenate([p[2][k] for p in parts], axis=0) for k in parts[0][2]}
    lifes = [x for p in parts for x in p[3]]
    if len(parts) > 1:
        s = dict(s)
        for key in ("drift_dx", "drift_dy", "drift_dt", "sk_time"):
            s[key] = np.sum([np.array(p[0][key]) for p in parts], axis=0).tolist()
        s["t_meas"] = parts[0][0]["t_meas"]
        s["config"] = dict(s["config"], R=sum(p[0]["config"]["R"] for p in parts),
                           run_id="+".join(p[0]["config"]["run_id"] for p in parts))
        for key in ("counts", "flagged"):
            s[key] = {k: np.sum([np.array(p[0][key][k]) for p in parts], axis=0).tolist() for k in s[key]}
        s["dQ_mismatch"] = sum(p[0]["dQ_mismatch"] for p in parts)
        s["jump_total"] = sum(p[0]["jump_total"] for p in parts)
    return s, cnt, E, lifes


def analyse_objects(run_dir, tau_c=5.0):
    dirs = run_dir if isinstance(run_dir, (list, tuple)) else [run_dir]
    s, cnt, E, lifes = object_counts_multi(dirs, tau_c)
    c = s["config"]; nb = c["nbands"]; L = c["L"]; bw = L // nb
    if c["drive"] == "shear":
        om = np.array(s["omega_band"]); om_kind = "shear"
    else:
        edges = np.arange(nb + 1) * bw
        u_ref = 0.04  # same profile as the main shear points P1 (u0 = 0.04), so slopes are comparable
        om = -(u_ref / bw) * (np.sin(2 * np.pi * edges[1:] / L) - np.sin(2 * np.pi * edges[:-1] / L))
        om_kind = "reference: omega(y) of the u0=0.04 shear (no shear applied)"
    out = dict(run_id=c["run_id"], tau_c=tau_c, omega_kind=om_kind, T=c["T"], alpha=c["alpha"], beta=c["beta"],
               drive=c["drive"], u0=c["u0"], L=L, R=c["R"], t_run=c["t_run"], dts=c["dts"], dt=c["dt"],
               D=c["D"], B=c["B"], ueff_factor=(c["beta"] - c["alpha"]) / c["alpha"],
               N={k: int(v.sum()) for k, v in cnt.items()})
    lf = np.array([x[0] for x in lifes]) if lifes else np.zeros(0)
    out["fav_objects"] = len(lf)
    out["frac_objects_surviving_tau"] = float((lf >= tau_c).mean()) if len(lf) else np.nan
    for kind in ("nuc", "col", "anti_nuc"):
        N, Ex = cnt[kind], E[kind]
        Nb, Eb = N.sum((0, 1)), Ex.sum((0, 1))
        if N.sum() < 3 or Eb.sum() <= 0:
            out[kind] = dict(rate=float(N.sum() / Eb.sum()) if Eb.sum() > 0 else np.nan, N_band=Nb.tolist())
            continue
        w = wls_fit(Nb, Eb, om); g = glm_fit(Nb, Eb, om); h = halves(Nb, Eb, om)
        _, s_jk = jackknife(_glm_s, N, Ex, om)
        _, r_jk = jackknife(_lnR, N, Ex, om)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(Nb / Eb)
        out[kind] = dict(rate=float(Nb.sum() / Eb.sum()), s_wls=w["s"], ds_wls=w["ds"], chi2dof=w["chi2dof"],
                         s_glm=g["s"], ds_glm=g["ds"], ds_jk=s_jk, lnR=h["lnR"], dlnR=h["dlnR"], dlnR_jk=r_jk,
                         Np=h["Np"], Nn=h["Nn"], lnrate_band=[None if not np.isfinite(x) else float(x) for x in lr],
                         N_band=Nb.tolist(), E_band=Eb.tolist())
    # transport: drift velocity and density profile of tracked favoured skyrmions
    ddt = np.array(s["drift_dt"])
    with np.errstate(invalid="ignore", divide="ignore"):
        out["drift_vx_band"] = (np.array(s["drift_dx"]) / ddt).tolist()
        out["drift_vy_band"] = (np.array(s["drift_dy"]) / ddt).tolist()
    out["sk_density_band"] = (np.array(s["sk_time"]) / s["t_meas"] / c["R"] / (bw * L)).tolist()
    out["omega_band"] = om.tolist()
    out["mean_sk_per_replica"] = float(np.sum(s["sk_time"]) / s["t_meas"] / c["R"])
    out["runs"] = dirs
    out["raw_jump_events"] = {k: int(np.sum(v)) for k, v in s["counts"].items()}
    out["raw_flagged"] = {k: int(np.sum(v)) for k, v in s["flagged"].items()}
    tot = sum(out["raw_jump_events"].values())
    out["raw_flag_fraction"] = sum(out["raw_flagged"].values()) / tot if tot else np.nan
    out["dQ_mismatch_per_jump"] = s["dQ_mismatch"] / max(s["jump_total"], 1)
    return out


def fmt(o):
    n, cl = o["nuc"], o["col"]
    line = (f"{o['run_id']:26s} T={o['T']:.2f} u0={o['u0']:+.3f} a={o['alpha']} b={o['beta']} tau={o['tau_c']:g} "
            f"Nnuc={o['N']['nuc']:5d} Ncol={o['N']['col']:5d} Nanti={o['N']['anti_nuc']:4d} "
            f"G0={n['rate']:.3e} nu={cl['rate']:.4f} nsk/rep={o['mean_sk_per_replica']:.2f} "
            f"flag={o['raw_flag_fraction']:.3f} mism={o['dQ_mismatch_per_jump']:.3f}")
    if "s_glm" in n:
        line += (f"\n    s_nuc={n['s_wls']:+.2f}±{n['ds_wls']:.2f}(P)±{n['ds_jk']:.2f}(JK)  lnR_nuc={n['lnR']:+.3f}"
                 f"±{n['dlnR']:.3f}(P)±{n['dlnR_jk']:.3f}(JK)")
    if "s_glm" in cl:
        line += (f"\n    s_col={cl['s_wls']:+.2f}±{cl['ds_wls']:.2f}(P)±{cl['ds_jk']:.2f}(JK)  lnR_col={cl['lnR']:+.3f}"
                 f"±{cl['dlnR']:.3f}(P)±{cl['dlnR_jk']:.3f}(JK)")
    return line
