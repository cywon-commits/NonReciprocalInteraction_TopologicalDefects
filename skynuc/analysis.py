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


def read_events(path):
    ev = []
    with open(path) as f:
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
    if os.path.exists(evp):
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
