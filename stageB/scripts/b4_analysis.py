#!/usr/bin/env python3
"""Stage B''' (request section 10): separate the stationary measure (population) from saddle kinetics.

Population slope  s_pop = s_nuc - s_col  (local steady state n = Gamma_nuc / nu),
saddle slope      s_sad = s_nuc + s_col.
Predictions from the undriven rates under the weak-damping Zeeman equivalence (B, T) -> (B - w/2, T(1 + w r/2)):
    s_X^pred = -1/2 d ln X0 / dB + (r T / 2) d ln X0 / dT,   r = -1/(4J + B),
with ln X0 = ln Gamma0 - ln nu0 (pop) or ln Gamma0 + ln nu0 (sad).  Re-analysis only; writes results/b4_summary.json.
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from skynuc.analysis import object_counts, glm_fit

J = 1.0
TAUS = (5.0, 10.0, 20.0)


def blocks(d, tau):
    """Per (replica, time block, band): N_nuc, area-time, N_col, established skyrmion-time; plus metadata."""
    s, cnt, E, _ = object_counts(d, tau)
    c = s["config"]
    tblk = c["t_run"] / c["n_tblocks"]
    band_area = c["L"] * (c["L"] // c["nbands"])
    return dict(Nn=cnt["nuc"], An=E["nuc"], Nc=cnt["col"], Ec=E["col"], om=np.array(s["omega_band"]),
                nominal=band_area * tblk, L=c["L"], T=c["T"], B=c["B"], D=c["D"])


# ---------------- undriven rates and their derivatives ----------------
def lnq(b, sel=None):
    """ln Gamma0, ln nu0, ln n0 = ln(Gamma0/nu0), ln n_direct (established skyrmions per site) from block sums."""
    m = np.ones(b["Nn"].shape[:2], bool) if sel is None else sel
    Nn = b["Nn"][m].sum(); An = b["An"][m].sum(); Nc = b["Nc"][m].sum(); Ec = b["Ec"][m].sum()
    nominal_tot = b["nominal"] * b["Nn"].shape[2] * m.sum()
    return np.array([np.log(Nn / An), np.log(Nc / Ec), np.log(Nn / An) - np.log(Nc / Ec), np.log(Ec / nominal_tot)])


NAMES = ["lnGamma", "lnnu", "lnn_ratio", "lnn_direct"]


def deriv(d1, d2, dx, tau, paired=True):
    """(q(d2) - q(d1))/dx for the four quantities, with a delete-one-block jackknife.
    paired=True: runs share seeds, delete the same (replica, block) from both (common random numbers).
    paired=False: independent runs, jackknife each and add variances."""
    b1, b2 = blocks(d1, tau), blocks(d2, tau)
    full = (lnq(b2) - lnq(b1)) / dx
    R, K = b1["Nn"].shape[:2]
    if paired:
        vals = []
        for r in range(R):
            for k in range(K):
                m = np.ones((R, K), bool); m[r, k] = False
                vals.append((lnq(b2, m) - lnq(b1, m)) / dx)
        vals = np.array(vals); n = len(vals)
        err = np.sqrt((n - 1) / n * ((vals - vals.mean(0)) ** 2).sum(0))
    else:
        err2 = np.zeros(4)
        for b in (b1, b2):
            R_, K_ = b["Nn"].shape[:2]
            vals = []
            for r in range(R_):
                for k in range(K_):
                    m = np.ones((R_, K_), bool); m[r, k] = False
                    vals.append(lnq(b, m))
            vals = np.array(vals); n = len(vals)
            err2 += (n - 1) / n * ((vals - vals.mean(0)) ** 2).sum(0)
        err = np.sqrt(err2) / abs(dx)
    return {k: (float(full[i]), float(err[i])) for i, k in enumerate(NAMES)}


def deriv_T_avg(pairs_lo, pairs_hi, dT, tau):
    """T derivative from the mean of ln q over the (B-, B+) runs at two temperatures (P17); independent runs."""
    def mean_q(ds):
        qs, es = [], []
        for d in ds:
            b = blocks(d, tau); qs.append(lnq(b))
            R_, K_ = b["Nn"].shape[:2]; vals = []
            for r in range(R_):
                for k in range(K_):
                    m = np.ones((R_, K_), bool); m[r, k] = False
                    vals.append(lnq(b, m))
            vals = np.array(vals); n = len(vals)
            es.append((n - 1) / n * ((vals - vals.mean(0)) ** 2).sum(0))
        return np.mean(qs, 0), np.sqrt(np.sum(es, 0)) / len(ds)
    lo, elo = mean_q(pairs_lo); hi, ehi = mean_q(pairs_hi)
    full = (hi - lo) / dT; err = np.hypot(elo, ehi) / dT
    return {k: (float(full[i]), float(err[i])) for i, k in enumerate(NAMES)}


def predict(dB, dT, B, T):
    r = -1.0 / (4 * J + B)
    out = {}
    for name, comb in (("nuc", {"lnGamma": 1}), ("col", {"lnnu": 1}), ("pop", {"lnGamma": 1, "lnnu": -1}),
                       ("pop_direct", {"lnn_direct": 1}), ("sad", {"lnGamma": 1, "lnnu": 1})):
        gB = sum(w * dB[k][0] for k, w in comb.items()); gT = sum(w * dT[k][0] for k, w in comb.items())
        # errors: lnGamma and lnnu derivatives treated as independent (conservative for pop / sad)
        eB = np.sqrt(sum((w * dB[k][1]) ** 2 for k, w in comb.items()))
        eT = np.sqrt(sum((w * dT[k][1]) ** 2 for k, w in comb.items()))
        t1, t2 = -0.5 * gB, r * T / 2 * gT
        out[name] = dict(term_B=t1, term_T=t2, pred=t1 + t2, err=float(np.hypot(0.5 * eB, abs(r * T / 2) * eT)))
    out["r"] = r
    return out


# ---------------- driven slopes with a joint jackknife ----------------
def slopes_from(b, m=None):
    R, K = b["Nn"].shape[:2]
    m = np.ones((R, K), bool) if m is None else m
    Nn, An = b["Nn"][m].sum(0), b["An"][m].sum(0)
    Nc, Ec = b["Nc"][m].sum(0), b["Ec"][m].sum(0)
    sn = glm_fit(Nn, An, b["om"])["s"]; sc = glm_fit(Nc, Ec, b["om"])["s"]
    dens = Ec / (b["nominal"] * m.sum())
    ok = dens > 0
    sd = np.polyfit(b["om"][ok], np.log(dens[ok]), 1)[0]
    return np.array([sn, sc, sn - sc, sn + sc, sd])


SL = ["s_nuc", "s_col", "s_pop", "s_sad", "s_dens"]


def driven(d, tau):
    b = blocks(d, tau)
    full = slopes_from(b)
    R, K = b["Nn"].shape[:2]; vals = []
    for r in range(R):
        for k in range(K):
            m = np.ones((R, K), bool); m[r, k] = False
            vals.append(slopes_from(b, m))
    vals = np.array(vals); n = len(vals)
    err = np.sqrt((n - 1) / n * ((vals - vals.mean(0)) ** 2).sum(0))
    nu0 = float(b["Nc"].sum() / b["Ec"].sum())
    res = {k: (float(full[i]), float(err[i])) for i, k in enumerate(SL)}
    res["s_nuc_survcorr"] = (float(full[0] + nu0 * tau * full[1]), float(np.hypot(err[0], nu0 * tau * err[1])))
    res["nu0"] = nu0
    return res


def wmean(items, key):
    v = np.array([x[key][0] for x in items]); e = np.array([x[key][1] for x in items]); w = 1 / e ** 2
    return (float((w * v).sum() / w.sum()), float(1 / np.sqrt(w.sum())))


def combine(runs, sign=None):
    sign = sign or [1] * len(runs)
    flipped = [{k: (sg * v[0], v[1]) if isinstance(v, tuple) else v for k, v in r.items()} for r, sg in zip(runs, sign)]
    return {k: wmean(flipped, k) for k in SL + ["s_nuc_survcorr"]}


def ratio(x, p):
    return (x[0] / p["pred"], float(abs(x[0] / p["pred"]) * np.hypot(x[1] / x[0] if x[0] else 0, p["err"] / p["pred"])))



def main():
    out = {}
    for tau in TAUS:
        T_ = f"tau{tau:g}"
        o = out[T_] = {}
        # undriven baseline, D = 0.3
        dB = deriv("results/b3/P16_B0.11", "results/b3/P16_B0.13", 0.02, tau, paired=True)
        dT = deriv("results/ladder/ladder_T0.45", "results/ladder/ladder_T0.55", 0.10, tau, paired=False)
        o["D0.3_undriven"] = dict(dB=dB, dT=dT, pred=predict(dB, dT, 0.12, 0.5))
        # D = 0.2, 0.4 (P17)
        for D, dBB in ((0.2, 0.005), (0.4, 0.01)):
            n = lambda Tt, s: f"results/b3/P17_D{D:.1f}_T{Tt:.2f}_B{s}"
            dBd = deriv(n(0.5, "m"), n(0.5, "p"), 2 * dBB, tau, paired=True)
            dTd = deriv_T_avg([n(0.5, "m"), n(0.5, "p")], [n(0.55, "m"), n(0.55, "p")], 0.05, tau)
            o[f"D{D}_undriven"] = dict(dB=dBd, dT=dTd, pred=predict(dBd, dTd, D ** 2 * 4 / 3, 0.5))
        # driven
        P10 = [driven("results/b2/P10", tau), driven("results/b3/P18_P10", tau)]
        P10b = [driven("results/b2/P10b", tau), driven("results/b3/P18_P10b", tau)]
        p10, p10b = combine(P10), combine(P10b)
        anti = {k: ((p10[k][0] - p10b[k][0]) / 2, float(np.hypot(p10[k][1], p10b[k][1]) / 2)) for k in p10}
        sym = {k: ((p10[k][0] + p10b[k][0]) / 2, float(np.hypot(p10[k][1], p10b[k][1]) / 2)) for k in p10}
        P5 = [driven(f"results/main/{r}", tau) for r in ("P5", "P5b", "P5_u2x")]
        rot_a = {0.025: driven("results/b3/P19_rot_a0.025", tau), 0.05: driven("results/b3/P19_rot_a0.05", tau), 0.1: anti}
        dr = o["driven"] = dict(rotation_P10=p10, rotation_P10b=p10b, rotation_antisym=anti, rotation_sym=sym,
                                shear_beta_alpha_P5group=combine(P5), rot_alpha=rot_a)
        allrot = {k: wmean([rot_a[0.025], rot_a[0.05], anti], k) for k in SL}
        best = {k: wmean([anti, dr["shear_beta_alpha_P5group"]], k) for k in SL}
        dr["rotation_all_alpha"] = allrot; dr["combined_rot_antisym_and_P5"] = best
        # D = 0.2, 0.4 driven (P11, beta = alpha shear)
        dr["P11_D0.2"] = driven("results/b2/P11_D0.2", tau); dr["P11_D0.4"] = driven("results/b2/P11_D0.4", tau)
        # ratios
        F = o["F"] = {}
        for lab, src, pk in (("D0.3_combined", best, "D0.3_undriven"), ("D0.3_rot_antisym", anti, "D0.3_undriven"),
                             ("D0.3_P5group", dr["shear_beta_alpha_P5group"], "D0.3_undriven"),
                             ("D0.3_rot_all_alpha", allrot, "D0.3_undriven"),
                             ("D0.2_P11", dr["P11_D0.2"], "D0.2_undriven"), ("D0.4_P11", dr["P11_D0.4"], "D0.4_undriven")):
            pr = o[pk]["pred"]
            F[lab] = dict(F_nuc=ratio(src["s_nuc"], pr["nuc"]), F_pop=ratio(src["s_pop"], pr["pop"]),
                          F_pop_direct_density=ratio(src["s_dens"], pr["pop_direct"]), F_sad=ratio(src["s_sad"], pr["sad"]),
                          F_col=ratio(src["s_col"], pr["col"]))

    json.dump(out, open("results/b4_summary.json", "w"), indent=1, default=float)
    for tau in TAUS:
        o = out[f"tau{tau:g}"]
        print(f"===== tau_c = {tau:g}")
        for k in ("D0.3_undriven", "D0.2_undriven", "D0.4_undriven"):
            u = o[k]
            print(k, "dB:", {q: tuple(round(x, 2) for x in v) for q, v in u["dB"].items()})
            print("   dT:", {q: tuple(round(x, 2) for x in v) for q, v in u["dT"].items()})
            print("   pred:", {q: round(v["pred"], 2) if isinstance(v, dict) else v for q, v in u["pred"].items()},
                  "errs", {q: round(v["err"], 2) for q, v in u["pred"].items() if isinstance(v, dict)})
        for k in ("rotation_antisym", "rotation_sym", "shear_beta_alpha_P5group", "rotation_all_alpha", "combined_rot_antisym_and_P5", "P11_D0.2", "P11_D0.4"):
            v = o["driven"][k]
            print(k, {q: (round(v[q][0], 1), round(v[q][1], 1)) for q in SL})
        for k, v in o["F"].items():
            print("F", k, {q: (round(x[0], 2), round(x[1], 2)) for q, x in v.items()})


if __name__ == "__main__":
    main()
