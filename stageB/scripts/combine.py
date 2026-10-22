#!/usr/bin/env python3
"""Group runs by physical parameters, combine slopes (inverse-variance, jackknife errors), and fit
s_nuc = s_adv + k (beta - alpha)/alpha.  Writes results/combined.json."""
import csv, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
rows = list(csv.DictReader(open("results/summary_table.csv")))


def wmean(vals):
    v = np.array([x[0] for x in vals]); e = np.array([x[1] for x in vals])
    w = 1 / e ** 2
    m = (w * v).sum() / w.sum(); err = 1 / np.sqrt(w.sum())
    chi2 = float(((v - m) ** 2 * w).sum()); return float(m), float(err), chi2, len(v)


GROUPS = {
    "beta=0, alpha=0.1 (P1,P1b,P2,P2b,P3,P1_dt0.01,P1_u2x)": ["P1", "P1b", "P2", "P2b", "P3", "P1_dt0.01", "P1_u2x"],
    "beta=alpha=0.1 (P5,P5b,P5_u2x)": ["P5", "P5b", "P5_u2x"],
    "beta=2alpha=0.2 (P4,P4b,P4_u2x)": ["P4", "P4b", "P4_u2x"],
    "alpha=0.3, beta=0 (P8)": ["P8"],
    "alpha=0.3, beta=0.6 (P9)": ["P9"],
    "alpha=0.3, beta=0.3 (diag_P5_a0.3)": ["diag_P5_a0.3"],
    "beta=alpha=0.1, L=128, u0=0.08 (diag_P5_L128)": ["diag_P5_L128"],
    "beta=0, L=128, u0=0.08 (dep_L128_P1)": ["dep_L128_P1"],
    "T=0.55 beta=0 (dep_T0.55_P1)": ["dep_T0.55_P1"],
    "T=0.55 beta=0.2 (dep_T0.55_P4)": ["dep_T0.55_P4"],
    "T=0.45 beta=0 (dep_T0.45_P1)": ["dep_T0.45_P1"],
    "uniform u, omega=0 (P6) [reference profile]": ["P6"],
    "no drive (P7,P7b) [reference profile]": ["P7", "P7b"],
}
out = {}
for tau in ("5", "10", "20"):
    res = {}
    for g, names in GROUPS.items():
        sel = [r for r in rows if r["run"] in names and r["tau_c"] == tau and r.get("nuc_s_glm")]
        if not sel:
            continue
        d = {}
        for k in ("nuc", "col"):
            d[f"s_{k}"] = wmean([(float(r[f"{k}_s_glm"]), float(r[f"{k}_ds_jk"])) for r in sel])
        d["s_nuc_corr"] = wmean([(float(r["nuc_s_corr"]), float(r["nuc_ds_corr"])) for r in sel])
        d["N_nuc"] = int(sum(int(r["N_nuc"]) for r in sel)); d["runs"] = [r["run"] for r in sel]
        d["T"] = float(sel[0]["T"]); d["alpha"] = float(sel[0]["alpha"]); d["beta"] = float(sel[0]["beta"])
        res[g] = d
    # linear fit in x = (beta - alpha)/alpha for T = 0.5 shear groups
    pts = [(d["alpha"], d["beta"], *d["s_nuc"][:2]) for g, d in res.items()
           if d["T"] == 0.5 and "reference" not in g and "L=128" not in g]
    if len(pts) >= 3:
        x = np.array([(b - a) / a for a, b, _, _ in pts]); y = np.array([p[2] for p in pts]); e = np.array([p[3] for p in pts])
        X = np.vstack([np.ones_like(x), x]).T; W = 1 / e ** 2
        cov = np.linalg.inv(X.T @ (W[:, None] * X)); b = cov @ (X.T @ (W * y))
        chi2 = float((W * (y - X @ b) ** 2).sum())
        res["fit s_nuc = s_adv + k x"] = dict(s_adv=float(b[0]), ds_adv=float(np.sqrt(cov[0, 0])), k=float(b[1]),
                                              dk=float(np.sqrt(cov[1, 1])), chi2=chi2, dof=len(x) - 2,
                                              x0=float(-b[0] / b[1]) if b[1] else None)
    out[f"tau{tau}"] = res
json.dump(out, open("results/combined.json", "w"), indent=1)
for g, d in out["tau10"].items():
    if "s_nuc" in d:
        print(f"{g:52s} N={d['N_nuc']:6d} s_nuc={d['s_nuc'][0]:+6.1f}±{d['s_nuc'][1]:4.1f} (chi2 {d['s_nuc'][2]:.1f}/{d['s_nuc'][3]-1}) "
              f"s_col={d['s_col'][0]:+6.1f}±{d['s_col'][1]:4.1f} s_corr={d['s_nuc_corr'][0]:+6.1f}±{d['s_nuc_corr'][1]:4.1f}")
    else:
        print(g, d)
