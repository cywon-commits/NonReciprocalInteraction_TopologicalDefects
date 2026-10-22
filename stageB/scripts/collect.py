#!/usr/bin/env python3
"""Collect per-run analyses into results/summary_table.csv and results/analysis/<run>.json."""
import csv, glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from skynuc.analysis import analyse_objects

os.chdir(ROOT)
os.makedirs("results/analysis", exist_ok=True)
rows = []
dirs = sorted(glob.glob("results/main/*")) + sorted(glob.glob("results/ladder/*")) + sorted(glob.glob("results/dep/*")) + sorted(glob.glob("results/b2/*"))
for d in dirs:
    if not os.path.exists(os.path.join(d, "summary.json")):
        continue
    s = json.load(open(os.path.join(d, "summary.json")))
    if s["phase"] != "done":
        continue
    res = {}
    for tau in (5.0, 10.0, 20.0):
        o = analyse_objects(d, tau)
        res[f"tau{tau:g}"] = o
        n, c = o["nuc"], o["col"]
        row = dict(run=o["run_id"], tau_c=tau, T=o["T"], alpha=o["alpha"], beta=o["beta"], drive=o["drive"],
                   u0=o["u0"], dt=o["dt"], ueff_u0=o["ueff_factor"] * o["u0"], L=o["L"], R=o["R"], t_run=o["t_run"],
                   N_nuc=o["N"]["nuc"], N_col=o["N"]["col"], Gamma0=n["rate"], nu=c["rate"],
                   n_sk_per_replica=o["mean_sk_per_replica"], raw_flag_frac=o["raw_flag_fraction"])
        for k, r in (("nuc", n), ("col", c)):
            for f in ("s_wls", "ds_wls", "s_glm", "ds_jk", "lnR", "dlnR", "dlnR_jk", "chi2dof"):
                row[f"{k}_{f}"] = r.get(f, np.nan)
        if "s_glm" in n and "s_glm" in c:
            # survival-corrected nucleation slope: N_{>=tau}(w) ~ Gamma(w) exp(-nu(w) tau)
            row["nuc_s_corr"] = n["s_glm"] + c["rate"] * tau * c["s_glm"]
            row["nuc_ds_corr"] = float(np.hypot(n["ds_jk"], c["rate"] * tau * c["ds_jk"]))
            row["col_over_nuc"] = c["s_glm"] / n["s_glm"] if n["s_glm"] else np.nan
        rows.append(row)
    json.dump(res, open(f"results/analysis/{os.path.basename(d)}.json", "w"), indent=1, default=float)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open("results/summary_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    if r["tau_c"] == 10.0:
        print(f"{r['run']:12s} N={r['N_nuc']:5d}/{r['N_col']:5d} s_nuc={r.get('nuc_s_glm', np.nan):+7.2f}±{r.get('nuc_ds_jk', np.nan):5.2f} "
              f"s_col={r.get('col_s_glm', np.nan):+7.2f}±{r.get('col_ds_jk', np.nan):5.2f} lnR={r.get('nuc_lnR', np.nan):+.3f}±{r.get('nuc_dlnR_jk', np.nan):.3f} "
              f"s_corr={r.get('nuc_s_corr', np.nan):+7.2f}±{r.get('nuc_ds_corr', np.nan):.2f} G0={r['Gamma0']:.3g} nu={r['nu']:.3g}")
