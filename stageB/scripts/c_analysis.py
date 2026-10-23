#!/usr/bin/env python3
"""Stage C (request section 11): F_pop(T) from rotation-only drive (+-omega, antisymmetric part) and undriven
B = 0.11 / 0.13 pairs at T = 0.45, 0.55, 0.60, 0.70; T = 0.50 from stage B'''. d/dT by central differences of the
undriven B = 0.12 series (0.40 ... 0.75). Writes results/c_summary.json."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from b4_analysis import deriv, predict, driven, ratio, SL

TAUS = (5.0, 10.0, 20.0)
B12 = {0.40: "results/ladder/ladder_T0.40", 0.45: "results/ladder/ladder_T0.45", 0.50: "results/main/P7",
       0.55: "results/ladder/ladder_T0.55", 0.60: "results/ladder/ladder_T0.60",
       0.65: "results/c/C_ladder_T0.65_B0.12", 0.70: "results/c/C_ladder_T0.70_B0.12",
       0.75: "results/c/C_ladder_T0.75_B0.12"}
NEIGH = {0.45: (0.40, 0.50), 0.50: (0.45, 0.55), 0.55: (0.50, 0.60), 0.60: (0.55, 0.65), 0.70: (0.65, 0.75)}
done = lambda d: os.path.exists(os.path.join(d, "summary.json")) and json.load(open(os.path.join(d, "summary.json")))["phase"] == "done"


def anti(p, m):
    return {k: ((p[k][0] - m[k][0]) / 2, float(np.hypot(p[k][1], m[k][1]) / 2)) for k in SL}


out = {}
for tau in TAUS:
    o = out[f"tau{tau:g}"] = {}
    for T in (0.45, 0.50, 0.55, 0.60, 0.70):
        tag = f"T{T:.2f}"
        if T == 0.50:
            pB = ("results/b3/P16_B0.11", "results/b3/P16_B0.13")
            dr = [("results/b2/P10", "results/b2/P10b"), ("results/b3/P18_P10", "results/b3/P18_P10b")]
        else:
            pB = (f"results/c/C_{tag}_B0.11", f"results/c/C_{tag}_B0.13")
            dr = [(f"results/c/C_{tag}_rotp", f"results/c/C_{tag}_rotm")]
        lo, hi = NEIGH[T]
        need = list(pB) + [x for pair in dr for x in pair] + [B12[lo], B12[hi]]
        if not all(done(d) for d in need):
            o[tag] = dict(status="incomplete", missing=[d for d in need if not done(d)]); continue
        dB = deriv(pB[0], pB[1], 0.02, tau, paired=True)
        dT = deriv(B12[lo], B12[hi], hi - lo, tau, paired=False)
        pr = predict(dB, dT, 0.12, T)
        # antisymmetric part of the rotation drive, combined over seeds (inverse variance)
        parts = [anti(driven(p, tau), driven(m, tau)) for p, m in dr]
        meas = {}
        for k in SL:
            v = np.array([x[k][0] for x in parts]); e = np.array([x[k][1] for x in parts]); w = 1 / e ** 2
            meas[k] = (float((w * v).sum() / w.sum()), float(1 / np.sqrt(w.sum())))
        nsk = []
        for p, m in dr:
            for d in (p, m):
                s = json.load(open(os.path.join(d, "summary.json")))
                nsk.append(float(np.sum(s["sk_time"]) / s["t_meas"] / s["config"]["R"]))
        o[tag] = dict(T=T, dB=dB, dT=dT, dT_points=(lo, hi), pred=pr, meas=meas,
                      F_pop=ratio(meas["s_pop"], pr["pop"]), F_pop_direct=ratio(meas["s_dens"], pr["pop_direct"]),
                      F_nuc=ratio(meas["s_nuc"], pr["nuc"]), F_sad=ratio(meas["s_sad"], pr["sad"]),
                      dM_T=(T * dB["lnn_ratio"][0], T * dB["lnn_ratio"][1]),
                      dM_T_direct=(T * dB["lnn_direct"][0], T * dB["lnn_direct"][1]),
                      sk_per_replica_driven=float(np.mean(nsk)), spacing=float(64 / np.sqrt(np.mean(nsk))))
    # trend of F_pop with T (weighted linear fit)
    pts = [(v["T"], *v["F_pop"]) for v in o.values() if isinstance(v, dict) and "F_pop" in v]
    if len(pts) >= 3:
        Tm = np.array([p[0] for p in pts]); F = np.array([p[1] for p in pts]); e = np.array([p[2] for p in pts])
        X = np.vstack([np.ones_like(Tm), Tm - 0.5]).T; W = 1 / e ** 2
        cov = np.linalg.inv(X.T @ (W[:, None] * X)); b = cov @ (X.T @ (W * F))
        o["trend"] = dict(F_at_0p5=float(b[0]), dF_dT=float(b[1]), d_dF_dT=float(np.sqrt(cov[1, 1])),
                          chi2=float((W * (F - X @ b) ** 2).sum()), npts=len(pts))
json.dump(out, open("results/c_summary.json", "w"), indent=1, default=float)
for tau in TAUS:
    print(f"===== tau_c = {tau:g}")
    for k, v in out[f"tau{tau:g}"].items():
        if k == "trend":
            print("trend", {a: round(b, 3) for a, b in v.items()}); continue
        if v.get("status") == "incomplete":
            print(k, "incomplete", len(v["missing"])); continue
        print(k, "F_pop %.2f±%.2f F_pop_dir %.2f±%.2f F_nuc %.2f±%.2f F_sad %.2f±%.2f" % (*v["F_pop"], *v["F_pop_direct"], *v["F_nuc"], *v["F_sad"]),
              "| s_pop %.1f±%.1f pred %.1f±%.1f | dM_T %.1f±%.1f | sk/rep %.2f spacing %.0f" % (
                  *v["meas"]["s_pop"], v["pred"]["pop"]["pred"], v["pred"]["pop"]["err"], *v["dM_T"], v["sk_per_replica_driven"], v["spacing"]))
