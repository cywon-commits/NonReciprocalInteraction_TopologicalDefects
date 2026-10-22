#!/usr/bin/env python3
"""Stage B'' analysis: s_rot^pred from the rate's B and T sensitivity (P16, P17, P15), P18-P20 slopes.
Writes results/b3_summary.json and prints a summary."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from skynuc.analysis import object_counts, analyse_objects

TAU = 10.0
J = 1.0


def rate_blocks(d):
    """Nucleation counts and exposure per (replica, time block), summed over bands."""
    _, cnt, E, _ = object_counts(d, TAU)
    return cnt["nuc"].sum(axis=2), E["nuc"].sum(axis=2)


def dln_pair(d1, d2, dx):
    """(ln Gamma(d2) - ln Gamma(d1)) / dx with a paired delete-one-block jackknife (common seeds)."""
    n1, e1 = rate_blocks(d1); n2, e2 = rate_blocks(d2)
    f = lambda a, b, c, e: (np.log(c / e) - np.log(a / b)) / dx
    full = f(n1.sum(), e1.sum(), n2.sum(), e2.sum())
    vals = []
    R, K = n1.shape
    for r in range(R):
        for k in range(K):
            vals.append(f(n1.sum() - n1[r, k], e1.sum() - e1[r, k], n2.sum() - n2[r, k], e2.sum() - e2[r, k]))
    vals = np.array(vals); n = len(vals)
    err = np.sqrt((n - 1) / n * ((vals - vals.mean()) ** 2).sum())
    pois = np.sqrt(1 / n1.sum() + 1 / n2.sum()) / abs(dx)
    return float(full), float(err), float(pois), int(n1.sum()), int(n2.sum())


def lnG(d):
    n, e = rate_blocks(d)
    return float(np.log(n.sum() / e.sum())), float(1 / np.sqrt(n.sum())), int(n.sum())


def slope(dirs):
    o = analyse_objects(dirs, TAU)
    return dict(s=o["nuc"]["s_glm"], ds=o["nuc"]["ds_jk"], N=o["N"]["nuc"], s_col=o["col"]["s_glm"], ds_col=o["col"]["ds_jk"])


def wmean(*xs):
    v = np.array([x[0] for x in xs]); e = np.array([x[1] for x in xs]); w = 1 / e ** 2
    return float((w * v).sum() / w.sum()), float(1 / np.sqrt(w.sum()))


def pred(dlnB, dlnT, B, T):
    r = -1.0 / (4 * J + B)
    t1 = -0.5 * dlnB[0]; t2 = r * T / 2 * dlnT[0]
    err = float(np.hypot(0.5 * dlnB[1], r * T / 2 * dlnT[1]))
    return dict(term_B=t1, term_B_err=0.5 * dlnB[1], term_T=t2, term_T_err=abs(r * T / 2) * dlnT[1],
                s_pred=t1 + t2, s_pred_err=err, r=r)


out = {}
ex = lambda d: os.path.exists(os.path.join(d, "summary.json")) and json.load(open(os.path.join(d, "summary.json")))["phase"] == "done"
b3 = lambda n: f"results/b3/{n}"

# ---- D = 0.3: P16 (B) and P15 ladder (T) ----
if ex(b3("P16_B0.11")) and ex(b3("P16_B0.13")):
    dB = dln_pair(b3("P16_B0.11"), b3("P16_B0.13"), 0.02)
    lo = lnG("results/ladder/ladder_T0.45"); hi = lnG("results/ladder/ladder_T0.55")
    dT = ((hi[0] - lo[0]) / 0.1, float(np.hypot(lo[1], hi[1]) / 0.1))
    p = pred(dB, dT, 0.12, 0.5)
    seg = {}
    if ex(b3("P16_B0.12")):
        seg["dlnG_dB_lower"] = dln_pair(b3("P16_B0.11"), b3("P16_B0.12"), 0.01)
        seg["dlnG_dB_upper"] = dln_pair(b3("P16_B0.12"), b3("P16_B0.13"), 0.01)
    meas = (20.1, 4.4)  # beta = alpha shear group (P5, P5b, P5_u2x), stage B
    out["D0.3"] = dict(dlnG_dB=dB, dlnG_dT=dT, **p, **seg, s_meas_P5group=meas,
                       ratio=meas[0] / p["s_pred"], ratio_err=meas[1] / p["s_pred"])

# ---- P18: rotation-only drive, all statistics ----
if ex(b3("P18_P10")) and ex(b3("P18_P10b")):
    a = slope(["results/b2/P10"]); a2 = slope([b3("P18_P10")])
    b = slope(["results/b2/P10b"]); b2 = slope([b3("P18_P10b")])
    sp = wmean((a["s"], a["ds"]), (a2["s"], a2["ds"])); sm = wmean((b["s"], b["ds"]), (b2["s"], b2["ds"]))
    out["P18"] = dict(P10_runs=[a, a2], P10b_runs=[b, b2], P10_comb=sp, P10b_comb=sm,
                      antisym=((sp[0] - sm[0]) / 2, float(np.hypot(sp[1], sm[1]) / 2)),
                      sym=((sp[0] + sm[0]) / 2, float(np.hypot(sp[1], sm[1]) / 2)))
    if "D0.3" in out:
        out["P18"]["ratio_antisym_to_pred"] = out["P18"]["antisym"][0] / out["D0.3"]["s_pred"]

# ---- P19: alpha = beta = 0.05 ----
if ex(b3("P19_shear_a0.05")):
    s_new = slope([b3("P19_shear_a0.05")]); s_old = slope(["results/b2/P12_a0.05"])
    out["P19_shear"] = dict(new=s_new, P12_old=s_old, comb=wmean((s_new["s"], s_new["ds"]), (s_old["s"], s_old["ds"])))
if ex(b3("P19_rot_a0.05")):
    out["P19_rot"] = slope([b3("P19_rot_a0.05")])

# ---- P20: period L/3 ----
if all(ex(b3(f"P20_{t}_s{s}")) for t in ("b0", "ba") for s in (1, 2)):
    s0 = slope([b3("P20_b0_s1"), b3("P20_b0_s2")]); sa = slope([b3("P20_ba_s1"), b3("P20_ba_s2")])
    diff = (s0["s"] - sa["s"], float(np.hypot(s0["ds"], sa["ds"])))
    ref = (29.8 - 20.1, float(np.hypot(3.4, 4.4)))  # single period L: beta=0 group minus beta=alpha group
    out["P20"] = dict(beta0=s0, beta_alpha=sa, diff=diff, diff_single_period=ref,
                      diff_minus_ref=(diff[0] - ref[0], float(np.hypot(diff[1], ref[1]))))

# ---- P17: D = 0.2, 0.4 ----
for D, Bc in ((0.2, 0.2 ** 2 * 4 / 3), (0.4, 0.4 ** 2 * 4 / 3)):
    n = lambda T, s: b3(f"P17_D{D:.1f}_T{T:.2f}_B{s}")
    if not all(ex(n(T, s)) for T in (0.5, 0.55) for s in "mp"):
        if all(ex(n(0.5, s)) for s in "mp"):
            out[f"D{D}"] = dict(partial=True, dlnG_dB=dln_pair(n(0.5, "m"), n(0.5, "p"), 2 * (0.005 if D == 0.2 else 0.01)))
        continue
    dBB = 0.005 if D == 0.2 else 0.01
    dB = dln_pair(n(0.5, "m"), n(0.5, "p"), 2 * dBB)
    l5 = [lnG(n(0.5, s)) for s in "mp"]; l55 = [lnG(n(0.55, s)) for s in "mp"]
    m5 = np.mean([x[0] for x in l5]); m55 = np.mean([x[0] for x in l55])
    e = np.sqrt(sum(x[1] ** 2 for x in l5 + l55)) / 2
    dT = ((m55 - m5) / 0.05, float(e / 0.05))
    p = pred(dB, dT, Bc, 0.5)
    meas = {0.2: (24.9, 13.8), 0.4: (7.8, 8.8)}[D]  # P11 (stage B'), beta = alpha shear
    out[f"D{D}"] = dict(B=Bc, dB=dBB, dlnG_dB=dB, dlnG_dT=dT, **p, s_meas_P11=meas,
                        ratio=meas[0] / p["s_pred"], ratio_err=meas[1] / p["s_pred"])

json.dump(out, open("results/b3_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=lambda x: round(float(x), 3)))
