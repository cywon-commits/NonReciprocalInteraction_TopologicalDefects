"""Figures for the report (matplotlib, static PNG)."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "legend.frameon": False, "lines.linewidth": 2,
                     "figure.dpi": 130})
FIG = "results/figures"


def fig_validation():
    os.makedirs(FIG, exist_ok=True)
    V = "results/validation"
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    v1 = [json.load(open(f"{V}/V1_a{a}.json")) for a in ("0.1", "0.5")]
    rel = [100 * x["rel_dev"] for x in v1]; err = [100 * x["mz_err"] / x["exact"] for x in v1]
    ax[0].bar([0, 1], rel, yerr=err, color=C[0], width=0.5)
    ax[0].axhspan(-1, 1, color=C[2], alpha=0.12, lw=0)
    ax[0].set_xticks([0, 1], ["α=0.1", "α=0.5"]); ax[0].set_ylabel("⟨S_z⟩ deviation from exact (%)")
    ax[0].set_title("V1  coth(B/T) − T/B = 0.3130", fontsize=9)
    v2 = json.load(open(f"{V}/V2.json"))
    ax[1].bar([0, 1], [100 * v2["rel_dE"], 100 * v2["rel_dmz"]], color=C[0], width=0.5)
    ax[1].axhspan(-0.5, 0.5, color=C[2], alpha=0.12, lw=0)
    ax[1].set_xticks([0, 1], ["E / site", "⟨S_z⟩"]); ax[1].set_ylabel("LLG − MC (%)")
    ax[1].set_title("V2  vs Metropolis MC (±0.5% band)", fontsize=9)
    v34 = json.load(open(f"{V}/V34.json")); lt = json.load(open(f"{V}/V34_lattice_thiele.json"))
    lab = ["β=α=0.1", "β=0", "β=0.2"]
    meas = np.array([[r["vx"], r["vy"]] for r in v34["runs"]]) / 0.02
    nom = np.array([[0.02, 0.0]] + [[x["vx_thiele"], x["vy_thiele"]] for x in v34["V4"]]) / 0.02
    lat = np.array([[x["vx_lattice_thiele"], x["vy_lattice_thiele"]] for x in lt["comparison"]]) / 0.02
    x = np.arange(3)
    for k, nm in enumerate(["v_x/u", "v_y/u"]):
        a = ax[2] if k == 0 else ax[2].twinx() if False else ax[2]
    ax[2].plot(x - 0.12, meas[:, 0], "o", color=C[0], ms=8, label="measured v_x/u")
    ax[2].plot(x, nom[:, 0], "s", mfc="none", color=INK2, ms=8, label="continuum Thiele (request)")
    ax[2].plot(x + 0.12, lat[:, 0], "^", mfc="none", color=C[1], ms=8, label="lattice Thiele")
    ax[2].plot(x - 0.12, 1 + 10 * meas[:, 1] - 1, "o", color=C[2], ms=8, label="measured 10·v_y/u")
    ax[2].plot(x, 10 * nom[:, 1], "s", mfc="none", color=INK2, ms=8)
    ax[2].plot(x + 0.12, 10 * lat[:, 1], "^", mfc="none", color=C[1], ms=8)
    ax[2].set_xticks(x, lab); ax[2].set_title("V3/V4  isolated skyrmion, u=0.02, α=0.1", fontsize=9)
    ax[2].legend(fontsize=7, loc="center right")
    sc = json.load(open(f"{V}/V3_scan.json"))
    pts = {}
    for r in sc:
        pts.setdefault(r["radius"], []).append(1 - r["v_over_u"])
    Rr = np.array(sorted(pts)); dd = np.array([np.mean(pts[r]) for r in Rr])
    ax[3].plot(1 / Rr ** 2, 100 * dd, "o-", color=C[0], ms=8)
    ax[3].axhline(1, color=C[7], lw=1, ls="--"); ax[3].text(0.09, 1.15, "1% tolerance", color=INK2, fontsize=7)
    for r, d in zip(Rr, dd):
        ax[3].annotate(f"R={r:.1f}", (1 / r ** 2, 100 * d), textcoords="offset points", xytext=(5, -10), fontsize=7, color=INK2)
    ax[3].set_xlabel("1/R²"); ax[3].set_ylabel("1 − v/u at β=α (%)")
    ax[3].set_title("V3 deficit: central-difference discretisation", fontsize=9)
    ax[3].set_xlim(0, None); ax[3].set_ylim(0, None)
    fig.tight_layout(); fig.savefig(f"{FIG}/validation.png"); plt.close(fig)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()["fig_" + name]()


def fig_points(dirs, tau_c=10.0, out="points.png", ncol=3):
    """ln Gamma_nuc(y) and ln nu_col(y) vs omega(y) for each run, with GLM fit lines."""
    import sys as _s
    _s.path.insert(0, ".")
    from skynuc.analysis import analyse_objects
    os.makedirs(FIG, exist_ok=True)
    n = len(dirs); nr = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nr, ncol, figsize=(4.2 * ncol, 3.0 * nr), squeeze=False)
    for ax, d in zip(axs.ravel(), dirs):
        o = analyse_objects(d, tau_c)
        om = np.array(o["omega_band"])
        for kind, col, lab in (("nuc", C[0], "nucleation  ln Γ"), ("col", C[1], "collapse  ln ν")):
            r = o[kind]
            if "lnrate_band" not in r:
                continue
            y = np.array([np.nan if v is None else v for v in r["lnrate_band"]])
            N = np.array(r["N_band"])
            yc = y - np.nanmean(y)
            err = 1 / np.sqrt(np.maximum(N, 1))
            ax.errorbar(om, yc, yerr=err, fmt="o", ms=5, color=col, elinewidth=1, capsize=0,
                        label=f"{lab}: s={r['s_glm']:+.1f}±{r['ds_jk']:.1f}")
            xx = np.linspace(om.min(), om.max(), 2)
            ax.plot(xx, r["s_glm"] * (xx - om.mean()), color=col, lw=1.5)
        ax.axhline(0, color=INK2, lw=0.6)
        ttl = f"{o['run_id']}  (β−α)/α={o['ueff_factor']:+.0f}, u₀={o['u0']:+.3g}"
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("ω(y)" if o["drive"] == "shear" else "reference ω(y) of the u₀=0.04 shear")
        ax.set_ylabel("log rate − mean")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(f"{FIG}/{out}"); plt.close(fig)


def _table():
    import csv as _c
    return [r for r in _c.DictReader(open("results/summary_table.csv"))]


def fig_summary(tau_c="10"):
    """Combined slopes (results/combined.json) vs (beta - alpha)/alpha, and s_col vs s_nuc."""
    comb = json.load(open("results/combined.json"))[f"tau{tau_c}"]
    fit = comb.get("fit s_nuc = s_adv + k x")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    style = {  # group-name fragment -> (colour, marker, label)
        "alpha=0.1 (P1": (C[0], "o", "α=0.1, L=64"), "beta=alpha=0.1 (P5": (C[0], "o", None),
        "beta=2alpha": (C[0], "o", None), "alpha=0.3": (C[1], "o", "α=0.3, L=64"),
        "L=128": (C[6], "D", "L=128, u₀=0.08 (same ω, 2u)"), "T=0.55": (C[2], "^", "T=0.55"),
        "T=0.45": (C[3], "v", "T=0.45")}
    seen = set()
    for g, d in comb.items():
        if "s_nuc" not in d or "reference" in g:
            continue
        x = (d["beta"] - d["alpha"]) / d["alpha"]
        key = next(k for k in style if k in g)
        col, mk, lab = style[key]
        lab = lab if lab and lab not in seen else None
        if lab:
            seen.add(lab)
        dx = {"o": 0, "D": 0.06, "^": -0.06, "v": -0.12}[mk]
        y, e = d["s_nuc"][0], d["s_nuc"][1]
        ax[0].errorbar(x + dx, y, yerr=e, fmt=mk, ms=7, color=col, elinewidth=1.2, label=lab)
        yc, ec = d["s_col"][0], d["s_col"][1]
        ax[1].errorbar(y, yc, xerr=e, yerr=ec, fmt=mk, ms=7, color=col, elinewidth=1.2)
    for g, d in comb.items():
        if "reference" in g:
            ax[0].errorbar(-1.25 if "uniform" in g else 1.25, d["s_nuc"][0], yerr=d["s_nuc"][1], fmt="s", ms=7,
                           color=C[7] if "uniform" in g else INK2, elinewidth=1.2,
                           label="P6 uniform u (ω=0)" if "uniform" in g else "P7 no drive")
    if fit:
        xx = np.linspace(-1.3, 1.3, 2)
        ax[0].plot(xx, fit["s_adv"] + fit["k"] * xx, color=INK2, lw=1, ls="--",
                   label=f"fit T=0.5: {fit['s_adv']:.0f} {fit['k']:+.0f}·(β−α)/α")
    ax[0].axhline(0, color=INK2, lw=0.8); ax[0].axvline(0, color=GRID, lw=0.8)
    ax[0].set_xlabel("(β − α)/α"); ax[0].set_ylabel(f"s_nuc = d ln Γ_nuc/dω  (τ_c={tau_c})")
    ax[0].set_title("Nucleation slope (= c·Q_fav, Q_fav = −1); controls at the edges", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper right")
    lim = 70
    xx = np.linspace(-20, lim, 2)
    ax[1].plot(xx, -xx, ls="--", color=INK2, lw=1, label="s_col = −s_nuc")
    ax[1].axhline(0, color=GRID, lw=0.8); ax[1].axvline(0, color=GRID, lw=0.8)
    ax[1].set_xlabel("s_nuc"); ax[1].set_ylabel("s_col = d ln ν/dω"); ax[1].legend(fontsize=7)
    ax[1].set_title("Collapse vs nucleation slope (combined groups)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/summary.png"); plt.close(fig)


def fig_ladder(tau_c="10"):
    rows = [r for r in _table() if r["tau_c"] == tau_c and float(r["u0"]) == 0 and r["drive"] == "none"
            and float(r["alpha"]) == 0.1 and float(r["dt"]) == 0.02]
    T = np.array([float(r["T"]) for r in rows]); G = np.array([float(r["Gamma0"]) for r in rows])
    nu = np.array([float(r["nu"]) for r in rows]); N = np.array([float(r["N_nuc"]) for r in rows])
    o = np.argsort(T); T, G, nu, N = T[o], G[o], nu[o], N[o]
    p = np.polyfit(1 / T, np.log(G), 1, w=np.sqrt(N))
    pn = np.polyfit(1 / T, np.log(nu), 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].errorbar(1 / T, np.log(G), yerr=1 / np.sqrt(N), fmt="o", color=C[0], ms=7)
    xx = np.linspace((1 / T).min(), (1 / T).max(), 2)
    ax[0].plot(xx, np.polyval(p, xx), color=C[0], lw=1.2, label=f"ΔE_b = {-p[0]:.2f} J")
    ax[0].set_xlabel("1/T"); ax[0].set_ylabel("ln Γ₀  (per site per unit time)"); ax[0].legend()
    ax[0].set_title(f"Undriven nucleation rate (τ_c = {tau_c})", fontsize=9)
    ax[1].plot(1 / T, np.log(nu), "o", color=C[1], ms=7)
    ax[1].plot(xx, np.polyval(pn, xx), color=C[1], lw=1.2, label=f"ΔE_col = {-pn[0]:.2f} J")
    ax[1].set_xlabel("1/T"); ax[1].set_ylabel("ln ν  (per skyrmion per unit time)"); ax[1].legend()
    ax[1].set_title("Collapse rate of established skyrmions", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/ladder.png"); plt.close(fig)
    return dict(dE_b=-p[0], dE_col=-pn[0], T=T.tolist(), Gamma0=G.tolist(), nu=nu.tolist(), N=N.tolist())
