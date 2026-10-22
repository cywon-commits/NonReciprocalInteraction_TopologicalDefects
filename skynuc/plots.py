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
