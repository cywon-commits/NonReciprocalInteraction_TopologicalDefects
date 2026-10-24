from style import *
import numpy as np, json
from scipy.special import i0, k0
fig, axs = plt.subplots(1, 2, figsize=(W1*1.95, 2.2), gridspec_kw=dict(wspace=0.38))
ax = axs[0]
a = 1.0
L = np.logspace(0.8, 3.7, 200)
for R0, c in ((3, C["blue"]), (8, C["red"])):
    z0, za = R0/(2*L), a/(2*L)
    ax.plot(L, 1 - i0(z0)*k0(z0)/k0(za), color=c, lw=1.2)
    ax.plot(L, np.log(R0)/(np.log(4*L) - np.euler_gamma), color=c, ls="--", lw=0.9)
    ax.plot(L, np.log(R0)/np.log(L), color=c, ls=":", lw=0.9)
ax.plot([10, 30, 100, 1000, 3000], [0.347, 0.261, 0.199, 0.143, 0.125], "o", color=C["blue"], label="$R_0=3a$")
ax.plot([100], [0.382], "s", color=C["red"], label="$R_0=8a$")
ax.plot([], [], color="k", label="exact $1-I_0K_0/K_0$")
ax.plot([], [], color="k", ls="--", label="$\\ln(R_0/a)/[\\ln(4\\ell_u/a)-\\gamma_E]$")
ax.plot([], [], color="k", ls=":", label="$\\ln(R_0/a)/\\ln(\\ell_u/a)$")
ax.set_xscale("log"); ax.set_xlabel("$\\ell_u/a=k_BT/(G|u_{\\rm eff}|a)$"); ax.set_ylabel("escape probability $P_{\\rm esc}$")
ax.set_ylim(0.05, 0.75); ax.set_xlim(6, 5000)
ax.legend(loc="upper right", fontsize=6.3, handlelength=1.8)
ax.text(-0.2, 1.02, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
ax = axs[1]
runs = [json.loads(l) for f in ("nf_runs_G1.jsonl", "nf_runs_G2.jsonl") for l in open("/home/claude/pkg/results/step3_gas/" + f)]
xinf = 1500
for d in runs:
    x = d["ell"] if np.isfinite(d["ell"]) else xinf
    base = (d["Gamma"] == 2.2e-08)
    ax.errorbar(x, d["ratio"], d["ratio_err"], fmt="o" if base else "D", color=C["blue"], mfc=C["blue"] if base else "white", ms=4)
ax.axhline(1, color="k", lw=0.8)
ax2 = ax.twinx()
ll = np.logspace(0.6, 2.6, 100); z0, za = 3/(2*ll), 1/(2*ll)
ax2.plot(ll, 1 - i0(z0)*k0(z0)/k0(za), color=C["orange"], lw=1.0)
ax2.set_ylabel("$P_{\\rm esc}$ (theory)", color=C["orange"]); ax2.tick_params(axis="y", colors=C["orange"])
ax2.set_ylim(0, 0.6)
ax.set_xscale("log"); ax.set_xlim(3, 2500)
ax.set_xticks([10, 100, 1500]); ax.set_xticklabels(["$10$", "$100$", "$\\infty$"])
ax.set_ylim(0.85, 1.12)
ax.set_xlabel("$\\ell_u/a$"); ax.set_ylabel("free density $n_f/n_{\\rm pred}$", color=C["blue"])
ax.plot([], [], "o", color=C["blue"], label="$\\Gamma_0$"); ax.plot([], [], "D", color=C["blue"], mfc="white", label="$4\\Gamma_0$")
ax.legend(loc="upper left", ncol=2, handlelength=1.0)
ax.text(30, 0.875, "$n_f$ independent of drive", fontsize=7, color=C["blue"], ha="center")
ax.text(-0.22, 1.02, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
save(fig, "fig5_escape")
