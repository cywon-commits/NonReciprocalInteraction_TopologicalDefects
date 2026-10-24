from style import *
import numpy as np, json
from matplotlib.patches import Circle, FancyArrowPatch
base = "/home/claude/pkg/results/step3_gas/"
sets = [("corr_l5R10.json", "$\\ell_u=5a,\\ R_0=10a$", C["red"]), ("corr_l5R5.json", "$\\ell_u=5a,\\ R_0=5a$", C["blue"]),
        ("corr_l15R3.json", "$\\ell_u=15a,\\ R_0=3a$", C["green"]), ("corr_linfR5.json", "no drive, $R_0=5a$", C["gray"])]
fig = plt.figure(figsize=(W2, 2.3))
# (a) schematic
ax = fig.add_axes([0.0, 0.05, 0.25, 0.85]); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.3, 2.6); ax.set_aspect("equal"); ax.axis("off")
yy, xx = np.mgrid[-2.4:2.4:200j, -2.4:2.4:200j]; r = np.hypot(xx, yy); th = np.arctan2(yy, xx)
from scipy.special import k0
shadow = -np.exp(r*np.cos(th))*k0(np.maximum(r, 0.35))*0.5
shadow[r < 0.35] = np.nan
ax.contourf(xx, yy, shadow, levels=np.linspace(-0.9, -0.04, 9), cmap="Greys_r", alpha=0.55)
ax.add_patch(Circle((0, 0), 0.35, fc="white", ec="k", lw=0.8))
ax.text(0, 0, "$-$", ha="center", va="center", fontsize=8)
ax.add_patch(FancyArrowPatch((-2.2, 2.1), (-0.8, 2.1), arrowstyle="-|>", mutation_scale=8, color=C["red"], lw=1.0))
ax.text(-1.5, 2.25, "$\\mathbf{v}_{\\rm rel}$", ha="center", va="bottom", fontsize=8, color=C["red"])
ax.text(1.35, -1.6, "depleted wake\n(shadow)", ha="center", va="center", fontsize=7)
ax.text(-1.55, -1.6, "relative\nexcess", ha="center", va="center", fontsize=7)
fig.text(0.005, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")
# (b) g0 and (c) g1
axb = fig.add_axes([0.32, 0.2, 0.3, 0.68]); axc = fig.add_axes([0.69, 0.2, 0.3, 0.68])
for f, lab, c in sets:
    d = json.load(open(base + f)); rc = np.array(d["rc"])
    axb.errorbar(rc, d["g0"], d["g0_err"], fmt="o", ms=2.3, color=c, elinewidth=0.5)
    axb.plot(rc, d["g0_th"], color=c, lw=0.9, label=lab)
    axc.errorbar(rc, d["g1"], d["g1_err"], fmt="o", ms=2.3, color=c, elinewidth=0.5)
    axc.plot(rc, d["g1_th"], color=c, lw=0.9)
axb.axhline(1, color="#bbbbbb", lw=0.5); axc.axhline(0, color="#bbbbbb", lw=0.5)
axb.set_xlabel("$r/a$"); axb.set_ylabel("isotropic part $g_0(r)$"); axb.set_xlim(0, 40); axb.set_ylim(0, 1.6)
axc.set_xlabel("$r/a$"); axc.set_ylabel("dipole $g_1(r)$ (along $\\mathbf{v}_{\\rm rel}$)"); axc.set_xlim(0, 40); axc.set_ylim(-0.22, 0.06)
axb.legend(loc="lower right", fontsize=6.3, handlelength=1.4)
axc.text(20, -0.205, "points: many-body gas; lines: exact two-body", ha="center", fontsize=6.5)
fig.text(0.275, 0.985, "(b)", fontsize=9, fontweight="bold", va="top")
fig.text(0.645, 0.985, "(c)", fontsize=9, fontweight="bold", va="top")
save(fig, "fig6_paircorr")
