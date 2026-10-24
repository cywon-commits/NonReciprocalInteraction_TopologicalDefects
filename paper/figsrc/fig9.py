from style import *
import numpy as np, matplotlib
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.15), gridspec_kw=dict(wspace=0.5, width_ratios=[1.25, 0.8, 1.1, 0.8]))
# (a)
ax = axs[0]
x = np.array([-1, 0, 1])
ax.errorbar(x - 0.05, [29.8, 20.1, -6.2], [3.4, 4.4, 4.3], fmt="o", color=C["blue"], label="$\\alpha=0.1$")
ax.errorbar(x + 0.05, [25.5, 9.7, -0.4], [10.4, 12.7, 10.2], fmt="o", mfc="white", color=C["blue"], label="$\\alpha=0.3$")
xx = np.linspace(-1.3, 1.3, 10)
ax.fill_between(xx, 14 - 17*xx - 3, 14 - 17*xx + 3, color="#dbe7f5", lw=0)
ax.plot(xx, 14 - 17*xx, color=C["black"], lw=0.9, label="$s_{\\rm rot}+k(\\beta-\\alpha)/\\alpha$")
ax.errorbar([1.6], [-6.7], [10.7], fmt="D", color=C["gray"], ms=3.5, label="uniform $u$ ($\\omega=0$)")
ax.errorbar([1.85], [-12.5], [6.8], fmt="v", color=C["gray"], ms=3.5, label="no drive")
ax.axhline(0, color="#bbbbbb", lw=0.5); ax.axvline(0, color="#bbbbbb", lw=0.5, ls=":")
ax.set_xlim(-1.35, 2.1); ax.set_ylim(-45, 48)
ax.set_xticks([-1, 0, 1]); ax.set_xlabel("$(\\beta-\\alpha)/\\alpha$"); ax.set_ylabel("$s_{\\rm nuc}=d\\ln\\Gamma_{\\rm nuc}/d\\omega$")
ax.annotate("residual at $\\beta=\\alpha$", xy=(0, 20), xytext=(-1.25, 41), fontsize=6.3, arrowprops=dict(arrowstyle="-", lw=0.5))
ax.legend(loc="lower left", fontsize=5.6, handlelength=1.1, borderaxespad=0.2, labelspacing=0.25)
ax.text(-0.3, 1.02, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
# (b) rotation only
ax = axs[1]
ax.errorbar([0], [20.1], [4.4], fmt="o", color=C["blue"])
ax.errorbar([1], [20.7], [6.1], fmt="s", color=C["red"])
ax.errorbar([2], [-19.7], [5.8], fmt="s", mfc="white", color=C["red"])
for xc, sgn in ((0, 1), (1, 1), (2, -1)):
    ax.plot([xc - 0.3, xc + 0.3], [27.3*sgn]*2, color=C["orange"], lw=1.2)
ax.axhline(0, color="#bbbbbb", lw=0.5)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["shear\n$\\beta=\\alpha$", "$-\\Omega\\hat z\\times$\n$u=0$", "$+\\Omega\\hat z\\times$\n$u=0$"], fontsize=6)
ax.set_xlim(-0.6, 2.6); ax.set_ylim(-32, 48); ax.set_ylabel("$s_{\\rm nuc}$")
ax.text(-0.42, 1.02, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
# (c) alpha dependence
ax = axs[2]
ar = [0.025, 0.05, 0.1]; sr = [17.8, 14.9, 20.7]; er = [7.5, 7.2, 6.1]
asb = [0.025, 0.05, 0.1, 0.2, 0.3]; ss = [11.0, 0.7, 20.1, 18.0, 9.7]; es = [7.9, 6.0, 4.4, 11.8, 12.7]
ax.errorbar(np.array(ar)*0.93, sr, er, fmt="s", color=C["red"], label="rotation only")
ax.errorbar(np.array(asb)*1.07, ss, es, fmt="o", color=C["blue"], label="shear, $\\beta=\\alpha$")
aa = np.logspace(np.log10(0.02), np.log10(0.35), 50)
ax.axhline(27.3, color=C["orange"], lw=1.1)
ax.text(0.38, 32, "prediction", ha="right", fontsize=6, color=C["orange"])
ax.plot(aa, 20*0.1/aa, color=C["gray"], ls=":", lw=0.9)
ax.text(0.047, 55, "$\\propto1/\\alpha$", fontsize=6.5, color=C["gray"])
ax.set_xscale("log"); ax.set_xlim(0.02, 0.4); ax.set_ylim(-30, 62); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([0.03, 0.1, 0.3]); ax.set_xticklabels(["0.03", "0.1", "0.3"])
ax.set_xlabel("$\\alpha=\\beta$"); ax.set_ylabel("$s_{\\rm nuc}$")
ax.legend(loc="lower left", fontsize=5.8, handlelength=1.2, borderaxespad=0.2)
ax.text(-0.3, 1.02, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold")
# (d) L dependence
ax = axs[3]
ax.errorbar([32, 64, 128], [39.8, 29.8, 39.2], [10.4, 3.4, 6.3], fmt="o", color=C["blue"])
ax.axhline(29.8, color="#9fbfe0", lw=0.8, ls="--")
ax.set_xscale("log", base=2); ax.set_xticks([32, 64, 128]); ax.set_xticklabels(["32", "64", "128"])
ax.set_xlim(24, 170); ax.set_ylim(-30, 62); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("$L/a$ (fixed $\\omega$)"); ax.set_ylabel("$s_{\\rm nuc}$ ($\\beta=0$)")
ax.text(64, 6, "$\\psi$ amplitude $\\times16$\nfrom $L=32$ to $128$", ha="center", fontsize=6)
ax.text(-0.42, 1.02, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold")
save(fig, "fig9_nucleation")
