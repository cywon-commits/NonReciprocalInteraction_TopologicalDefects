from style import *
import numpy as np
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw=dict(wspace=0.45, width_ratios=[1.15, 1, 1]))
T = np.array([0.45, 0.5, 0.55, 0.6])
pp = np.array([44.4, 36.7, 29.0, 21.3]); pm = np.array([36.6, 30.4, 26.5, 16.6]); pe = np.array([9.5, 5.0, 3.9, 3.5])
Fn = np.array([0.75, 0.75, 0.89, 0.71]); nm = np.array([21.2, 20.4, 19.6, 12.3]); ne = np.array([7.3, 4.2, 3.1, 2.8]); npred = nm/Fn
ax = axs[0]
cmap = plt.get_cmap("viridis")
for i in range(4):
    c = cmap(0.1 + 0.25*i)
    ax.errorbar(pp[i], pm[i], pe[i], fmt="o", color=c)
    ax.errorbar(npred[i], nm[i], ne[i], fmt="s", mfc="white", color=c)
ax.errorbar([36.7, 25.0], [21.8, 26.7], [16.5, 11.6], fmt="D", color=C["red"], ms=3.5)
ax.errorbar([39.9, 14.3], [24.9, 7.8], [13.8, 8.8], fmt="D", mfc="white", color=C["red"], ms=3.5)
xx = np.linspace(0, 55, 10)
ax.plot(xx, xx, color="k", lw=0.8); ax.plot(xx, 0.85*xx, color=C["gray"], ls="--", lw=0.8)
ax.text(47, 49, "$y=x$", fontsize=6.5, ha="right"); ax.text(49.5, 33.0, "$0.85x$", fontsize=6.5, color=C["gray"], ha="left")
ax.set_xlim(0, 55); ax.set_ylim(-5, 55)
ax.set_xlabel("weak-damping prediction"); ax.set_ylabel("measured slope")
ax.plot([], [], "o", color="k", label="population $s_{\\rm nuc}-s_{\\rm col}$")
ax.plot([], [], "s", color="k", mfc="white", label="nucleation $s_{\\rm nuc}$")
ax.plot([], [], "D", color=C["red"], ms=3.5, label="$D=0.2J,\\,0.4J$")
ax.legend(loc="upper left", fontsize=6, handlelength=1.0, borderaxespad=0.2)
ax.text(-0.25, 1.02, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
# colorbar-like legend for T
for i in range(4): ax.plot([], [], "o", color=cmap(0.1 + 0.25*i))
ax.text(54, 3, "colour: $k_BT/J=0.45\\ldots0.60$", fontsize=6, ha="right", color=cmap(0.45))
ax = axs[1]
Fp = np.array([0.83, 0.83, 0.92, 0.78]); Fpe = np.array([0.22, 0.14, 0.14, 0.17])
ax.axhspan(0.77, 0.93, color="#e4e4e4", lw=0)
ax.axhline(0.85, color=C["gray"], ls="--", lw=0.8); ax.axhline(1, color="k", lw=0.8)
ax.errorbar(T, Fp, Fpe, fmt="o", color=C["blue"], label="$F_{\\rm pop}$ (stationary measure)")
ax.errorbar(T + 0.006, Fn, [0.26, 0.16, 0.15, 0.16], fmt="s", mfc="white", color=C["blue"], label="$F_{\\rm nuc}$ (rate)")
ax.set_xlim(0.42, 0.63); ax.set_ylim(0.3, 1.4)
ax.set_xlabel("$k_BT/J$"); ax.set_ylabel("measured / predicted")
ax.text(0.425, 0.36, "band: mean $F_{\\rm pop}=0.85\\pm0.08$", fontsize=6.3, color=C["gray"], ha="left")
ax.legend(loc="upper left", fontsize=6, handlelength=1.0, borderaxespad=0.2)
ax.text(-0.25, 1.02, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
ax = axs[2]
Tm = np.array([0.45, 0.5, 0.55, 0.6, 0.7]); dM = np.array([-41.2, -37.7, -32.5, -25.7, -13.7]); dMe = [1.1, 1.4, 1.2, 1.2, 1.4]
ax.errorbar(Tm[:4], dM[:4], dMe[:4], fmt="o", color=C["green"], label="$\\Delta\\langle M\\rangle_T=k_BT\\,\\partial\\ln n_0/\\partial B$")
ax.errorbar(Tm[4:], dM[4:], dMe[4:], fmt="o", mfc="white", color=C["green"], label="not dilute ($k_BT=0.7J$)")
ax.axhline(-67.7, color="k", ls="--", lw=0.8)
ax.text(0.43, -64, "$T=0$ skyrmion: $\\Delta M_0=-67.7$", fontsize=6.3, va="bottom")
ax.set_xlim(0.42, 0.73); ax.set_ylim(-75, 0)
ax.set_xlabel("$k_BT/J$"); ax.set_ylabel("magnetization deficit")
ax.legend(loc="upper left", fontsize=5.8, handlelength=1.0, borderaxespad=0.2)
ax.text(-0.27, 1.02, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold")
save(fig, "fig10_magnitude")
