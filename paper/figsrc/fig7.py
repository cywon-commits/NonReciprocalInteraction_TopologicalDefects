from style import *
import numpy as np, matplotlib
fig = plt.figure(figsize=(W2, 2.2))
x = np.linspace(0, 2*np.pi, 160); X, Y = np.meshgrid(x, x); psi = np.sin(X)*np.sin(Y); b = 1.5
titles = ["$n_+/n_0=e^{-b\\psi}$", "$n_-/n_0=e^{+b\\psi}$", "$n_+n_-/n_0^2=1$"]
fields = [np.exp(-b*psi), np.exp(b*psi), np.ones_like(psi)]
for i, (t, f) in enumerate(zip(titles, fields)):
    ax = fig.add_axes([0.005 + i*0.135, 0.2, 0.12, 0.6])
    im = ax.imshow(f, origin="lower", extent=[0, 2, 0, 2], cmap="RdBu_r", norm=matplotlib.colors.LogNorm(vmin=np.exp(-b), vmax=np.exp(b)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=7, pad=3)
cax = fig.add_axes([0.03, 0.12, 0.36, 0.03])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.25, 0.5, 1, 2, 4]); cb.ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); cb.ax.tick_params(labelsize=6); cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
cb.set_label("density (cellular flow, $b=G\\psi_{\\max}/k_BT=1.5$)", fontsize=6.5)
fig.text(0.005, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")
# (b) density ratio
ax = fig.add_axes([0.5, 0.2, 0.2, 0.66])
bb = np.linspace(0, 3.3, 60)
xx = np.linspace(0, 2*np.pi, 400); XX, YY = np.meshgrid(xx, xx); P = np.sin(XX)*np.sin(YY)
mean = [np.mean(np.exp(-v*P)) for v in bb]
ax.plot(bb, mean, color="k", label="$\\langle e^{-G\\psi_e/k_BT}\\rangle$")
ax.plot([0.5, 1.5, 3.0], [1.050, 1.318, 2.505], "o", color=C["blue"], label="particle gas")
ax.set_xlabel("$b$"); ax.set_ylabel("$\\bar n/\\bar n(b=0)$"); ax.set_xlim(0, 3.3); ax.set_ylim(0.9, 3.0)
ax.legend(loc="upper left", handlelength=1.4)
fig.text(0.445, 0.985, "(b)", fontsize=9, fontweight="bold", va="top")
# (c) crossover
ax = fig.add_axes([0.79, 0.2, 0.2, 0.66])
kl = np.logspace(-1, 1, 100)
ax.plot(kl, kl**2/(1 + kl**2), color="k", label="$k^2/(k^2+\\ell_\\nu^{-2})$")
ax.plot([0.5, 1.0], [0.17, 0.48], "o", color=C["blue"], label="particle gas")
ax.plot([9.0], [1.0], "o", color=C["blue"])
ax.text(6.0, 0.9, "$\\nu=0$", fontsize=6.5, ha="center", va="top")
ax.set_xscale("log"); ax.set_xlabel("$k_c\\ell_\\nu$"); ax.set_ylabel("charge amplitude / ($\\nu=0$)")
ax.set_ylim(0, 1.12); ax.set_xlim(0.1, 10)
ax.legend(loc="upper left", handlelength=1.4, fontsize=6.3)
fig.text(0.735, 0.985, "(c)", fontsize=9, fontweight="bold", va="top")
save(fig, "fig7_segregation")
