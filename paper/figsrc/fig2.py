from style import *
import numpy as np, json
from matplotlib.patches import Polygon
ph = json.load(open("/home/claude/pkg/results/step2_highT/ph_T2.5.json"))
fig = plt.figure(figsize=(W2, 2.3))
# (a) schematic
ax = fig.add_axes([0.0, 0.02, 0.27, 0.86]); ax.set_xlim(-0.6, 2.1); ax.set_ylim(-0.8, 1.6); ax.set_aspect("equal"); ax.axis("off")
cols = [C["blue"], C["red"], C["green"], C["orange"]]
corners = [((0,0),(1,0),(0,1)), ((1,0),(1,1),(0,0)), ((1,1),(0,1),(1,0)), ((0,1),(0,0),(1,1))]
for (o, a, b), c in zip(corners, cols):
    o, a, b = map(np.array, (o, a, b)); s = 0.42
    tri = [o + 0.08*(a-o) + 0.08*(b-o), o + s*(a-o) + 0.08*(b-o), o + 0.08*(a-o) + s*(b-o)]
    ax.add_patch(Polygon(tri, closed=True, fc=c, alpha=0.35, ec=c, lw=0.7))
ax.plot([0,1,1,0,0],[0,0,1,1,0], color="#aaaaaa", lw=0.6, zorder=1)
for (x, y) in [(0,0),(1,0),(0,1),(1,1)]: ax.plot(x, y, "o", ms=4, color=C["black"], zorder=3)
ax.annotate("", xy=(0.88, 0), xytext=(0.12, 0), arrowprops=dict(arrowstyle="-|>", color=C["purple"], lw=1.2, mutation_scale=8))
ax.annotate("", xy=(0.88, 1), xytext=(0.12, 1), arrowprops=dict(arrowstyle="-|>", color=C["purple"], lw=1.2, mutation_scale=8))
ax.text(0.5, -0.1, "$d(y)$", ha="center", va="top", fontsize=8, color=C["purple"])
ax.text(0.5, 1.1, "$d(y+1)$", ha="center", va="bottom", fontsize=8, color=C["purple"])
ax.text(1.18, 0.5, "$\\sum_{\\rm corners}\\chi\\propto 2a\\,\\omega_p$", ha="left", va="center", fontsize=8)
ax.text(-0.5, -0.62, "corner triangles of one plaquette\nbelong to four different sites", ha="left", va="center", fontsize=7)
fig.text(0.005, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")
# (b) projections: dot plot
ax = fig.add_axes([0.36, 0.2, 0.3, 0.68])
names = ["UR", "UL", "LL", "LR"]
xs = np.arange(4); xc = np.arange(4) + 5
th_s = [1.052]*4; th_c = [-0.126, -0.126, 0.126, 0.126]
for x, t in zip(xs, th_s): ax.plot([x-0.32, x+0.32], [t, t], color=C["gray"], lw=1.4)
for x, t in zip(xc, th_c): ax.plot([x-0.32, x+0.32], [t, t], color=C["gray"], lw=1.4)
ax.plot(xs, [ph[n] for n in names], "o", color=C["blue"], label="LL simulation")
ax.plot(xc, [ph[n + "_cos"] for n in names], "o", color=C["red"])
ax.plot([], [], color=C["gray"], lw=1.4, label="theory (2nd order)")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(4, color="#cccccc", lw=0.5)
ax.set_xticks(list(xs) + list(xc)); ax.set_xticklabels(names + names, fontsize=6.5)
ax.set_xlim(-0.7, 8.7); ax.set_ylim(-0.45, 1.45)
ax.set_ylabel("response / $[J\\delta J/27\\alpha T^2]$")
ax.text(1.5, 0.72, "sine (row $y$)", ha="center", fontsize=7, color=C["blue"])
ax.text(6.5, 0.45, "cosine (row $y\\pm1$)", ha="center", fontsize=7, color=C["red"])
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14), handlelength=1.4, columnspacing=0.8)
fig.text(0.315, 0.985, "(b)", fontsize=9, fontweight="bold", va="top")
# (c) kappa(T)
ax = fig.add_axes([0.75, 0.2, 0.24, 0.68])
jt = np.linspace(0, 0.5, 50)
ax.plot(jt, 1 + 8/9*jt, color=C["black"], label="$1+\\frac{8}{9}J/k_BT$")
ax.axhline(1, color=C["gray"], ls=":", lw=0.8, label="leading order")
ax.plot([0.4], [ph["plaquette"]], "o", color=C["blue"], label="LL, $J/k_BT=0.4$")
ax.plot([0.2, 0.2], [1.06, 1.34], "s", mfc="white", mec=C["blue"], label="LL, two runs")
ax.set_xlabel("$J/k_BT$"); ax.set_ylabel("$\\kappa/\\kappa_0$")
ax.set_xlim(0, 0.5); ax.set_ylim(0.9, 1.9)
ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
fig.text(0.70, 0.985, "(c)", fontsize=9, fontweight="bold", va="top")
save(fig, "fig2_highT")
