from style import *
import numpy as np, json
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
runs = {json.loads(l)["run"]: json.loads(l) for l in open("/home/claude/pkg/results/rotation/rot_llg_runs.jsonl")}
fig = plt.figure(figsize=(W2, 2.55))
ax = fig.add_axes([0.0, 0.0, 0.58, 1.0]); ax.set_xlim(0, 20); ax.set_ylim(0, 9.4); ax.axis("off")
def shear(x0, y0):
    for k, y in enumerate([-1, -0.5, 0.5, 1]):
        L = 0.9*y
        ax.add_patch(FancyArrowPatch((x0 - L, y0 + y), (x0 + L, y0 + y), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=C["blue"]))
def strain(x0, y0):
    for sx, sy in ((1, 1), (-1, -1)):
        ax.add_patch(FancyArrowPatch((x0 + 0.25*sx, y0 + 0.25*sy), (x0 + 1.0*sx, y0 + 1.0*sy), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=C["green"]))
    for sx, sy in ((1, -1), (-1, 1)):
        ax.add_patch(FancyArrowPatch((x0 + 1.0*sx, y0 + 1.0*sy), (x0 + 0.25*sx, y0 + 0.25*sy), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=C["green"]))
def rot(x0, y0, col=C["red"], r=0.95):
    ax.add_patch(Arc((x0, y0), 2*r, 2*r, theta1=20, theta2=320, lw=0.9, color=col))
    t = np.deg2rad(320); ax.add_patch(FancyArrowPatch((x0 + r*np.cos(t-0.15), y0 + r*np.sin(t-0.15)), (x0 + r*np.cos(t+0.05), y0 + r*np.sin(t+0.05)), arrowstyle="-|>", mutation_scale=7, lw=0.9, color=col))
y1 = 7.0
shear(2.0, y1); ax.text(2.0, y1 - 1.55, "shear $u_x(y)$", ha="center", fontsize=7)
ax.text(4.1, y1, "$=$", ha="center", va="center", fontsize=10)
strain(6.2, y1); ax.text(6.2, y1 - 1.55, "strain", ha="center", fontsize=7)
ax.text(8.3, y1, "$+$", ha="center", va="center", fontsize=10)
rot(10.4, y1); ax.text(10.4, y1 - 1.55, "rotation $\\Omega=\\omega/2$", ha="center", fontsize=7)
ax.text(14.9, y1 + 0.35, "strain: traceless tensor,\nno linear coupling to a\nscalar rate ($C_{4v}$)", ha="center", va="center", fontsize=6.8, color=C["green"])
def box(x, y, w, h, txt, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.25", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7, linespacing=1.25)
box(0.4, 1.0, 5.6, 2.6, "rotation of space\n$(\\mathbf{u}\\cdot\\nabla)\\mathbf{S}=\\Omega\\,\\partial_\\varphi\\mathbf{S}$", "#fbeeee", C["red"])
box(7.1, 1.0, 5.6, 2.6, "joint rotation of\nspace + spin $-\\Omega\\mathcal{G}$\n(symmetry: removed)", "#f1f1f1", C["gray"])
box(13.8, 1.0, 5.8, 2.6, "spin-only rotation\n$\\Rightarrow$ precession-only\nfield $-\\Omega\\hat z$", "#fff6e6", C["orange"])
ax.text(6.55, 2.3, "$=$", ha="center", va="center", fontsize=10); ax.text(13.25, 2.3, "$+$", ha="center", va="center", fontsize=10)
ax.add_patch(FancyArrowPatch((10.4, y1 - 1.9), (3.2, 3.75), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=C["gray"], connectionstyle="arc3,rad=0.15"))
fig.text(0.005, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")
# (b) crossover
ax = fig.add_axes([0.665, 0.17, 0.32, 0.72])
eq = {-0.25: runs["EQ2"], -0.5: runs["EQ"], -1.0: runs["EQ3"]}
pts = []
for name in ["I", "I2", "H", "D", "A", "C"]:
    r = runs[name]; e = eq[-r["Om"]]
    F = r["q"]/e["q"]; Fe = F*np.hypot(r["q_err"]/r["q"], e["q_err"]/e["q"])
    S = r["q"]/r["pred"]; Se = r["q_err"]/r["pred"]
    pts.append((r["lam"]*r["T"], F, Fe, S, Se))
pts = np.array(pts)
w = 1/pts[:2, 2]**2; Fm = (pts[:2, 1]*w).sum()/w.sum(); Fe_ = w.sum()**-0.5
w2 = 1/pts[:2, 4]**2; Sm = (pts[:2, 3]*w2).sum()/w2.sum(); Se_ = w2.sum()**-0.5
pts = np.vstack([[pts[0, 0], Fm, Fe_, Sm, Se_], pts[2:]])
ax.errorbar(pts[:, 0], pts[:, 1], pts[:, 2], fmt="o", color=C["blue"], label="$\\langle q\\rangle/q_{\\rm eq}(B=-\\Omega)$")
ax.errorbar(pts[:, 0], pts[:, 3], pts[:, 4], fmt="s", color=C["red"], label="$\\langle q\\rangle/[\\Omega D^2/324\\pi\\alpha^2T^3]$")
rb = runs["B"]; ax.errorbar([rb["lam"]*rb["T"]], [rb["q"]/rb["pred"]], [rb["q_err"]/rb["pred"]], fmt="s", mfc="white", color=C["red"], label="same, $k_BT=4J$")
ax.axhline(1, color="k", lw=0.7)
ax.axvline(1, color="#bbbbbb", lw=0.6, ls=":")
ax.set_xscale("log"); ax.set_xlim(0.07, 3.0); ax.set_ylim(0, 1.95)
ax.set_xlabel("$\\alpha k_BT/J$"); ax.set_ylabel("ratio")
ax.text(0.085, 0.16, "weak damping:\nacts as a Zeeman field", fontsize=6.5, color=C["blue"], va="bottom")
ax.text(2.8, 0.62, "strong\ndamping", fontsize=6.5, color=C["red"], ha="right", va="bottom")
ax.legend(loc="upper left", fontsize=6.2, handlelength=1.2, borderaxespad=0.2)
fig.text(0.6, 0.985, "(b)", fontsize=9, fontweight="bold", va="top")
save(fig, "fig8_rotation")
