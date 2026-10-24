from style import *
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
fig = plt.figure(figsize=(W2, 2.45))
# (a) symmetry logic as boxes
ax = fig.add_axes([0.0, 0.0, 0.34, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10.9); ax.axis("off")
def box(x, y, w, h, txt, fc="#eef3fa", ec=C["blue"], fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.25", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs, linespacing=1.25)
def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=C["gray"]))
box(1.3, 7.6, 8.3, 1.9, "improper spin rotation $R$ (det $R=-1$)\n$\\mathbf{S}\\times\\mathbf{h}\\to-R(\\mathbf{S}\\times\\mathbf{h})$")
box(0.4, 4.7, 4.3, 2.1, "precession\nflips sign", fc="#fbeeee", ec=C["red"])
box(5.3, 4.7, 4.3, 2.1, "damping\nunchanged", fc="#eef7f0", ec=C["green"])
arr(2.55, 7.45, 2.55, 6.95); arr(7.45, 7.45, 7.45, 6.95)
box(0.4, 1.2, 9.2, 2.6, "out of equilibrium, precession\nbreaks time reversal like a field\n$\\Rightarrow\\ \\langle Q\\rangle\\neq0$ iff no orientation-\nreversing operation survives", fc="#fff6e6", ec=C["orange"])
arr(2.55, 4.55, 2.55, 3.95); arr(7.45, 4.55, 7.45, 3.95)
ax.text(0.0, 10.85, "(a)", fontsize=9, fontweight="bold", ha="left", va="top")
# lattice helper
def lattice(ax, n=5):
    for i in range(n):
        for j in range(n):
            ax.plot(i, j, "o", ms=1.8, color="#999999", zorder=1)
    ax.set_xlim(-0.8, n - 0.2); ax.set_ylim(-0.8, n - 0.2); ax.set_aspect("equal"); ax.axis("off")
# (b) uniform flow with mirror
ax = fig.add_axes([0.355, 0.17, 0.205, 0.66]); lattice(ax)
for j in range(5):
    ax.annotate("", xy=(3.9, j), xytext=(0.1, j), arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=0.8, mutation_scale=7))
ax.plot([-0.6, 4.6], [2, 2], ls="--", color=C["red"], lw=0.9)
ax.text(4.65, 2.0, "$M$", color=C["red"], fontsize=8, va="center", ha="left")
ax.text(2.0, -1.35, "uniform flow: mirror $M\\parallel\\hat{\\mathbf{v}}$\n$\\langle Q\\rangle=0$ to all orders", ha="center", va="top", fontsize=7.2)
fig.text(0.36, 0.985, "(b)", fontsize=9, fontweight="bold", va="top")
# (c) vortical flow
ax = fig.add_axes([0.585, 0.17, 0.205, 0.66]); lattice(ax)
th = np.linspace(0, 2*np.pi, 9)[:-1]
for t in th:
    r = 1.45; x, y = 2 + r*np.cos(t), 2 + r*np.sin(t); dx, dy = -np.sin(t)*0.55, np.cos(t)*0.55
    ax.annotate("", xy=(x+dx, y+dy), xytext=(x-dx, y-dy), arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=0.8, mutation_scale=7))
ax.add_patch(Circle((2, 2), 0.42, fc="#fbeeee", ec=C["red"], lw=0.8, zorder=3))
ax.text(2, 2, "$Q$", ha="center", va="center", fontsize=8, color=C["red"], zorder=4)
ax.text(2.0, -1.35, "vortical flow: no mirror\n$\\langle Q\\rangle\\propto\\oint\\hat{\\mathbf{v}}\\cdot d\\mathbf{l}$", ha="center", va="top", fontsize=7.2)
fig.text(0.59, 0.985, "(c)", fontsize=9, fontweight="bold", va="top")
# (d) delta D_z with DMI
ax = fig.add_axes([0.815, 0.17, 0.185, 0.66]); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.6); ax.set_aspect("equal"); ax.axis("off")
for k in range(8):
    t = 2*np.pi*k/8; x, y = np.cos(t), np.sin(t)
    ax.annotate("", xy=(x + 0.28*np.cos(t), y + 0.28*np.sin(t)), xytext=(x - 0.28*np.cos(t), y - 0.28*np.sin(t)),
                arrowprops=dict(arrowstyle="-|>", color=C["purple"], lw=0.8, mutation_scale=6))
ax.plot(0, 0, marker="o", ms=9, mfc="white", mec=C["red"], mew=0.9)
ax.plot(0, 0, marker=".", ms=3, color=C["red"])
ax.text(0.0, 1.38, "$\\delta D_z\\hat z$ + interfacial DMI", ha="center", va="bottom", fontsize=7.2)
ax.text(0.0, -1.48, "acts as a field\n$B_{\\rm eff}=-\\frac{11}{72}\\,\\delta D_z/\\alpha$", ha="center", va="top", fontsize=7.2)
fig.text(0.815, 0.985, "(d)", fontsize=9, fontweight="bold", va="top")
save(fig, "fig1_concept")
