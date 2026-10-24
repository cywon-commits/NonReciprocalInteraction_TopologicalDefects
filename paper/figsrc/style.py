import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.frameon": False, "savefig.dpi": 300,
    "lines.linewidth": 1.1, "lines.markersize": 4, "errorbar.capsize": 1.5,
})
C = dict(blue="#2b6cb0", red="#c0392b", green="#2e8b57", orange="#e08e0b", purple="#6b46c1",
         gray="#666666", teal="#1f8a8a", black="#222222")
W1, W2 = 3.4, 7.0
def label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
def save(fig, name):
    fig.savefig(f"/home/claude/paper/figs/{name.split("_",1)[1]}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"/home/claude/paper/figs/{name.split("_",1)[1]}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
