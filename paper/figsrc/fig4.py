from style import *
import numpy as np, json
Cc = [json.loads(l) for l in open("/home/claude/pkg/results/step3_thiele/cellular_flow.jsonl")]
Ss = [json.loads(l) for l in open("/home/claude/pkg/results/step3_thiele/strip.jsonl")]
fig = plt.figure(figsize=(W2, 2.45))
# (a) map
ax = fig.add_axes([0.03, 0.12, 0.27, 0.78])
x = np.linspace(0, 2*np.pi, 200); X, Y = np.meshgrid(x, x)
psi = 1.5*np.sin(X)*np.sin(Y)
im = ax.imshow(np.exp(-psi), origin="lower", extent=[0, 2, 0, 2], cmap="Blues", aspect="equal")
ax.contour(X/np.pi, Y/np.pi, psi, levels=[-1, -0.5, 0, 0.5, 1], colors="k", linewidths=0.4)
xs = np.linspace(0.125, 1.875, 8)*np.pi; XS, YS = np.meshgrid(xs, xs)
gx = 1.5*np.cos(XS)*np.sin(YS); gy = 1.5*np.sin(XS)*np.cos(YS)
ax.quiver(XS/np.pi, YS/np.pi, -gx, -gy, color=C["red"], scale=22, width=0.006)
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2]); ax.set_xlabel("$kx/\\pi$"); ax.set_ylabel("$ky/\\pi$")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("$P_+\\propto e^{-G\\psi_e/k_BT}$", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.text(0.005, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")
ax.set_title("Hall drift $-\\mu\\nabla(G\\psi_e)$ (arrows)", fontsize=7, pad=3)
# (b) ratios
ax = fig.add_axes([0.45, 0.2, 0.53, 0.66])
rows = []
for d in Cc: rows.append((d["alpha"], d["beta"], d["G"], d["fit_G_over_T"]/d["pred_G_over_T"], C["blue"]))
for d in Ss:
    if "fit_inv_ell" in d and not (d["alpha"] == 0.1): rows.append((d["alpha"], d["beta"], d["G"], d["fit_inv_ell"]/d["pred_inv_ell"], C["green"]))
for i, (a, b, G, v, c) in enumerate(rows):
    ax.plot(i, v, "o", color=c, mfc=(c if G > 0 else "white"), mew=1.0)
ax.axhline(1, color=C["black"], lw=0.8)
ax.axhspan(0.98, 1.02, color="#e4e4e4", zorder=0)
ax.set_xticks(range(len(rows))); ax.set_xticklabels(["(%.1f, %.1f)" % (r[0], r[1]) for r in rows], fontsize=6.3)
ax.set_xlabel("$(\\alpha,\\beta)$")
ax.set_ylim(0.95, 1.04); ax.set_xlim(-0.6, len(rows)-0.4)
ax.set_ylabel("measured / predicted")
ax.axvline(4.5, color="#bbbbbb", lw=0.5)
ax.text(2.0, 1.03, "cellular flow: slope of $\\ln P_Q$ vs $\\psi_e$", ha="center", fontsize=7, color=C["blue"])
ax.text(5.5, 1.03, "strip: $1/\\ell_u$", ha="center", fontsize=7, color=C["green"])
ax.plot([], [], "o", color="k", label="$G=+1$"); ax.plot([], [], "o", color="k", mfc="white", label="$G=-1$")
ax.legend(loc="lower left", ncol=2, handlelength=1.0)
fig.text(0.40, 0.985, "(b)", fontsize=9, fontweight="bold", va="top")
save(fig, "fig4_stream")
