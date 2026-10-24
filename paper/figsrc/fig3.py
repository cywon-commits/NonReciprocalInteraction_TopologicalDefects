from style import *
import numpy as np, json
U = [json.loads(l) for l in open("/home/claude/pkg/results/stageA_llg/uniform_direction.jsonl")]
S = [json.loads(l) for l in open("/home/claude/pkg/results/stageA_llg/open_strip.jsonl")]
fig, axs = plt.subplots(1, 2, figsize=(W1*1.95, 2.1), gridspec_kw=dict(wspace=0.42))
ax = axs[0]
lab = {0.0: "$0$", 0.785: "$\\pi/4$", 0.393: "$\\pi/8$", -0.393: "$-\\pi/8$"}
U = sorted(U, key=lambda d: d["theta"])
th = [d["theta"] for d in U]
ax.errorbar(th, [d["q_per_site"]*1e5 for d in U], [d["q_err"]*1e5 for d in U], fmt="o", color=C["blue"], label="LL, $\\delta J=J$, $k_BT=1.5J$")
ax.axhline(0, color=C["gray"], lw=0.8)
ax.set_xticks(th); ax.set_xticklabels([lab[round(t, 3)] for t in th])
ax.set_xlabel("bias direction $\\theta$"); ax.set_ylabel("$\\langle q\\rangle$ per site ($10^{-5}$)")
ax.set_ylim(-6, 8); ax.set_xlim(-0.6, 1.0)
for t in (0, np.pi/4): ax.axvline(t, color=C["red"], ls=":", lw=0.7)
ax.text(0.39, 5.6, "mirror present at\n$\\theta=0,\\pi/4$: exact zero", ha="center", fontsize=7, color=C["red"])
ax.legend(loc="lower center", handlelength=1.2)
ax.text(-0.2, 1.02, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
ax = axs[1]
lin = [d for d in S if d["mode"] == "linear"]; uni = [d for d in S if d["mode"] == "uniform"][0]
om = np.array([-d["slope"] for d in lin]); Q = np.array([d["Q_mean"] for d in lin]); Qe = [d["Q_err"] for d in lin]
w = np.linspace(-0.05, 0.05, 10)
k2 = lin[0]["Q_pred_2nd"]/(-lin[0]["slope"]); k1 = lin[0]["Q_pred_lead"]/(-lin[0]["slope"])
ax.plot(w, k2*w, color=C["black"], label="theory, $\\kappa(T)$ to 2nd order")
ax.plot(w, k1*w, color=C["gray"], ls="--", label="leading order")
ax.errorbar(om, Q, Qe, fmt="o", color=C["blue"], label="LL, linear shear")
ax.errorbar([0], [uni["Q_mean"]], [uni["Q_err"]], fmt="s", mfc="white", color=C["blue"], label="LL, uniform bias")
ax.axhline(0, color="#bbbbbb", lw=0.5); ax.axvline(0, color="#bbbbbb", lw=0.5)
ax.set_xlabel("vorticity $\\omega$ of the bias"); ax.set_ylabel("total charge $\\langle Q\\rangle$")
ax.set_xlim(-0.05, 0.05); ax.set_ylim(-0.085, 0.085)
ax.legend(loc="upper left", handlelength=1.4, fontsize=6.5)
ax.text(-0.22, 1.02, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
save(fig, "fig3_selection")
