"""Isolated skyrmion: relaxation, shape data (radius, Q, script-D) and current-driven motion (V3, V4)."""
import json, sys, time
import numpy as np

from .model import Integrator, skyrmion_state, charge_map


def relax_skyrmion(L=64, D=0.3, B=0.12, K=0.0, J=1.0, r0=3.0, t_relax=800.0, dt=0.02):
    it = Integrator(skyrmion_state(1, L, L, radius=r0), J=J, D=D, B=B, K=K, alpha=1.0, beta=0.0, T=0.0, dt=dt)
    it.run(int(t_relax / dt))
    return it.S.copy()


def skyrmion_props(S):
    """Q (one diagonal and averaged), radius (S_z = 0 contour, area-equivalent), centre, script-D."""
    S1 = S[:1]
    q = np.empty(S1.shape[:3])
    charge_map(S1, q, False)
    Q1 = q.sum()
    charge_map(S1, q, True)
    Qav = q.sum()
    Sz = S1[0, ..., 2]
    # area of Sz<0, interpolated smoothly: count fraction (1 - Sz)/2 weighted is not radius; use linear interp
    area = np.clip((-Sz + 1) / 2, 0, 1)  # not used for R
    # radius from the Sz=0 crossing along the radial profile around centre
    xc, yc = centre(q[0])
    L = Sz.shape[0]
    yy, xx = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    dx = (xx - xc + L / 2) % L - L / 2
    dy = (yy - yc + L / 2) % L - L / 2
    rho = np.hypot(dx, dy).ravel()
    order = np.argsort(rho)
    rs, sz = rho[order], Sz.ravel()[order]
    # bin and find first zero crossing
    bins = np.arange(0, L / 2, 0.5)
    idx = np.digitize(rs, bins)
    prof = np.array([sz[idx == k].mean() if np.any(idx == k) else np.nan for k in range(1, len(bins))])
    rb = 0.5 * (bins[1:] + bins[:-1])
    good = ~np.isnan(prof)
    rb, prof = rb[good], prof[good]
    k = np.argmax(prof > 0)
    R0 = rb[k - 1] + (0 - prof[k - 1]) * (rb[k] - rb[k - 1]) / (prof[k] - prof[k - 1])
    # script-D = sum_i (d_x S_i)^2, central difference
    dxS = 0.5 * (np.roll(S1[0], -1, axis=1) - np.roll(S1[0], 1, axis=1))
    dyS = 0.5 * (np.roll(S1[0], -1, axis=0) - np.roll(S1[0], 1, axis=0))
    Dxx = float((dxS ** 2).sum())
    Dyy = float((dyS ** 2).sum())
    return dict(Q_onediag=float(Q1), Q_avg=float(Qav), radius=float(R0), centre=[float(xc), float(yc)],
                scriptD_xx=Dxx, scriptD_yy=Dyy, sz_min=float(Sz.min()))


def centre(qmap, guess=None, win=12):
    """Charge-weighted centroid with periodic wrapping (2D q map, rows = y)."""
    L = qmap.shape[0]
    if guess is None:
        iy, ix = np.unravel_index(np.argmax(np.abs(qmap)), qmap.shape)
        guess = (ix + 0.5, iy + 0.5)
    gx, gy = guess
    yy, xx = np.meshgrid(np.arange(L) + 0.5, np.arange(L) + 0.5, indexing="ij")  # plaquette centres
    dx = (xx - gx + L / 2) % L - L / 2
    dy = (yy - gy + L / 2) % L - L / 2
    m = (np.abs(dx) <= win) & (np.abs(dy) <= win)
    w = qmap[m]
    cx = gx + (w * dx[m]).sum() / w.sum()
    cy = gy + (w * dy[m]).sum() / w.sum()
    return cx, cy


def drive_run(S0, alpha, beta, u0=0.02, t_run=1600.0, t_skip=300.0, dt=0.02, D=0.3, B=0.12, K=0.0, J=1.0):
    L = S0.shape[1]
    it = Integrator(S0.copy(), J=J, D=D, B=B, K=K, alpha=alpha, beta=beta, T=0.0, dt=dt, drive="uniform", u0=u0)
    q = np.empty(it.S.shape[:3])
    charge_map(it.S, q, True)
    c = centre(q[0])
    ux, uy = c  # unwrapped coordinates
    ts, xs, ys = [0.0], [ux], [uy]
    nsamp = int(round(2.0 / dt))
    while it.t < t_run - 1e-9:
        it.run(nsamp)
        charge_map(it.S, q, True)
        cn = centre(q[0], guess=c)
        ddx = (cn[0] - c[0] + L / 2) % L - L / 2
        ddy = (cn[1] - c[1] + L / 2) % L - L / 2
        ux += ddx; uy += ddy
        c = (cn[0] % L, cn[1] % L)
        ts.append(it.t); xs.append(ux); ys.append(uy)
    ts, xs, ys = map(np.array, (ts, xs, ys))
    m = ts >= t_skip
    vx = np.polyfit(ts[m], xs[m], 1)[0]
    vy = np.polyfit(ts[m], ys[m], 1)[0]
    Qf = q[0].sum()
    return dict(alpha=alpha, beta=beta, u0=u0, vx=float(vx), vy=float(vy), Q_end=float(Qf),
                traj=dict(t=ts.tolist(), x=xs.tolist(), y=ys.tolist()))


def thiele(u, alpha, beta, G, Dsc):
    """v from G z x (v - u) + D (alpha v - beta u) = 0, u along x. Returns (vx, vy)."""
    ueff = (beta - alpha) / alpha * u
    den = alpha ** 2 * Dsc ** 2 + G ** 2
    vx = beta / alpha * u - G ** 2 / den * ueff
    vy = -alpha * Dsc * G / den * ueff  # z x (ueff x) = ueff y
    return vx, vy


def main():
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    out = cfg.pop("out", "results/validation/V34.json")
    t0 = time.time()
    model = {k: cfg[k] for k in ("D", "B", "K") if k in cfg}
    S0 = relax_skyrmion(**model)
    props = skyrmion_props(S0)
    res = dict(model=model, props=props, runs=[])
    for a, b in [(0.1, 0.1), (0.1, 0.0), (0.1, 0.2)]:
        res["runs"].append(drive_run(S0, a, b, **model))
    u = 0.02
    v3 = res["runs"][0]
    res["V3"] = dict(vx=v3["vx"], vy=v3["vy"], rel_err_vx=(v3["vx"] - u) / u,
                     hall_angle_deg=float(np.degrees(np.arctan2(v3["vy"], v3["vx"]))),
                     pass_1pct=bool(abs(v3["vx"] - u) / u < 0.01 and abs(v3["vy"]) / u < 0.01))
    Dsc = props["scriptD_xx"]
    # sign of G from the beta = 0 Hall direction: vy(beta=0) = -alpha D G/den * ueff, ueff = -u
    vy0 = res["runs"][1]["vy"]
    G = 4 * np.pi * abs(props["Q_avg"]) * np.sign(vy0) * 1.0
    res["G_used"] = float(G)
    res["G_from_Q"] = float(4 * np.pi * props["Q_avg"])
    res["V4"] = []
    for r in res["runs"][1:]:
        px, py = thiele(u, r["alpha"], r["beta"], G, Dsc)
        res["V4"].append(dict(beta=r["beta"], vx=r["vx"], vy=r["vy"], vx_thiele=px, vy_thiele=py,
                              rel_err_vx=(r["vx"] - px) / px, rel_err_vy=(r["vy"] - py) / py,
                              rel_err_vec=float(np.hypot(r["vx"] - px, r["vy"] - py) / np.hypot(px, py)),
                              pass_5pct=bool(abs(r["vx"] - px) / abs(px) < 0.05 and abs(r["vy"] - py) / abs(py) < 0.05)))
    res["wall_s"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))


def v3_scan(out="results/validation/V3_scan.json"):
    """Diagnose the V3 deficit: u dependence (pinning?) and skyrmion-size dependence (discretisation?)."""
    res = []
    for D, B, us in [(0.3, 0.12, [0.01, 0.02, 0.04]), (0.2, 0.12 * (0.2 / 0.3) ** 2, [0.02]),
                     (0.15, 0.12 * 0.25, [0.02]), (0.1, 0.12 / 9, [0.02])]:
        L = 64 if D >= 0.15 else 96
        S0 = relax_skyrmion(L=L, D=D, B=B, r0=0.85 / D)
        pr = skyrmion_props(S0)
        for u in us:
            r = drive_run(S0, 0.1, 0.1, u0=u, D=D, B=B, t_run=1200.0, t_skip=200.0)
            r.pop("traj")
            r.update(D=D, B=B, L=L, radius=pr["radius"], Q=pr["Q_avg"], scriptD=pr["scriptD_xx"],
                     v_over_u=r["vx"] / u)
            res.append(r)
            print(json.dumps(r), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "v3scan":
        v3_scan()
    else:
        main()
