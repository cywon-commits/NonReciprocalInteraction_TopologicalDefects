"""Driven stochastic LLG with on-the-fly topological event detection and skyrmion tracking.

usage: python -m skynuc.nucleation CONFIG.json [--resume]

Config keys (defaults in DEFAULTS). Outputs in out_dir/run_id: events.csv, summary.json, state.npz.
"""
import csv, json, os, sys, time
import numpy as np

from .model import Integrator, uniform_state, shear_profiles
from .topo import triangle_omega, find_jumps, window_dq, find_skyrmions, excluded_rows

DEFAULTS = dict(
    run_id="test", out_dir="results/runs",
    L=64, R=16, J=1.0, D=0.3, B=0.12, K=0.0, T=0.4, alpha=0.1, beta=0.0,
    drive="shear", u0=0.0, theta=0.0, dmi="neel", Dy_scale=1.0, u3=0.0, rot_sign=1.0, nper=1,
    dt=0.02, dts=0.24, t_eq=200.0, t_run=1000.0, seed=1,
    nbands=16, Q_fav=-1, r_sk=2.56, sz_thr=0.0, rq=5.0, r_match=4.0, r_group=2.5,
    pending_time=5.0, checkpoint_wall=600.0, n_tblocks=8,
    obj_match=3.0, obj_gap=1.0, tau_est=[2.0, 5.0, 10.0, 20.0],
)

KINDS = ["nuc", "col", "anti_nuc", "anti_col", "pair", "multi"]


def _mind(a, b, L):
    d = (a - b + L / 2) % L - L / 2
    return d


class Run:
    def __init__(self, cfg):
        self.c = c = {**DEFAULTS, **cfg}
        self.dir = os.path.join(c["out_dir"], c["run_id"])
        os.makedirs(self.dir, exist_ok=True)
        L, R = c["L"], c["R"]
        self.L, self.R = L, R
        self.ns = int(round(c["dts"] / c["dt"]))
        assert abs(self.ns * c["dt"] - c["dts"]) < 1e-9
        self.it = Integrator(uniform_state(R, L, L), J=c["J"], D=c["D"], B=c["B"], K=c["K"], alpha=c["alpha"],
                             beta=c["beta"], T=c["T"], dt=c["dt"], drive=c["drive"], u0=c["u0"],
                             theta=c["theta"], dmi=c["dmi"], Dy_scale=c["Dy_scale"], seed=c["seed"],
                             u3=c["u3"], rot_sign=c["rot_sign"], nper=c["nper"])
        nb = c["nbands"]
        assert L % nb == 0
        self.bw = L // nb
        self.acc = dict(
            counts={k: np.zeros(nb, np.int64) for k in KINDS},
            flagged={k: np.zeros(nb, np.int64) for k in KINDS},
            area_time=np.zeros(nb), sk_time=np.zeros(nb), anti_time=np.zeros(nb),
            area_rkb=np.zeros((R, c["n_tblocks"], nb)), sk_rkb=np.zeros((R, c["n_tblocks"], nb)),
            est_rkb=np.zeros((len(c["tau_est"]), R, c["n_tblocks"], nb)),
            drift_dx=np.zeros(nb), drift_dy=np.zeros(nb), drift_dt=np.zeros(nb), drift_n=np.zeros(nb, np.int64),
            n_samples=0, t_meas=0.0, jump_total=0, dQ_mismatch=0, n_sk_hist=np.zeros(64, np.int64),
            dqw_hist=np.zeros(80, np.int64), Q_total_hist=np.zeros(41, np.int64), next_id=0,
        )
        self.phase = "eq"
        self.t_start_wall = time.time()

    # ---------- state I/O ----------
    def save(self):
        a = self.acc
        flat = {}
        for k, v in a.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}__{kk}"] = vv
            else:
                flat[k] = np.asarray(v)
        np.savez(os.path.join(self.dir, "state.npz"), S=self.it.S, rng=self.it.rng, t=self.it.t,
                 phase=self.phase, **flat)
        self.write_summary()

    def load(self):
        z = np.load(os.path.join(self.dir, "state.npz"))
        self.it.S[...] = z["S"]; self.it.rng[...] = z["rng"]; self.it.t = float(z["t"])
        self.phase = str(z["phase"])
        for k, v in self.acc.items():
            if isinstance(v, dict):
                for kk in v:
                    v[kk] = z[f"{k}__{kk}"]
            else:
                self.acc[k] = z[k] if np.ndim(z[k]) else z[k].item()

    # ---------- main loop ----------
    def run(self, resume=False):
        c = self.c
        if resume and os.path.exists(os.path.join(self.dir, "state.npz")):
            self.load()
        else:
            open(os.path.join(self.dir, "events.csv"), "w").write(
                "run_id,replica,t,x,y,dQ,kind,flag,dq_w,n_jumps,obj_id,lifetime\n")
            json.dump(c, open(os.path.join(self.dir, "config.json"), "w"), indent=1)
            open(os.path.join(self.dir, "objects.csv"), "w").write(
                "run_id,replica,obj,sign,t_birth,x_birth,y_birth,t_death,x_death,y_death,lifetime,censored,n_frames\n")
        if self.phase == "eq":
            nleft = int(round((c["t_eq"] - self.it.t) / c["dt"]))
            while nleft > 0:
                k = min(nleft, 5000)
                self.it.run(k); nleft -= k
            self.phase = "meas"
            self.save()
        self.measure()

    def measure(self):
        c, it, L, R = self.c, self.it, self.L, self.R
        Om_old = np.empty((R, L, L, 2)); Om_new = np.empty_like(Om_old)
        triangle_omega(it.S, Om_old)
        q_old = Om_old.sum(axis=3) / (4 * np.pi)
        sk_old = self.track(q_old)
        for r in range(R):
            for sk in sk_old[r]:
                sk[4] = self.new_id(); sk[5] = np.nan
        pending = [[] for _ in range(R)]  # recent nucleations not yet seen as clusters: [x, y, sign, t, id]
        fcsv = open(os.path.join(self.dir, "events.csv"), "a")
        w = csv.writer(fcsv)
        fobj = open(os.path.join(self.dir, "objects.csv"), "a")
        self.wobj = csv.writer(fobj)
        self.objs = [dict() for _ in range(R)]
        self.objects_update(sk_old, it.t, initial=True)
        t_end = c["t_eq"] + c["t_run"]
        last_ck = time.time()
        a = self.acc
        Qtot_old = np.rint(q_old.sum(axis=(1, 2))).astype(int)
        while it.t < t_end - 1e-9:
            it.run(self.ns)
            t = it.t
            triangle_omega(it.S, Om_new)
            q_new = Om_new.sum(axis=3) / (4 * np.pi)
            Qtot = np.rint(q_new.sum(axis=(1, 2))).astype(int)
            nj, jumps = find_jumps(Om_old, Om_new, 4096)
            a["jump_total"] += nj
            jumps = jumps[:min(nj, 4096)]
            msum = np.zeros(R, int)
            for j in jumps:
                msum[j[0]] += j[4]
            a["dQ_mismatch"] += int(np.abs(msum - (Qtot - Qtot_old)).sum())
            # exposures for the interval just finished use the skyrmion list at its start
            self.accumulate(sk_old)
            # events
            for r in range(R):
                jr = jumps[jumps[:, 0] == r]
                if len(jr) == 0:
                    continue
                for grp in self.group(jr):
                    self.classify(r, grp, t, sk_old, pending, q_old, q_new, w)
            sk_new = self.track(q_new)
            self.link(sk_old, sk_new, pending, t)
            self.objects_update(sk_new, t)
            Om_old, Om_new = Om_new, Om_old
            q_old = q_new; sk_old = sk_new; Qtot_old = Qtot
            for Qr in Qtot:
                a["Q_total_hist"][min(max(Qr + 20, 0), 40)] += 1
            if time.time() - last_ck > c["checkpoint_wall"]:
                fcsv.flush(); fobj.flush(); self.save(); last_ck = time.time()
        for r in range(R):
            for oid in list(self.objs[r]):
                self.objects_finalize(r, oid, censored=1)
        fcsv.close(); fobj.close()
        self.phase = "done"
        self.save()

    # ---------- object tracker (primary nucleation / collapse definition) ----------
    def objects_update(self, sk, t, initial=False):
        """Match tracked clusters to live objects (same sign, within obj_match, unseen for <= obj_gap).

        Unmatched clusters start new objects (born at t; born_before_meas if initial). Objects unseen for
        longer than obj_gap die at their last-seen time. Established-object time (age >= tau) is
        accumulated per band for each tau in tau_est."""
        c, L, a = self.c, self.L, self.acc
        dts = c["dts"]
        kblk = min(int((t - c["t_eq"]) / c["t_run"] * c["n_tblocks"]), c["n_tblocks"] - 1)
        for r in range(self.R):
            live = self.objs[r]
            pairs = []
            for i, s in enumerate(sk[r]):
                for oid, o in live.items():
                    if o["sign"] != s[2]:
                        continue
                    d = np.hypot(_mind(s[0], o["x"], L), _mind(s[1], o["y"], L))
                    if d < c["obj_match"]:
                        pairs.append((d, i, oid))
            pairs.sort()
            used_c, used_o = set(), set()
            for d, i, oid in pairs:
                if i in used_c or oid in used_o:
                    continue
                used_c.add(i); used_o.add(oid)
                o = live[oid]; s = sk[r][i]
                o["x"], o["y"], o["last"] = s[0], s[1], t
                o["n"] += 1
            for i, s in enumerate(sk[r]):
                if i in used_c:
                    continue
                oid = self.new_id()
                live[oid] = dict(sign=s[2], birth=(np.nan if initial else t), bx=s[0], by=s[1],
                                 x=s[0], y=s[1], last=t, n=1)
            for oid in list(live):
                o = live[oid]
                if t - o["last"] > c["obj_gap"]:
                    self.objects_finalize(r, oid, censored=0)
                elif not initial and o["sign"] == c["Q_fav"] and o["last"] == t:
                    age = t - o["birth"] if not np.isnan(o["birth"]) else np.inf
                    b = self.band(o["y"])
                    for j, tau in enumerate(c["tau_est"]):
                        if age >= tau:
                            a["est_rkb"][j, r, kblk, b] += dts

    def objects_finalize(self, r, oid, censored):
        o = self.objs[r].pop(oid)
        life = o["last"] - o["birth"]
        self.wobj.writerow([self.c["run_id"], r, oid, o["sign"], f"{o['birth']:.3f}", f"{o['bx']:.3f}",
                            f"{o['by']:.3f}", f"{o['last']:.3f}", f"{o['x']:.3f}", f"{o['y']:.3f}",
                            f"{life:.3f}", censored, o["n"]])

    # ---------- helpers ----------
    def new_id(self):
        self.acc["next_id"] += 1
        return int(self.acc["next_id"])

    def track(self, q):
        c = self.c
        out, cnt = find_skyrmions(self.it.S, q, c["sz_thr"], c["rq"], 256)
        sk = []
        for r in range(self.R):
            lst = []
            # largest clusters first; fragments of the same texture (within 3 sites) are dropped
            cand = [k for k in range(cnt[r]) if abs(out[r, k, 2]) > 0.5]  # thermal specks have Qc ~ 0
            for k in sorted(cand, key=lambda k: -out[r, k, 3]):
                cx, cy, Qc, size = out[r, k]
                if any(np.hypot(_mind(cx, o[0], self.L), _mind(cy, o[1], self.L)) < 3.0 for o in lst):
                    continue
                m = int(np.rint(Qc))
                lst.append([cx, cy, m, size, -1, np.nan, Qc])  # id, birth time set by link()
            sk.append([o[:6] for o in lst if abs(o[2]) == 1 and abs(o[6] - o[2]) < 0.4])
        return sk

    def band(self, y):
        return int(np.floor(y)) // self.bw % self.c["nbands"]

    def accumulate(self, sk):
        c, L, a = self.c, self.L, self.acc
        dts, nb, bw = c["dts"], c["nbands"], self.bw
        rex = 2 * c["r_sk"]
        a["n_samples"] += 1
        a["t_meas"] += dts
        area = np.full(nb, bw * L * self.R, float)
        kblk = min(int((self.it.t - c["t_eq"]) / c["t_run"] * c["n_tblocks"]), c["n_tblocks"] - 1)
        a["area_rkb"][:, kblk, :] += bw * L * dts
        ntot = 0
        for r in range(self.R):
            lst = sk[r]
            ntot += len(lst)
            if lst:
                cen = np.array([[s[0], s[1]] for s in lst])
                rows = excluded_rows(cen, len(lst), L, L, rex)
                ex = rows.reshape(nb, bw).sum(axis=1)
                area -= ex
                a["area_rkb"][r, kblk, :] -= ex * dts
                for s in lst:
                    if s[2] == c["Q_fav"]:
                        a["sk_time"][self.band(s[1])] += dts
                        a["sk_rkb"][r, kblk, self.band(s[1])] += dts
                    else:
                        a["anti_time"][self.band(s[1])] += dts
        a["area_time"] += area * dts
        a["n_sk_hist"][min(ntot, 63)] += 1

    def link(self, sk_old, sk_new, pending, t):
        """Carry object identity (id, birth time) from the previous snapshot or from a pending
        nucleation to the new clusters; accumulate the drift of favoured skyrmions."""
        c, L, a = self.c, self.L, self.acc
        for r in range(self.R):
            used = set()
            for s in sk_new[r]:
                best, bd = None, 2.0
                for k, o in enumerate(sk_old[r]):
                    if o[2] != s[2] or k in used:
                        continue
                    dx = _mind(s[0], o[0], L); dy = _mind(s[1], o[1], L)
                    d = np.hypot(dx, dy)
                    if d < bd:
                        bd, best = d, (dx, dy, k)
                if best is not None:
                    dx, dy, k = best
                    used.add(k)
                    o = sk_old[r][k]
                    s[4], s[5] = o[4], o[5]
                    if s[2] == c["Q_fav"]:
                        ym = (o[1] + dy / 2) % L
                        b = self.band(ym)
                        a["drift_dx"][b] += dx; a["drift_dy"][b] += dy
                        a["drift_dt"][b] += c["dts"]; a["drift_n"][b] += 1
                    continue
                # newly visible cluster: inherit from the closest pending nucleation of the same sign
                best, bd = None, c["r_match"]
                for k, p in enumerate(pending[r]):
                    if p[2] != s[2]:
                        continue
                    d = np.hypot(_mind(s[0], p[0], L), _mind(s[1], p[1], L))
                    if d < bd:
                        bd, best = d, k
                if best is not None:
                    p = pending[r].pop(best)
                    s[4], s[5] = p[4], p[3]
                else:
                    s[4], s[5] = self.new_id(), np.nan
            pending[r] = [p for p in pending[r] if t - p[3] <= c["pending_time"]]

    def group(self, jr):
        """Group jumps of one replica within r_group (periodic) by single linkage."""
        L, rg = self.L, self.c["r_group"]
        pos = []
        for j in jr:
            y, x, k = j[1], j[2], j[3]
            pos.append((x + (2 / 3 if k == 0 else 1 / 3), y + (1 / 3 if k == 0 else 2 / 3)))
        n = len(jr)
        lab = list(range(n))

        def f(i):
            while lab[i] != i:
                lab[i] = lab[lab[i]]; i = lab[i]
            return i
        for i in range(n):
            for k in range(i + 1, n):
                if np.hypot(_mind(pos[i][0], pos[k][0], L), _mind(pos[i][1], pos[k][1], L)) <= rg:
                    lab[f(i)] = f(k)
        groups = {}
        for i in range(n):
            groups.setdefault(f(i), []).append(i)
        res = []
        for g in groups.values():
            x0, y0 = pos[g[0]]
            xs = [x0 + _mind(pos[i][0], x0, L) for i in g]
            ys = [y0 + _mind(pos[i][1], y0, L) for i in g]
            m = int(sum(jr[i][4] for i in g))
            res.append(dict(x=float(np.mean(xs)) % L, y=float(np.mean(ys)) % L, m=m, n=len(g),
                            py=int(jr[g[0]][1]), px=int(jr[g[0]][2])))
        return res

    def classify(self, r, g, t, sk_old, pending, q_old, q_new, w):
        c, L, a = self.c, self.L, self.acc
        dqw = window_dq(q_old, q_new, r, g["py"], g["px"], 1)
        a["dqw_hist"][min(max(int(np.floor((dqw + 2) * 20)), 0), 79)] += 1
        m, Qf = g["m"], c["Q_fav"]
        flag = int(g["n"] > 1 or abs(dqw - m) > 0.5)
        obj, life = -1, np.nan
        if m == 0:
            kind = "pair"
        elif abs(m) > 1:
            kind = "multi"
        else:
            # a jump of -sign(object) at an existing object of that sign = collapse
            obj_sign = -m  # the object that would be destroyed has charge -m
            # nearest candidate among tracked objects and pending (not yet visible) nucleations
            hit, bd = None, c["r_match"]
            for idx, s in enumerate(sk_old[r]):
                d = np.hypot(_mind(s[0], g["x"], L), _mind(s[1], g["y"], L))
                if s[2] == obj_sign and d < bd:
                    hit, bd = ("sk", idx), d
            for idx, p in enumerate(pending[r]):
                d = np.hypot(_mind(p[0], g["x"], L), _mind(p[1], g["y"], L))
                if p[2] == obj_sign and d < bd:
                    hit, bd = ("pend", idx), d
            if hit is not None:
                kind = "col" if obj_sign == Qf else "anti_col"
                if hit[0] == "sk":
                    o = sk_old[r].pop(hit[1])
                    obj, life = o[4], t - o[5]
                else:
                    p = pending[r].pop(hit[1])
                    obj, life = p[4], t - p[3]
            else:
                kind = "nuc" if m == Qf else "anti_nuc"
                obj = self.new_id()
                pending[r].append([g["x"], g["y"], m, t, obj])
        b = self.band(g["y"])
        a["counts"][kind][b] += 1
        a["flagged"][kind][b] += flag
        w.writerow([c["run_id"], r, f"{t:.3f}", f"{g['x']:.3f}", f"{g['y']:.3f}", m, kind, flag,
                    f"{dqw:.3f}", g["n"], obj, f"{life:.3f}"])

    def write_summary(self):
        c, a = self.c, self.acc
        nb, L = c["nbands"], self.L
        edges = np.arange(nb + 1) * self.bw
        # band averages of omega(y) and psi(y) over [edge_b, edge_b+1); for "rot" omega is the profile of the
        # u0 shear whose Omega = omega/2 drives the precession term (rot_sign only flips the dynamics)
        if c["drive"] in ("shear", "shear3", "rot"):
            _, omf, psif = shear_profiles(c["drive"], c["u0"], c["u3"], L, c.get("nper", 1))
            yy = edges[:-1, None] + (np.arange(400)[None, :] + 0.5) / 400 * self.bw
            om, ps = omf(yy).mean(axis=1), psif(yy).mean(axis=1)
        else:
            om = np.zeros(nb); ps = np.zeros(nb)
        out = dict(config=c, phase=self.phase, t=self.it.t, wall_s=time.time() - self.t_start_wall,
                   band_edges=edges.tolist(), omega_band=om.tolist(), psi_band=ps.tolist())
        for k, v in a.items():
            if isinstance(v, dict):
                out[k] = {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
            else:
                out[k] = np.asarray(v).tolist()
        json.dump(out, open(os.path.join(self.dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    for kv in sys.argv[2:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            cfg[k] = json.loads(v) if v[:1] in "0123456789-[{tfn\"" else v
    Run(cfg).run(resume="--resume" in sys.argv)
