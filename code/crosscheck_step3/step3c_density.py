"""Step 3c: free skyrmion density vs drive.

Pairs (+,-) are created at rate Gamma per area at separation R0 (random angle),
each particle diffuses with D1 = D_r/2 and drifts +-(v/2) x_hat (relative drift v),
a +/- pair closer than a annihilates.  Periodic box.  l_u = D_r / v.
Leading-log prediction (doc 5.3):  n^2 = Gamma ln(R0/a) / (2 pi D_r), independent of v
for l_u >> R0;  drift-dominated limit (l_u << a):  n^2 ~ Gamma / (2 a v).
"""
import numpy as np, sys, json

def run(v, Lbox=160.0, a=1.0, R0=3.0, Dr=1.0, n_target=1/256, dt=0.01,
        tburn=600.0, tmeas=1500.0, seed=0):
    rng = np.random.default_rng(seed)
    Gamma = n_target ** 2 * 2 * np.pi * Dr / np.log(R0 / a)
    rate = Gamma * Lbox ** 2 * dt
    D1 = Dr / 2
    P = np.zeros((0, 2)); M = np.zeros((0, 2))
    nb = int(tburn / dt); nm = int(tmeas / dt)
    acc = 0.0; cnt = 0
    for n in range(nb + nm):
        k = rng.poisson(rate)
        if k:
            c = rng.uniform(0, Lbox, (k, 2)); th = rng.uniform(0, 2 * np.pi, k)
            P = np.vstack([P, c]); M = np.vstack([M, (c + R0 * np.stack([np.cos(th), np.sin(th)], 1)) % Lbox])
        s = np.sqrt(2 * D1 * dt)
        P = (P + rng.normal(size=P.shape) * s + np.array([v / 2 * dt, 0])) % Lbox
        M = (M + rng.normal(size=M.shape) * s - np.array([v / 2 * dt, 0])) % Lbox
        if len(P) and len(M):
            d = P[:, None, :] - M[None, :, :]
            d -= Lbox * np.round(d / Lbox)
            r2 = (d ** 2).sum(-1)
            ip, im = np.where(r2 < a * a)
            if len(ip):
                order = np.argsort(r2[ip, im])
                usedp, usedm = set(), set()
                kp, km = [], []
                for o in order:
                    i, j = ip[o], im[o]
                    if i not in usedp and j not in usedm:
                        usedp.add(i); usedm.add(j); kp.append(i); km.append(j)
                P = np.delete(P, kp, 0); M = np.delete(M, km, 0)
        if n >= nb and n % 10 == 0:
            acc += len(P); cnt += 1
    n_meas = acc / cnt / Lbox ** 2
    return n_meas

if __name__ == "__main__":
    v = float(sys.argv[1])
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    vals = [run(v, seed=s) for s in range(seeds)]
    lu = (1.0 / v) if v > 0 else float("inf")
    print(json.dumps({"v": v, "l_u": lu, "n": float(np.mean(vals)), "n_runs": [float(x) for x in vals],
                      "n_pred_leadlog": 1 / 256}), flush=True)
