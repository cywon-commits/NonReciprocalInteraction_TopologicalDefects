"""Item 4: high-T coefficient of <Q> ~ n_z D^2 (SOC case, no magnetic field).

E = -J sum S_i.S_j - sum D_ij.(S_i x S_j),  interfacial D_ij = D (zhat x e_ij)
FL nonreciprocal delta D along zhat:  hN_i = dD * zhat x sum_j S_j
   (N_ij = dD [zhat]_x = N_ji = -N_ij^T : nonreciprocal, exchange-even)
Dynamics as hiT_series.py.  <chi_t> = -beta[<chi E1>_0 - beta<chi E1 E>_0 + <chi E1^(2)>_0]
with E1^(1) = (lam T Lap)^-1 Edot (inverse Laplacian by per-spin harmonic decomposition)
and <chi E1^(2)>_0 = (1/(6 lam T)) <chi L1 E1^(1)>_0.
Q density per site: q_o = (1/8pi) * sum of the 4 corner chiralities at o.
"""
import numpy as np, json
from collections import defaultdict
from functools import lru_cache
from math import factorial
import hiT_series as H
from hiT_series import padd, pmul, pscale, pdiff, moment, var, dot, cross

# ---------- single-spin harmonic decomposition (homogeneous monomials) ----------
def lap3(p):            # R^3 Laplacian of a 3-variable polynomial dict over vars 0,1,2
    r = {}
    for c in range(3):
        r = padd(r, pdiff(pdiff(p, c), c))
    return r
def r2(p):              # multiply by x^2+y^2+z^2
    r = {}
    for c in range(3):
        r = padd(r, pmul(pmul(p, var(c)), var(c)))
    return r
def dfact(n): return H.dfact2(n)

@lru_cache(maxsize=None)
def harm_decomp(expo):
    """expo = (a,b,c) monomial x^a y^b z^c on the unit sphere ->
       list of (l, poly over vars 0..2) with each poly harmonic of degree l."""
    n = sum(expo)
    key = tuple((v, p) for v, p in enumerate(expo) if p)
    p = {key: 1.0}
    out = []
    while n >= 0 and p:
        # harmonic projection of homogeneous p of degree n
        h = {}; Lk = p; k = 0
        while Lk:
            coef = (-1) ** k * dfact(2 * n - 2 * k - 1) / (dfact(2 * n - 1) * (2 ** k) * factorial(k))
            term = Lk
            for _ in range(k): term = r2(term)
            h = padd(h, pscale(term, coef))
            Lk = lap3(Lk); k += 1
        if h: out.append((n, h))
        rest = padd(p, h, -1)          # = r^2 q
        # divide rest by r^2 on the sphere: q such that r^2 q = rest; get q by projection trick:
        # on the sphere rest = q, and q is homogeneous degree n-2 -> recover q by solving
        q = divide_r2(rest, n)
        p = q; n -= 2
    return out

def divide_r2(rest, n):
    """rest is homogeneous degree n and divisible by r^2; return q (degree n-2)."""
    if not rest: return {}
    # polynomial long division by x^2+y^2+z^2 using leading power of x
    rem = dict(rest); q = {}
    while rem:
        # pick monomial with highest x power
        k = max(rem, key=lambda m: dict(m).get(0, 0) * 100 + dict(m).get(1, 0))
        c = rem[k]; d = dict(k)
        if d.get(0, 0) >= 2:
            d[0] -= 2
            if d[0] == 0: del d[0]
        elif d.get(1, 0) >= 2:
            d[1] -= 2
            if d[1] == 0: del d[1]
        elif d.get(2, 0) >= 2:
            d[2] -= 2
            if d[2] == 0: del d[2]
        else:
            raise ValueError("not divisible")
        m = tuple(sorted(d.items()))
        q[m] = q.get(m, 0) + c
        rem = padd(rem, pmul({m: c}, r2(const1())), -1)
        rem = {k2: v for k2, v in rem.items() if abs(v) > 1e-13}
    return q
def const1(): return {(): 1.0}

def inv_lap(f):
    """(Lap on (S^2)^N)^-1 f, constant part dropped."""
    out = defaultdict(float)
    for mono, c in f.items():
        per = defaultdict(lambda: [0, 0, 0])
        for v, p in mono: per[v // 3][v % 3] = p
        spins = list(per)
        pieces = [harm_decomp(tuple(per[s])) for s in spins]
        # product over spins of sums -> combos
        combos = [((), 0, {(): c})]
        for s, pc in zip(spins, pieces):
            new = []
            for (ls, lsum, poly) in combos:
                for (l, h) in pc:
                    # relabel vars 0..2 -> 3s..3s+2
                    hr = {tuple(sorted((3 * s + v, p) for v, p in m)): cc for m, cc in h.items()}
                    new.append((ls + (l,), lsum + l * (l + 1), pmul(poly, hr)))
            combos = new
        for ls, lsum, poly in combos:
            if lsum == 0: continue
            for m, cc in poly.items(): out[m] += -cc / lsum
    return {k: v for k, v in out.items() if abs(v) > 1e-14}

def _sig(m):
    return frozenset(v for v, p in m if p % 2)
def moment_prod(A, B, N):
    """<A*B> over uniform spheres without forming the product."""
    groups = defaultdict(list)
    for m, c in B.items(): groups[_sig(m)].append((m, c))
    tot = 0.0
    for m, c in A.items():
        for m2, c2 in groups.get(_sig(m), ()):
            tot += c * c2 * moment({H.mmul(m, m2): 1.0}, N)
    return tot

# ---------- model ----------
def build(J=1.0, D=1.0, dD=1.0, Rc=2):
    sites, idx = H.sites, H.idx
    S = H.S; Sv = H.Svec
    zx = lambda w: [pscale(w[1], -1.0), w[0], {}]               # zhat x w
    def dmi_vec(e):                                              # D (zhat x e)
        return np.array([-e[1], e[0], 0.0]) * D
    E = {}
    for s, t in H.bonds:
        e = (t[0] - s[0], t[1] - s[1])
        Dv = dmi_vec(e)
        E = padd(E, dot(Sv(s), Sv(t)), -J)
        cr = cross(Sv(s), Sv(t))
        for c in range(3):
            if Dv[c]: E = padd(E, cr[c], -Dv[c])
    near = lambda s: abs(s[0]) + abs(s[1]) <= Rc
    Edot = {}
    for s in sites:
        if not near(s): continue
        tot = [{}, {}, {}]
        for t in H.nbrs(s):
            if t in idx:
                for c in range(3): tot[c] = padd(tot[c], Sv(t)[c])
        hN = [pscale(x, dD) for x in zx(tot)]
        h = H.field_h(E, s)
        Edot = padd(Edot, dot(h, cross(Sv(s), hN)))
    return E, Edot

def run(J=1.0, D=1.0, dD=1.0):
    E, Edot = build(J, D, dD)
    E1u = inv_lap(Edot)                       # E1^(1) = E1u / (lam T)
    sites = H.sites; Sv = H.Svec; N = len(sites)
    diss = {}; gyro = {}
    for s in sites:
        gF = H.grad(E1u, s)
        if all(len(g) == 0 for g in gF): continue
        gE = H.grad(E, s); S_ = Sv(s)
        diss = padd(diss, padd(dot(gE, gF), pmul(dot(S_, gE), dot(S_, gF)), -1), -1.0)
        gyro = padd(gyro, dot(cross(S_, H.field_h(E, s)), gF))
    o = (0, 0)
    Qo = {}
    for a, b in [((1, 0), (0, 1)), ((0, 1), (-1, 0)), ((-1, 0), (0, -1)), ((0, -1), (1, 0))]:
        Qo = padd(Qo, H.chi(o, a, b))
    Qo = pscale(Qo, 1 / (8 * np.pi))
    first = moment_prod(Qo, E1u, N)          # <Q E1^(1)> * lam T
    stat = moment_prod(pmul(Qo, E1u), E, N)
    dis = moment_prod(Qo, diss, N)
    gyr = moment_prod(Qo, gyro, N)
    # <q> = -1/(lam T^2) first + 1/(lam T^3) stat - 1/(6 lam T^3) dis - 1/(6 lam^2 T^3) gyr
    return {"first_coef_1/(lamT^2)": -first, "T3_static": stat, "T3_diss": -dis / 6,
            "T3_gyro_1/(lam^2T^3)": -gyr / 6}

if __name__ == "__main__":
    out = {}
    for (J, D, dD) in [(1, 1, 1), (1, 0.5, 1), (1, 1, -1), (1, -1, 1), (0, 1, 1)]:
        r = run(J, D, dD); out[f"J{J}_D{D}_dD{dD}"] = r
        print(J, D, dD, json.dumps({k: float(f"{v:.6g}") for k, v in r.items()}), flush=True)
    json.dump(out, open("soc_kernel.json", "w"))
