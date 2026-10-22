"""Item 3: exact second-order high-T response of corner chiralities to FL delta J.

Dynamics (same as hiT_check.py, gamma = 1):
  dS = [-S x (h + hN) - lam S x (S x h)] dt + noise(2 lam T),
  E = -J sum_<ij> S_i.S_j,  h = -dE/dS,  hN_i = sum_j dJ_ij S_j,  dJ_ji = -dJ_ij.
Stationary p = p0 (1 + phi), phi = -beta E1, with  Lt E1 = dE_N/dt,
  Lt = lam T Lap - lam gradE.grad + (S x h).grad      (precession reversed)
Expansion:  E1 = (lam T Lap)^-1 Edot - (lam T Lap)^-1 L1 (lam T Lap)^-1 Edot + ...
  <chi_t> = -beta [ <chi E1^(1)>_0 - beta <chi E1^(1) E>_0 + <chi E1^(2)>_0 ]
  <chi E1^(2)>_0 = (1/(6 lam T)) <chi L1 E1^(1)>_0       (Lap chi = -6 chi)
Polynomials on (S^2)^N are handled exactly with uniform-sphere moments.
"""
import numpy as np, itertools, json, sys
from collections import defaultdict
from math import prod

# ---------- sparse polynomials: dict{ tuple(sorted (var, pow)) : coeff } ----------
def padd(a, b, s=1.0):
    r = defaultdict(float, a)
    for k, v in b.items(): r[k] += s * v
    return {k: v for k, v in r.items() if v != 0}
def mmul(m1, m2):
    d = defaultdict(int)
    for v, p in m1: d[v] += p
    for v, p in m2: d[v] += p
    return tuple(sorted(d.items()))
def pmul(a, b):
    r = defaultdict(float)
    for k1, v1 in a.items():
        for k2, v2 in b.items():
            r[mmul(k1, k2)] += v1 * v2
    return {k: v for k, v in r.items() if v != 0}
def pscale(a, s): return {k: s * v for k, v in a.items()}
def var(v): return {((v, 1),): 1.0}
def const(c): return {(): c}
def pdiff(a, v):
    r = defaultdict(float)
    for k, c in a.items():
        d = dict(k)
        if v in d:
            p = d[v]; c2 = c * p
            if p == 1: del d[v]
            else: d[v] = p - 1
            r[tuple(sorted(d.items()))] += c2
    return dict(r)
def dfact2(n):  # (n)!! for odd n >= -1
    return prod(range(n, 0, -2)) if n > 0 else 1
def moment(a, nsites):
    tot = 0.0
    for k, c in a.items():
        pw = defaultdict(lambda: [0, 0, 0])
        for v, p in k: pw[v // 3][v % 3] = p
        val = c
        for s, (p1, p2, p3) in pw.items():
            if p1 % 2 or p2 % 2 or p3 % 2: val = 0; break
            val *= dfact2(p1 - 1) * dfact2(p2 - 1) * dfact2(p3 - 1) / dfact2(p1 + p2 + p3 + 1)
        tot += val
    return tot

# ---------- lattice cluster ----------
R = 3
sites = [(x, y) for x in range(-R, R + 2) for y in range(-R, R + 2)]
idx = {s: i for i, s in enumerate(sites)}
def S(site, c): return var(3 * idx[site] + c)
def Svec(site): return [S(site, c) for c in range(3)]
def dot(u, w):
    r = {}
    for a, b in zip(u, w): r = padd(r, pmul(a, b))
    return r
def cross(u, w):
    return [padd(pmul(u[1], w[2]), pmul(u[2], w[1]), -1),
            padd(pmul(u[2], w[0]), pmul(u[0], w[2]), -1),
            padd(pmul(u[0], w[1]), pmul(u[1], w[0]), -1)]
def chi(a, b, c): return dot(Svec(a), cross(Svec(b), Svec(c)))
nbrs = lambda s: [(s[0] + 1, s[1]), (s[0] - 1, s[1]), (s[0], s[1] + 1), (s[0], s[1] - 1)]
bonds = [(s, t) for s in sites for t in [(s[0] + 1, s[1]), (s[0], s[1] + 1)] if t in idx]

def energy(J=1.0):
    E = {}
    for s, t in bonds: E = padd(E, dot(Svec(s), Svec(t)), -J)
    return E
def field_h(E, s):           # h = -dE/dS
    return [pscale(pdiff(E, 3 * idx[s] + c), -1.0) for c in range(3)]
def grad(f, s):              # raw R^3 gradient
    return [pdiff(f, 3 * idx[s] + c) for c in range(3)]

def run(row, J=1.0):
    """dJ = 1 on every x-bond (x,row)->(x+1,row) inside the cluster; returns the
    three coefficients (leading, static, gyro) of <chi_t> for each triangle at y=0,
    in units of J dJ/(27 lam T^2) * {1, J/T, J/(lam T)}."""
    E = energy(J)
    # nonreciprocal field and E_dot from precession by hN
    dJ = {}
    for x in range(-R, R + 1):
        a, b = (x, row), (x + 1, row)
        if a in idx and b in idx: dJ[(a, b)] = 1.0; dJ[(b, a)] = -1.0
    Edot = {}
    act = set(s for p in dJ for s in p)
    for s in act:
        hN = [{}, {}, {}]
        for t in nbrs(s):
            if (s, t) in dJ:
                for c in range(3): hN[c] = padd(hN[c], pscale(S(t, c), dJ[(s, t)]))
        h = field_h(E, s)
        Edot = padd(Edot, dot(h, cross(Svec(s), hN)))
    # E1^(1) = Edot / (lam T * (-6))  ->  keep as  E1u = -Edot/6  (factor 1/(lam T) outside)
    E1u = pscale(Edot, -1.0 / 6.0)
    # L1 pieces acting on E1u:  diss = -gradE.gradE1 (projected), gyro = (S x h).grad E1
    diss = {}; gyro = {}
    for s in sites:
        gE = grad(E, s); gF = grad(E1u, s)
        if all(len(g) == 0 for g in gF): continue
        Sv = Svec(s)
        proj = padd(dot(gE, gF), pmul(dot(Sv, gE), dot(Sv, gF)), -1)   # P-projected dot
        diss = padd(diss, proj, -1.0)
        gyro = padd(gyro, dot(cross(Sv, field_h(E, s)), gF))
    o = (0, 0)
    tri = {"UR": chi(o, (1, 0), (0, 1)), "UL": chi(o, (0, 1), (-1, 0)),
           "LL": chi(o, (-1, 0), (0, -1)), "LR": chi(o, (0, -1), (1, 0)),
           "chain": chi(o, (1, 0), (-1, 0))}
    out = {}
    N = len(sites)
    for k, ch in tri.items():
        # <chi> = -beta/(lam T) [ <chi E1u> - beta <chi E1u E> + (1/(6 T)) <chi (lam diss + gyro)/lam ... > ]
        lead = moment(pmul(ch, E1u), N)                 # x (-1/(lam T^2))
        stat = moment(pmul(pmul(ch, E1u), E), N)         # x (+1/(lam T^3))
        dis = moment(pmul(ch, diss), N)                  # x (-1/(6 T^3 lam))   (lam cancels)
        gyr = moment(pmul(ch, gyro), N)                  # x (-1/(6 lam^2 T^3))
        u = J / 27.0                                     # unit J dJ/(27 lam T^2), dJ = 1
        out[k] = {"lead": -lead / u, "static": stat / (u * J), "diss": -dis / (6 * u * J),
                  "gyro": -gyr / (6 * u * J)}
    return out

if __name__ == "__main__":
    res = {}
    for row in (-2, -1, 0, 1, 2):
        res[row] = run(row)
        print(row, json.dumps({k: {a: round(b, 5) for a, b in v.items()} for k, v in res[row].items()}), flush=True)
    json.dump({str(k): v for k, v in res.items()}, open("hiT_kernel.json", "w"))
