"""Equilibrium reference: <q> ~ B_z D^2 at high T, same lattice and conventions.
<q>_eq = -(beta^3/2) <Q E_B E_D^2>_0 + (lower connected pieces vanish by parity)"""
import numpy as np
import hiT_series as H, soc_series as S
from hiT_series import padd, pmul, pscale, dot, cross
def run(D=1.0, B=1.0):
    Sv = H.Svec; N = len(H.sites)
    ED = {}
    for s, t in H.bonds:
        e = (t[0]-s[0], t[1]-s[1]); Dv = np.array([-e[1], e[0], 0.0]) * D
        cr = cross(Sv(s), Sv(t))
        for c in range(3):
            if Dv[c]: ED = padd(ED, cr[c], -Dv[c])
    EB = {}
    for s in H.sites: EB = padd(EB, Sv(s)[2], -B)
    o = (0, 0); Qo = {}
    for a, b in [((1,0),(0,1)),((0,1),(-1,0)),((-1,0),(0,-1)),((0,-1),(1,0))]:
        Qo = padd(Qo, H.chi(o, a, b))
    Qo = pscale(Qo, 1/(8*np.pi))
    QB = pmul(Qo, EB)
    # <Q EB ED ED> = sum over ED terms of <(Q EB ED) * ED>
    val = S.moment_prod(pmul(QB, ED), ED, N)
    return -0.5 * val          # coefficient of 1/T^3
for D, B in [(1,1),(0.5,1),(1,-1)]:
    v = run(D, B); print(D, B, v, v*8*np.pi*729)
