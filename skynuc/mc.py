"""Independent Metropolis Monte Carlo for the same Hamiltonian (check V2)."""
import numpy as np
from numba import njit, prange

from .model import _cross, _splitmix, make_rng


@njit(parallel=True, cache=True)
def metropolis_sweeps(S, nsweep, T, J, Dx, Dy, B, K, cone, rng, acc):
    """Checkerboard Metropolis; proposal = uniform in a cone of half-angle-ish size `cone` around S_i.

    The proposal S' = normalize(S + cone * v), v uniform in the unit ball, is symmetric.
    rng has one stream per (replica, row).
    """
    R, Ly, Lx, _ = S.shape
    beta_t = 1.0 / T
    for _ in range(nsweep):
        for colour in range(2):
            for idx in prange(R * Ly):
                r = idx // Ly
                y = idx - r * Ly
                yp = (y + 1) % Ly
                ym = (y - 1 + Ly) % Ly
                na = 0
                for x in range((y + colour) % 2, Lx, 2):
                    xp = (x + 1) % Lx
                    xm = (x - 1 + Lx) % Lx
                    a0 = S[r, y, xp, 0]; a1 = S[r, y, xp, 1]; a2 = S[r, y, xp, 2]
                    b0 = S[r, y, xm, 0]; b1 = S[r, y, xm, 1]; b2 = S[r, y, xm, 2]
                    c0 = S[r, yp, x, 0]; c1 = S[r, yp, x, 1]; c2 = S[r, yp, x, 2]
                    d0 = S[r, ym, x, 0]; d1 = S[r, ym, x, 1]; d2 = S[r, ym, x, 2]
                    H0 = J * (a0 + b0 + c0 + d0)
                    H1 = J * (a1 + b1 + c1 + d1)
                    H2 = J * (a2 + b2 + c2 + d2) + B
                    e0, e1, e2 = _cross(a0 - b0, a1 - b1, a2 - b2, Dx[0], Dx[1], Dx[2])
                    f0, f1, f2 = _cross(c0 - d0, c1 - d1, c2 - d2, Dy[0], Dy[1], Dy[2])
                    H0 += e0 + f0
                    H1 += e1 + f1
                    H2 += e2 + f2
                    s0 = S[r, y, x, 0]; s1 = S[r, y, x, 1]; s2 = S[r, y, x, 2]
                    while True:
                        v0 = 2.0 * _splitmix(rng, idx) - 1.0
                        v1 = 2.0 * _splitmix(rng, idx) - 1.0
                        v2 = 2.0 * _splitmix(rng, idx) - 1.0
                        if v0 * v0 + v1 * v1 + v2 * v2 <= 1.0:
                            break
                    n0 = s0 + cone * v0
                    n1 = s1 + cone * v1
                    n2 = s2 + cone * v2
                    nn = 1.0 / np.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
                    n0 *= nn; n1 *= nn; n2 *= nn
                    # E_i = -S_i.H_lin - K S_z^2 (local field is linear except the anisotropy)
                    dE = -((n0 - s0) * H0 + (n1 - s1) * H1 + (n2 - s2) * H2) - K * (n2 * n2 - s2 * s2)
                    if dE <= 0.0 or _splitmix(rng, idx) < np.exp(-beta_t * dE):
                        S[r, y, x, 0] = n0; S[r, y, x, 1] = n1; S[r, y, x, 2] = n2
                        na += 1
                acc[idx] += na
