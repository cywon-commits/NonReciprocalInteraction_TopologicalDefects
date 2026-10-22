"""Numba kernels for event detection (triangle solid-angle jumps) and skyrmion tracking."""
import numpy as np
from numba import njit, prange

from .model import _omega


@njit(parallel=True, cache=True)
def triangle_omega(S, Om):
    """Om[r, y, x, k]: solid angle of triangle k of plaquette (y, x); one diagonal (i -> i+x+y).

    k = 0: (i, i+x, i+x+y), k = 1: (i, i+x+y, i+y). q_p = (Om[...,0] + Om[...,1]) / 4 pi.
    """
    R, Ly, Lx, _ = S.shape
    for idx in prange(R * Ly):
        r = idx // Ly
        y = idx - r * Ly
        yp = y + 1
        if yp == Ly:
            yp = 0
        for x in range(Lx):
            xp = x + 1
            if xp == Lx:
                xp = 0
            i0 = S[r, y, x, 0]; i1 = S[r, y, x, 1]; i2 = S[r, y, x, 2]
            a0 = S[r, y, xp, 0]; a1 = S[r, y, xp, 1]; a2 = S[r, y, xp, 2]
            b0 = S[r, yp, xp, 0]; b1 = S[r, yp, xp, 1]; b2 = S[r, yp, xp, 2]
            c0 = S[r, yp, x, 0]; c1 = S[r, yp, x, 1]; c2 = S[r, yp, x, 2]
            Om[r, y, x, 0] = _omega(i0, i1, i2, a0, a1, a2, b0, b1, b2)
            Om[r, y, x, 1] = _omega(i0, i1, i2, b0, b1, b2, c0, c1, c2)


@njit(cache=True)
def find_jumps(Om_old, Om_new, max_out):
    """Triangles whose solid angle jumped by a multiple of 4 pi between two samples.

    Returns (n, out) with out[k] = (r, y, x, tri, m), m = round(dOmega / 4 pi) != 0.
    Also returns, per replica, the sum of the continuous remainder (diagnostic).
    """
    R, Ly, Lx, _ = Om_old.shape
    out = np.zeros((max_out, 5), np.int64)
    n = 0
    inv = 1.0 / (4.0 * np.pi)
    for r in range(R):
        for y in range(Ly):
            for x in range(Lx):
                for k in range(2):
                    d = (Om_new[r, y, x, k] - Om_old[r, y, x, k]) * inv
                    m = int(np.floor(d + 0.5))
                    if m != 0:
                        if n < max_out:
                            out[n, 0] = r; out[n, 1] = y; out[n, 2] = x; out[n, 3] = k; out[n, 4] = m
                        n += 1
    return n, out


@njit(cache=True)
def window_dq(q_old, q_new, r, y, x, h):
    """Sum of (q_new - q_old) over the (2h+1)^2 plaquette window centred on (y, x), periodic."""
    _, Ly, Lx = q_old.shape
    s = 0.0
    for dy in range(-h, h + 1):
        yy = (y + dy) % Ly
        for dx in range(-h, h + 1):
            xx = (x + dx) % Lx
            s += q_new[r, yy, xx] - q_old[r, yy, xx]
    return s


@njit(parallel=True, cache=True)
def find_skyrmions(S, q, sz_thr, rq, max_per):
    """Clusters of sites with S_z < sz_thr (4-connected, periodic), per replica.

    For each cluster: unwrapped site centroid -> refined charge-weighted centre within radius rq,
    charge Q_c = sum of q over plaquettes within rq of the site centroid, size (sites).
    Output out[r, k] = (cx, cy, Qc, size); count[r].
    """
    R, Ly, Lx, _ = S.shape
    out = np.zeros((R, max_per, 4))
    count = np.zeros(R, np.int64)
    for r in prange(R):
        lab = np.full((Ly, Lx), -1, np.int64)
        stack_y = np.empty(Ly * Lx, np.int64)
        stack_x = np.empty(Ly * Lx, np.int64)
        uy_ = np.empty(Ly * Lx)
        ux_ = np.empty(Ly * Lx)
        nc = 0
        for y0 in range(Ly):
            for x0 in range(Lx):
                if lab[y0, x0] >= 0 or S[r, y0, x0, 2] >= sz_thr:
                    continue
                # flood fill with unwrapped coordinates
                top = 0
                stack_y[0] = y0; stack_x[0] = x0
                uy_[0] = y0; ux_[0] = x0
                lab[y0, x0] = nc
                sy = 0.0; sx = 0.0; size = 0
                while top >= 0:
                    yy = stack_y[top]; xx = stack_x[top]
                    fy = uy_[top]; fx = ux_[top]
                    top -= 1
                    sy += fy; sx += fx; size += 1
                    for d in range(4):
                        if d == 0:
                            ny = yy; nx = xx + 1; gy = fy; gx = fx + 1
                        elif d == 1:
                            ny = yy; nx = xx - 1; gy = fy; gx = fx - 1
                        elif d == 2:
                            ny = yy + 1; nx = xx; gy = fy + 1; gx = fx
                        else:
                            ny = yy - 1; nx = xx; gy = fy - 1; gx = fx
                        ny %= Ly; nx %= Lx
                        if lab[ny, nx] < 0 and S[r, ny, nx, 2] < sz_thr:
                            lab[ny, nx] = nc
                            top += 1
                            stack_y[top] = ny; stack_x[top] = nx
                            uy_[top] = gy; ux_[top] = gx
                cy = sy / size; cx = sx / size
                # charge and charge-weighted centre in a disc of radius rq (plaquette centres at +0.5)
                Qc = 0.0; wy = 0.0; wx = 0.0
                irq = int(rq) + 1
                for dy in range(-irq, irq + 1):
                    for dx in range(-irq, irq + 1):
                        py = np.floor(cy) + dy
                        px = np.floor(cx) + dx
                        ry = py + 0.5 - cy; rx = px + 0.5 - cx
                        if ry * ry + rx * rx <= rq * rq:
                            qq = q[r, int(py) % Ly, int(px) % Lx]
                            Qc += qq
                            wy += qq * ry; wx += qq * rx
                if abs(Qc) > 0.3:
                    cy += wy / Qc; cx += wx / Qc
                if nc < max_per:
                    out[r, nc, 0] = cx % Lx
                    out[r, nc, 1] = cy % Ly
                    out[r, nc, 2] = Qc
                    out[r, nc, 3] = size
                nc += 1
        count[r] = min(nc, max_per)
    return out, count


@njit(cache=True)
def excluded_rows(centres, n, Ly, Lx, rex):
    """Number of sites per row y within distance rex of any centre (union of discs, periodic)."""
    mask = np.zeros((Ly, Lx), np.bool_)
    irx = int(rex) + 1
    for k in range(n):
        cx = centres[k, 0]; cy = centres[k, 1]
        for dy in range(-irx, irx + 1):
            for dx in range(-irx, irx + 1):
                py = int(np.floor(cy)) + dy
                px = int(np.floor(cx)) + dx
                ry = py - cy; rx = px - cx
                if ry * ry + rx * rx <= rex * rex:
                    mask[py % Ly, px % Lx] = True
    rows = np.zeros(Ly)
    for y in range(Ly):
        for x in range(Lx):
            if mask[y, x]:
                rows[y] += 1
    return rows
