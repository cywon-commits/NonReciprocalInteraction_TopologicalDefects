"""Lattice spin model, stochastic LLG + Zhang-Li integrator, topological charge.

Conventions (see README):
  square lattice, a = 1, periodic in x and y, |S_i| = 1, J = gamma = mu = k_B = 1.
  E = -J sum S_i.S_j - sum D_ij.(S_i x S_j) - B sum S^z - K sum (S^z)^2
  D_ij for a bond i -> i+x is Dx (3-vector), for i -> i+y is Dy; D_ji = -D_ij.
  Neel (interfacial) DMI: Dx = D (z x x) = D y,  Dy = D (z x y) = -D x.
  Gilbert form: dS/dt = -S x (H + h) + alpha S x dS/dt + tau,
  tau = -P (u.grad) S + beta S x (u.grad) S,  P = 1 - S S^T.
  Integrated in LL form with Stratonovich Heun, renormalising |S| = 1
  after both predictor and corrector, same noise in both stages.
  Arrays: S[r, y, x, 3]  (replica, row y, column x, component).
"""
import numpy as np
from numba import njit, prange

FLOAT = np.float64


@njit(inline="always")
def _cross(a0, a1, a2, b0, b1, b2):
    return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0


@njit(inline="always")
def _rhs_site(S, r, y, x, Ly, Lx, J, Dx, Dy, B, K, alpha, beta, ux, uy, omz, h0, h1, h2):
    """Right-hand side of the LL-form equation at one site (S at that site read from S).

    omz[y, x] = Omega adds the pure spin precession -Omega z x S (stage B' point P10: Gilbert
    D_t = d_t + Omega z x, which in LL form is exactly -Omega z x S with no damping partner)."""
    xp = x + 1
    if xp == Lx:
        xp = 0
    xm = x - 1
    if xm < 0:
        xm = Lx - 1
    yp = y + 1
    if yp == Ly:
        yp = 0
    ym = y - 1
    if ym < 0:
        ym = Ly - 1
    s0 = S[r, y, x, 0]
    s1 = S[r, y, x, 1]
    s2 = S[r, y, x, 2]
    a0 = S[r, y, xp, 0]; a1 = S[r, y, xp, 1]; a2 = S[r, y, xp, 2]
    b0 = S[r, y, xm, 0]; b1 = S[r, y, xm, 1]; b2 = S[r, y, xm, 2]
    c0 = S[r, yp, x, 0]; c1 = S[r, yp, x, 1]; c2 = S[r, yp, x, 2]
    d0 = S[r, ym, x, 0]; d1 = S[r, ym, x, 1]; d2 = S[r, ym, x, 2]
    # exchange
    H0 = J * (a0 + b0 + c0 + d0)
    H1 = J * (a1 + b1 + c1 + d1)
    H2 = J * (a2 + b2 + c2 + d2)
    # DMI: H_i += (S_{i+x} - S_{i-x}) x Dx + (S_{i+y} - S_{i-y}) x Dy
    e0, e1, e2 = _cross(a0 - b0, a1 - b1, a2 - b2, Dx[0], Dx[1], Dx[2])
    f0, f1, f2 = _cross(c0 - d0, c1 - d1, c2 - d2, Dy[0], Dy[1], Dy[2])
    H0 += e0 + f0 + h0
    H1 += e1 + f1 + h1
    H2 += e2 + f2 + B + 2.0 * K * s2 + h2
    # -S x H - alpha S x (S x H)
    p0, p1, p2 = _cross(s0, s1, s2, H0, H1, H2)
    q0, q1, q2 = _cross(s0, s1, s2, p0, p1, p2)
    o0 = -p0 - alpha * q0
    o1 = -p1 - alpha * q1
    o2 = -p2 - alpha * q2
    uxv = ux[y, x]
    uyv = uy[y, x]
    if uxv != 0.0 or uyv != 0.0:
        # g = (u.grad) S, central differences, u at site i
        g0 = 0.5 * (uxv * (a0 - b0) + uyv * (c0 - d0))
        g1 = 0.5 * (uxv * (a1 - b1) + uyv * (c1 - d1))
        g2 = 0.5 * (uxv * (a2 - b2) + uyv * (c2 - d2))
        sg = s0 * g0 + s1 * g1 + s2 * g2
        pg0 = g0 - s0 * sg
        pg1 = g1 - s1 * sg
        pg2 = g2 - s2 * sg
        w0, w1, w2 = _cross(s0, s1, s2, g0, g1, g2)
        t0 = -pg0 + beta * w0
        t1 = -pg1 + beta * w1
        t2 = -pg2 + beta * w2
        v0, v1, v2 = _cross(s0, s1, s2, t0, t1, t2)
        o0 += t0 + alpha * v0
        o1 += t1 + alpha * v1
        o2 += t2 + alpha * v2
    inv = 1.0 / (1.0 + alpha * alpha)
    om = omz[y, x]
    # -Omega z x S = (Omega S_y, -Omega S_x, 0)
    return o0 * inv + om * s1, o1 * inv - om * s0, o2 * inv


@njit(inline="always")
def _splitmix(state, idx):
    """splitmix64 step on state[idx]; returns a uniform double in (0, 1)."""
    z = state[idx] + np.uint64(0x9E3779B97F4A7C15)
    state[idx] = z
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)) + np.uint64(1)) * (1.0 / 9007199254740994.0)


def _zig_tables():
    """Marsaglia-Tsang (2000) 128-layer ziggurat tables for the standard normal."""
    m1 = 2147483648.0
    dn = 3.442619855899
    tn = dn
    vn = 9.91256303526217e-3
    kn = np.zeros(128, np.int64)
    wn = np.zeros(128)
    fn = np.zeros(128)
    q = vn / np.exp(-0.5 * dn * dn)
    kn[0] = int((dn / q) * m1)
    kn[1] = 0
    wn[0] = q / m1
    wn[127] = dn / m1
    fn[0] = 1.0
    fn[127] = np.exp(-0.5 * dn * dn)
    for i in range(126, 0, -1):
        dn = np.sqrt(-2.0 * np.log(vn / dn + np.exp(-0.5 * dn * dn)))
        kn[i + 1] = int((dn / tn) * m1)
        tn = dn
        fn[i] = np.exp(-0.5 * dn * dn)
        wn[i] = dn / m1
    return kn, wn, fn


ZIG_KN, ZIG_WN, ZIG_FN = _zig_tables()


@njit(inline="always")
def _bits32(state, idx):
    """Signed 32-bit integer from one splitmix64 draw."""
    z = state[idx] + np.uint64(0x9E3779B97F4A7C15)
    state[idx] = z
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.int64(z >> np.uint64(32)) - 2147483648


@njit(inline="always")
def _gauss(state, idx, kn, wn, fn):
    """Standard normal by the ziggurat method (exact distribution)."""
    hz = _bits32(state, idx)
    iz = hz & 127
    if abs(hz) < kn[iz]:
        return hz * wn[iz]
    r = 3.442620
    while True:
        x = hz * wn[iz]
        if iz == 0:
            while True:
                x = -np.log(_splitmix(state, idx)) * 0.2904764
                yv = -np.log(_splitmix(state, idx))
                if yv + yv >= x * x:
                    break
            return r + x if hz > 0 else -r - x
        if fn[iz] + _splitmix(state, idx) * (fn[iz - 1] - fn[iz]) < np.exp(-0.5 * x * x):
            return x
        hz = _bits32(state, idx)
        iz = hz & 127
        if abs(hz) < kn[iz]:
            return hz * wn[iz]


@njit(cache=True)
def gauss_samples(state, n, kn, wn, fn):
    out = np.empty(n)
    for i in range(n):
        out[i] = _gauss(state, 0, kn, wn, fn)
    return out


@njit(parallel=True, fastmath=True, cache=True)
def heun_steps(S, Sp, K1, noise, nsteps, dt, J, Dx, Dy, B, K, alpha, beta, ux, uy, omz, T, rng, kn, wn, fn):
    """Advance S in place by nsteps Stratonovich-Heun steps.

    Sp, K1, noise are work arrays; rng is a uint64 state array of length R*Ly (one stream per row).
    """
    R, Ly, Lx, _ = S.shape
    sig = np.sqrt(2.0 * alpha * T / dt) if T > 0 else 0.0
    nrow = R * Ly
    for _ in range(nsteps):
        # predictor
        for idx in prange(nrow):
            r = idx // Ly
            y = idx - r * Ly
            for x in range(Lx):
                if sig > 0.0:
                    h0 = sig * _gauss(rng, idx, kn, wn, fn)
                    h1 = sig * _gauss(rng, idx, kn, wn, fn)
                    h2 = sig * _gauss(rng, idx, kn, wn, fn)
                else:
                    h0 = 0.0; h1 = 0.0; h2 = 0.0
                noise[r, y, x, 0] = h0
                noise[r, y, x, 1] = h1
                noise[r, y, x, 2] = h2
                k0, k1, k2 = _rhs_site(S, r, y, x, Ly, Lx, J, Dx, Dy, B, K, alpha, beta, ux, uy, omz, h0, h1, h2)
                K1[r, y, x, 0] = k0
                K1[r, y, x, 1] = k1
                K1[r, y, x, 2] = k2
                n0 = S[r, y, x, 0] + dt * k0
                n1 = S[r, y, x, 1] + dt * k1
                n2 = S[r, y, x, 2] + dt * k2
                nn = 1.0 / np.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
                Sp[r, y, x, 0] = n0 * nn
                Sp[r, y, x, 1] = n1 * nn
                Sp[r, y, x, 2] = n2 * nn
        # corrector (writes S[i] only, reads Sp neighbours)
        for idx in prange(nrow):
            r = idx // Ly
            y = idx - r * Ly
            for x in range(Lx):
                k0, k1, k2 = _rhs_site(Sp, r, y, x, Ly, Lx, J, Dx, Dy, B, K, alpha, beta, ux, uy, omz,
                                       noise[r, y, x, 0], noise[r, y, x, 1], noise[r, y, x, 2])
                n0 = S[r, y, x, 0] + 0.5 * dt * (K1[r, y, x, 0] + k0)
                n1 = S[r, y, x, 1] + 0.5 * dt * (K1[r, y, x, 1] + k1)
                n2 = S[r, y, x, 2] + 0.5 * dt * (K1[r, y, x, 2] + k2)
                nn = 1.0 / np.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
                S[r, y, x, 0] = n0 * nn
                S[r, y, x, 1] = n1 * nn
                S[r, y, x, 2] = n2 * nn


def make_rng(seed, n):
    """Independent splitmix64 stream states (one per lattice row and replica)."""
    ss = np.random.SeedSequence(seed)
    return ss.generate_state(n, dtype=np.uint64)


@njit(inline="always")
def _omega(a0, a1, a2, b0, b1, b2, c0, c1, c2):
    x0, x1, x2 = _cross(b0, b1, b2, c0, c1, c2)
    num = a0 * x0 + a1 * x1 + a2 * x2
    den = 1.0 + (a0 * b0 + a1 * b1 + a2 * b2) + (b0 * c0 + b1 * c1 + b2 * c2) + (c0 * a0 + c1 * a1 + c2 * a2)
    return 2.0 * np.arctan2(num, den)


@njit(parallel=True, cache=True)
def charge_map(S, q, both):
    """Plaquette charge q[r, y, x] for plaquette (i, i+x, i+x+y, i+y), i = (y, x).

    both = False: one diagonal (i -> i+x+y), integer total, used for event detection.
    both = True : average of the two diagonal splittings (used for symmetry averages).
    """
    R, Ly, Lx, _ = S.shape
    inv4pi = 1.0 / (4.0 * np.pi)
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
            # counter-clockwise triangles: (i, i+x, i+x+y), (i, i+x+y, i+y)
            w = _omega(i0, i1, i2, a0, a1, a2, b0, b1, b2) + _omega(i0, i1, i2, b0, b1, b2, c0, c1, c2)
            if both:
                # other diagonal: (i, i+x, i+y), (i+x, i+x+y, i+y)
                w2 = _omega(i0, i1, i2, a0, a1, a2, c0, c1, c2) + _omega(a0, a1, a2, b0, b1, b2, c0, c1, c2)
                w = 0.5 * (w + w2)
            q[r, y, x] = w * inv4pi


@njit(parallel=True, cache=True)
def energy_per_replica(S, J, Dx, Dy, B, K):
    R, Ly, Lx, _ = S.shape
    E = np.zeros(R)
    for r in prange(R):
        e = 0.0
        for y in range(Ly):
            yp = (y + 1) % Ly
            for x in range(Lx):
                xp = (x + 1) % Lx
                s0 = S[r, y, x, 0]; s1 = S[r, y, x, 1]; s2 = S[r, y, x, 2]
                a0 = S[r, y, xp, 0]; a1 = S[r, y, xp, 1]; a2 = S[r, y, xp, 2]
                c0 = S[r, yp, x, 0]; c1 = S[r, yp, x, 1]; c2 = S[r, yp, x, 2]
                e -= J * (s0 * a0 + s1 * a1 + s2 * a2 + s0 * c0 + s1 * c1 + s2 * c2)
                p0, p1, p2 = _cross(s0, s1, s2, a0, a1, a2)
                e -= Dx[0] * p0 + Dx[1] * p1 + Dx[2] * p2
                p0, p1, p2 = _cross(s0, s1, s2, c0, c1, c2)
                e -= Dy[0] * p0 + Dy[1] * p1 + Dy[2] * p2
                e -= B * s2 + K * s2 * s2
        E[r] = e
    return E


def dmi_vectors(D, kind="neel", Dy_scale=1.0):
    """Return (Dx, Dy) bond vectors. kind='neel': D z x e;  kind='bloch': D_x x, D_y y."""
    if kind == "neel":
        return np.array([0.0, D, 0.0]), np.array([-D * Dy_scale, 0.0, 0.0])
    if kind == "bloch":
        return np.array([D, 0.0, 0.0]), np.array([0.0, D * Dy_scale, 0.0])
    raise ValueError(kind)


def shear_profiles(kind, u0, u3, Ly):
    """Analytic u_x(y), omega(y) = -d_y u_x and psi(y) = int u_x dy for the y-dependent drives.

    shear : u_x = u0 sin(ky);  shear3: u_x = u0 sin(ky) + u3 sin(3ky);  k = 2 pi / Ly.
    rot   : no flow; omega(y) is the profile of the u0 shear (used for the P10 precession term).
    """
    k = 2 * np.pi / Ly
    a3 = u3 if kind == "shear3" else 0.0
    ux = lambda y: u0 * np.sin(k * y) + a3 * np.sin(3 * k * y)
    om = lambda y: -k * (u0 * np.cos(k * y) + 3 * a3 * np.cos(3 * k * y))
    psi = lambda y: -(u0 * np.cos(k * y) + a3 * np.cos(3 * k * y) / 3) / k
    return ux, om, psi


def drive_field(kind, u0, Ly, Lx, theta=0.0, u3=0.0):
    """u(r) arrays (ux, uy) of shape (Ly, Lx). Row index = y coordinate."""
    y = np.arange(Ly, dtype=FLOAT)[:, None] * np.ones((1, Lx))
    if kind in ("shear", "shear3"):
        ux = shear_profiles(kind, u0, u3, Ly)[0](y)
        uy = np.zeros_like(ux)
    elif kind == "rot":
        ux = np.zeros((Ly, Lx)); uy = np.zeros((Ly, Lx))
    elif kind == "uniform":
        ux = u0 * np.cos(theta) * np.ones((Ly, Lx))
        uy = u0 * np.sin(theta) * np.ones((Ly, Lx))
    elif kind == "none":
        ux = np.zeros((Ly, Lx)); uy = np.zeros((Ly, Lx))
    else:
        raise ValueError(kind)
    return np.ascontiguousarray(ux), np.ascontiguousarray(uy)


def omega_of_y(kind, u0, Ly, y):
    """Vorticity omega = z.(curl u) = -d_y u_x at continuous position y."""
    if kind == "shear":
        return -2 * np.pi * u0 / Ly * np.cos(2 * np.pi * y / Ly)
    return np.zeros_like(np.asarray(y, dtype=float))


class Integrator:
    def __init__(self, S, J=1.0, D=0.3, B=0.12, K=0.0, alpha=0.1, beta=0.0, T=0.0,
                 dt=0.02, drive="none", u0=0.0, theta=0.0, dmi="neel", Dy_scale=1.0, seed=0,
                 u3=0.0, rot_sign=1.0):
        self.S = np.ascontiguousarray(S, dtype=FLOAT)
        R, Ly, Lx, _ = self.S.shape
        self.Sp = np.empty_like(self.S)
        self.K1 = np.empty_like(self.S)
        self.noise = np.empty_like(self.S)
        self.Dx, self.Dy = dmi_vectors(D, dmi, Dy_scale)
        self.ux, self.uy = drive_field(drive, u0, Ly, Lx, theta, u3)
        self.omz = np.zeros((Ly, Lx))
        if drive == "rot":  # Omega = omega/2 of the u0 shear profile, at the site rows
            y = np.arange(Ly, dtype=FLOAT)[:, None] * np.ones((1, Lx))
            self.omz = np.ascontiguousarray(rot_sign * 0.5 * shear_profiles("shear", u0, 0.0, Ly)[1](y))
        self.p = dict(J=J, D=D, B=B, K=K, alpha=alpha, beta=beta, T=T, dt=dt)
        self.t = 0.0
        self.rng = make_rng(seed, R * Ly)

    def run(self, nsteps):
        p = self.p
        heun_steps(self.S, self.Sp, self.K1, self.noise, nsteps, p["dt"], p["J"], self.Dx, self.Dy,
                   p["B"], p["K"], p["alpha"], p["beta"], self.ux, self.uy, self.omz, p["T"], self.rng,
                   ZIG_KN, ZIG_WN, ZIG_FN)
        self.t += nsteps * p["dt"]

    def energy(self):
        p = self.p
        return energy_per_replica(self.S, p["J"], self.Dx, self.Dy, p["B"], p["K"])


def uniform_state(R, Ly, Lx, mz=1.0):
    S = np.zeros((R, Ly, Lx, 3))
    S[..., 2] = mz
    return S


def skyrmion_state(R, Ly, Lx, radius=3.0, x0=None, y0=None, helicity=0.0, w=1.5):
    """Neel skyrmion, core -z, background +z (helicity 0: in-plane part radial)."""
    x0 = Lx / 2 if x0 is None else x0
    y0 = Ly / 2 if y0 is None else y0
    yy, xx = np.meshgrid(np.arange(Ly), np.arange(Lx), indexing="ij")
    dx = (xx - x0 + Lx / 2) % Lx - Lx / 2
    dy = (yy - y0 + Ly / 2) % Ly - Ly / 2
    rho = np.hypot(dx, dy)
    th = 2 * np.arctan(np.exp((radius - rho) / w))  # theta: pi at core, 0 far away
    phi = np.arctan2(dy, dx) + helicity
    S = np.empty((Ly, Lx, 3))
    S[..., 0] = np.sin(th) * np.cos(phi)
    S[..., 1] = np.sin(th) * np.sin(phi)
    S[..., 2] = np.cos(th)
    return np.repeat(S[None], R, axis=0)
