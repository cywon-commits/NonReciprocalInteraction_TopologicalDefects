"""Lattice-consistent Thiele equation for the central-difference current torque (V3/V4 diagnosis).

Rigid motion dS/dt = -(v.d)S uses the true (spectral) derivative d, while the Zhang-Li torque uses the
central difference d^c. Projecting the Gilbert equation on the translation modes d_j S gives
    v_k (alpha d_jk + g_jk) = u_k (m_jk + beta dm_jk)
    g_jk = sum S.(d_k S x d_j S),  m_jk = sum S.(d^c_k S x d_j S),
    d_jk = sum d_j S . d_k S,      dm_jk = sum d_j S . d^c_k S.
With d^c -> d this reduces to the continuum Thiele equation of the request (V4 formula).
"""
import json, sys
import numpy as np


def derivs(S):
    L = S.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(L)
    ds = [np.real(np.fft.ifft(1j * k[None, :, None] * np.fft.fft(S, axis=1), axis=1)),
          np.real(np.fft.ifft(1j * k[:, None, None] * np.fft.fft(S, axis=0), axis=0))]
    dc = [0.5 * (np.roll(S, -1, 1) - np.roll(S, 1, 1)), 0.5 * (np.roll(S, -1, 0) - np.roll(S, 1, 0))]
    return ds, dc


def tensors(S):
    ds, dc = derivs(S)
    g = np.zeros((2, 2)); m = np.zeros((2, 2)); d = np.zeros((2, 2)); dm = np.zeros((2, 2))
    for j in range(2):
        for k in range(2):
            g[j, k] = np.einsum("yxi,yxi->", S, np.cross(ds[k], ds[j]))
            m[j, k] = np.einsum("yxi,yxi->", S, np.cross(dc[k], ds[j]))
            d[j, k] = np.einsum("yxi,yxi->", ds[j], ds[k])
            dm[j, k] = np.einsum("yxi,yxi->", ds[j], dc[k])
    return g, m, d, dm


def velocity(S, alpha, beta, u):
    g, m, d, dm = tensors(S)
    A = alpha * d + g          # v_k A_jk
    b = (m + beta * dm) @ np.asarray(u, float)
    return np.linalg.solve(A, b), dict(g=g.tolist(), m=m.tolist(), d=d.tolist(), dm=dm.tolist())


if __name__ == "__main__":
    from .single import relax_skyrmion
    v34 = json.load(open("results/validation/V34.json"))
    S0 = relax_skyrmion()[0]
    out = []
    for r in v34["runs"]:
        v, T = velocity(S0, r["alpha"], r["beta"], [r["u0"], 0.0])
        out.append(dict(alpha=r["alpha"], beta=r["beta"], vx=r["vx"], vy=r["vy"],
                        vx_lattice_thiele=float(v[0]), vy_lattice_thiele=float(v[1]),
                        rel_err_vx=float((r["vx"] - v[0]) / v[0]),
                        rel_err_vy=float((r["vy"] - v[1]) / v[1]) if abs(v[1]) > 1e-12 else None))
    res = dict(tensors=T, comparison=out)
    json.dump(res, open("results/validation/V34_lattice_thiele.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
