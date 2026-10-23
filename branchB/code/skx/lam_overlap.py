from skx import *
# Doppler efficiency from first-order (overlap / Hellmann-Feynman) perturbation:
# alpha=0, reactive advection only (w1), frozen u=0 background.
# A Bloch mode Psi_k gets omega -> omega - w1.K,  K = <Psi|tau_z(-i grad)|Psi>/<Psi|tau_z|Psi>,  lambda = K/k.
J=1.0; D=0.4; Lx,Ly=19,33; z=np.zeros(2)
S0=np.load("S0.npy")
def phonon(m,S,w1,k):
    lam=np.linalg.eigvals(bloch_jac(m,S,w1,z,0.0,k))
    p=lam[np.argsort(np.abs(lam))[:2]]
    return np.sort(p.imag)
for bb in [0.3,0.5,0.7,0.85]:
    m=Model(Lx,Ly,J,D,bb*D*D/J)
    S=rk4(m,S0.copy(),z,z,0.5,0.1,12000)
    np.save(f"S_B{bb}.npy",S)
    for nk in [20,40]:
        kx=2*np.pi/(Lx*nk); k=(kx,0.); dw=1e-4
        pp=phonon(m,S,np.array([dw,0.]),k); pm=phonon(m,S,np.array([-dw,0.]),k)
        K=-(pp-pm)/(2*dw)
        print(f"B/(D^2/J)={bb} k={kx:.5f}: K/k (two branches) = {np.round(K/kx,4)}",flush=True)
