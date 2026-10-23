from sot import *
from scipy.sparse.linalg import eigs
J,D=1.0,0.4; B=0.5*D*D; a=0.3; S0=np.load("S_B0.5.npy")
m=SOTModel(19,33,J,D,B,0.,0.004,(0,1,0)); S,V,r=find_drift(m,S0.copy(),a); m0=SOTModel(19,33,J,D,B,0.,0.,(0,1,0))
for nk in [40,20,10]:
    k=2*np.pi/(19*nk)
    for ang in [0,45,90]:
        kv=(k*np.cos(np.radians(ang)),k*np.sin(np.radians(ang)))
        p=eigs(bloch_jac(m,S,V,-a*V,a,kv),k=6,sigma=1e-9,return_eigenvectors=False); p=p[np.argsort(np.abs(p))[:2]]
        p0=eigs(bloch_jac(m0,S0,np.zeros(2),np.zeros(2),a,kv),k=6,sigma=1e-9,return_eigenvectors=False); p0=p0[np.argsort(np.abs(p0))[:2]]
        G0=-p0.real.mean(); w0=abs(p0.imag).max(); split=(p.real.max()-p.real.min())/2
        print(f"k={k:.4f} angle={ang:2d}: Gamma0={G0:.3e} omega={w0:.3e}  half-split={split:.3e}  x=split/Gamma0={split/G0:.3f}  x*k={split/G0*k:.5f}",flush=True)
print("drift V",V)
