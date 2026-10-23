from skx import *
import json
from scipy.sparse.linalg import eigs
# Linear decay rates Gamma_k(u) of the phonon for the k allowed in a 4x2 supercell (76x66), co-moving frame.
J=1.0; D=0.4; a=0.3; b=0.6
m=Model(19,33,J,D,0.5*D*D/J); S0=np.load("S_B0.5.npy")
ks=[(2*np.pi*mm/76,2*np.pi*ll/66) for (mm,ll) in [(1,0),(2,0),(0,1),(1,1),(-1,1),(2,1),(-2,1),(3,0)]]
out={}
for ue in [0.0,0.03,0.05,0.07]:
    if ue==0: S,w1,w2=S0,np.zeros(2),np.zeros(2); V=np.zeros(2)
    else: S,V,w1,w2,r=find_V(m,S0.copy(),np.array([ue,0.]),a,b,iters=6,n=800)
    np.save(f"Scomov_{ue}.npy",S); out[str(ue)]=dict(w1=list(w1),w2=list(w2),V=list(V),rates=[])
    for k in ks:
        lam=eigs(bloch_jac(m,S,w1,w2,a,k),k=6,sigma=1e-9,return_eigenvectors=False)
        p=lam[np.argsort(np.abs(lam))[:2]]
        out[str(ue)]["rates"].append([k[0],k[1],float(p.real.max()),float(np.abs(p.imag).max())])
        print(f"u_eff={ue} k=({k[0]:+.3f},{k[1]:+.3f}): Re={p.real.max():+.3e} |Im|={np.abs(p.imag).max():.3e}",flush=True)
json.dump(out,open("bloch_rates.json","w"))
