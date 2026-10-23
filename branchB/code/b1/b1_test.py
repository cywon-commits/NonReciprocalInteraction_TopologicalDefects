from sot import *
import sys
from scipy.sparse.linalg import eigs
J,D=1.0,0.4; B=0.5*D*D; a=0.3; S0=np.load("S_B0.5.npy")
kind=sys.argv[1]
def growth(m,S,V,k):
    lam=eigs(bloch_jac(m,S,V,-a*V,a,k),k=6,sigma=1e-9,return_eigenvectors=False); p=lam[np.argsort(np.abs(lam))[:2]]; return np.sort(p.real)
k=2*np.pi/(19*20)
for c in [0.0,0.002,0.004]:
    cF,cD=(c,0.) if kind=="FL" else (0.,c)
    m=SOTModel(19,33,J,D,B,cF,cD,(0,1,0)); S,V,r=find_drift(m,S0.copy(),a)
    gx=growth(m,S,V,(k,0.)); gy=growth(m,S,V,(0.,k))
    print(f"{kind} c={c}: drift V=({V[0]:+.4f},{V[1]:+.4f}) resid={r:.0e} | branches Re(lambda) k||x {gx*1e5} | k||y {gy*1e5}  (1e-5)",flush=True)
