from series import *
import numpy as np
def kappa_shells(d,B,nsh):
    Lx,Ly=d['Lx'],d['Ly']
    G2=lambda t:(t[0]/Lx)**2+(t[1]/Ly)**2
    vals=sorted(set(round(G2(t),8) for t in [(p,q) for p in range(-2,3) for q in range(-4,5)]))
    keep=set(vals[:nsh+1])
    def sol(K,C,tags):
        idx=np.array([i for i,t in enumerate(tags) if round(G2(t[1]),8) in keep]); s=np.zeros(len(C),complex)
        s[idx]=np.linalg.lstsq(K[np.ix_(idx,idx)],-C[idx],rcond=1e-12)[0]; return s
    return kappa_from(d,B,sol)
g=json.load(open("grid_static.json"))
for key in ["19_0.3","19_0.5","19_0.85","13_0.5","23_0.5","23_0.7"]:
    Lx,bb=key.split('_'); d=setup(np.load(f"Sgrid_{Lx}_{bb}.npy"),float(bb)); B=build(d,2)
    print(key,f"exact={g[key]['kLT']:.3f}",' '.join(f"n{n}={kappa_shells(d,B,n):.3f}" for n in range(0,9)),flush=True)
