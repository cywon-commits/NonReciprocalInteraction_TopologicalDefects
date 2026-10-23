from skx import *
import json
J=1.0; D=0.4; a=0.3; b=0.6
m=Model(19,33,J,D,0.5*D*D/J); S0=np.load("S_B0.5.npy"); out={}
for ue in [0.15,0.2]:
    S,V,w1,w2,r=find_V(m,S0.copy(),np.array([ue,0.]),a,b,iters=6,n=800); out[ue]=[]
    for n in (1,2,3):
        lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,(2*np.pi*n/190,0.))); lam=lam[np.abs(lam)>1e-9]; out[ue].append(float(lam.real.max()))
    print(ue,out[ue],flush=True)
json.dump(out,open("rates_T0.json","w"))
