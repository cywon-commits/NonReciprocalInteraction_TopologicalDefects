from skx import *
import json
from scipy.sparse.linalg import eigs
J=1.0; D=0.4; a=0.3; b=0.6
m=Model(19,33,J,D,0.5*D*D/J); info=json.load(open("bloch_rates.json")); out={}
qs=[(mm,ll) for mm in (1,2) for ll in (0,1)]+[(0,1),(-1,1),(-2,1)]
for ue in ["0.0","0.07"]:
    S=np.load(f"Scomov_{ue}.npy"); w1=np.array(info[ue]["w1"]); w2=np.array(info[ue]["w2"]); out[ue]={}
    for (mm,ll) in qs:
        k=(2*np.pi*mm/76,2*np.pi*ll/66)
        lam=eigs(bloch_jac(m,S,w1,w2,a,k),k=6,sigma=1e-9,return_eigenvectors=False); p=lam[np.argsort(np.abs(lam))[:2]]
        out[ue][f"{mm},{ll}"]=[[float(z.real),float(z.imag)] for z in sorted(p,key=lambda z:z.imag)]
        print(ue,mm,ll,np.round(sorted(p,key=lambda z:z.imag),6),flush=True)
json.dump(out,open("branches_big.json","w"))
