from skx import *
import json
from scipy.sparse.linalg import eigs
J=1.0; D=0.4; a=0.3; b=0.6
m=Model(19,33,J,D,0.5*D*D/J); info=json.load(open("bloch_rates.json")); out={}
for ue in ["0.0","0.05","0.07"]:
    S=np.load(f"Scomov_{ue}.npy"); w1=np.array(info[ue]["w1"]); w2=np.array(info[ue]["w2"]); out[ue]={}
    for n in (1,2):
        k=(2*np.pi*n/76,0.)
        lam=eigs(bloch_jac(m,S,w1,w2,a,k),k=6,sigma=1e-9,return_eigenvectors=False); p=lam[np.argsort(np.abs(lam))[:2]]
        out[ue][n]=[[float(z.real),float(z.imag)] for z in sorted(p,key=lambda z:z.imag)]
        print(ue,n,np.round(sorted(p,key=lambda z:z.imag),6),flush=True)
json.dump(out,open("branches.json","w"))
