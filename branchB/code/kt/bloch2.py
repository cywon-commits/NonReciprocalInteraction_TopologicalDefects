from skx import *
import sys, json
from scipy.sparse.linalg import eigs
J=1.0; D=0.4; a=0.3; b=0.6
m=Model(19,33,J,D,0.5*D*D/J); S0=np.load("S_B0.5.npy")
Lx,Ly=int(sys.argv[1]),int(sys.argv[2]); ues=[float(x) for x in sys.argv[3].split(',')]; kmax=float(sys.argv[4]); tag=sys.argv[5]
ks=[]
for mm in range(-Lx//19*19//2, Lx//2+1):
    for ll in range(0,Ly//2+1):
        k=(2*np.pi*mm/Lx,2*np.pi*ll/Ly)
        if (mm,ll)==(0,0) or np.hypot(*k)>kmax: continue
        if ll==0 and mm<0: continue
        ks.append(k)
out={}
for ue in ues:
    if ue==0: S,w1,w2=S0,np.zeros(2),np.zeros(2)
    else: S,V,w1,w2,r=find_V(m,S0.copy(),np.array([ue,0.]),a,b,iters=6,n=800)
    np.save(f"Scomov_{ue}.npy",S); out[str(ue)]=dict(w1=list(w1),w2=list(w2),k=[])
    for k in ks:
        lam=eigs(bloch_jac(m,S,w1,w2,a,k),k=6,sigma=1e-9,return_eigenvectors=False); p=sorted(lam[np.argsort(np.abs(lam))[:2]],key=lambda z:z.imag)
        out[str(ue)]["k"].append([k[0],k[1],[[float(z.real),float(z.imag)] for z in p]])
        print(ue,np.round(k,4),np.round(p,6),flush=True)
json.dump(out,open(f"bloch_{tag}.json","w"))
