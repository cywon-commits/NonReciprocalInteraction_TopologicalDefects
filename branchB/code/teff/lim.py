import numpy as np, json, sys
from scipy.linalg import logm
def load(ue): r=np.load(f"traj_u{ue}_T0.01_s{ {'0.0':2,'0.05':3,'0.07':1}[ue] }.npy"); return r[:,0].real, r[:,1:]
def lim(z,dtau,lag):
    n=len(z)-lag; C0=(z[:n,:,None]*z[:n,None,:].conj()).mean(0); Ct=(z[lag:lag+n,:,None]*z[:n,None,:].conj()).mean(0)
    A=logm(Ct@np.linalg.inv(C0))/(lag*dtau); ev,V=np.linalg.eig(A); o=np.argsort(ev.imag); return ev[o],V[:,o]
def blocks(x,nb=10):
    L=len(x)//nb; b=np.array([x[i*L:(i+1)*L].mean() for i in range(nb)]); return b.mean(), b.std()/np.sqrt(nb)
br=json.load(open("branches.json"))
avail=[u for u in ["0.0","0.05","0.07"] if __import__('os').path.exists(f"traj_u{u}_T0.01_s{ {'0.0':2,'0.05':3,'0.07':1}[u] }.npy")]
t0,z0=load("0.0"); dtau=t0[1]-t0[0]; ev0,V0=lim(z0,dtau,10); W0=np.linalg.inv(V0)
base={}
for u in avail:
    t,z=load(u); ev,V=lim(z,dtau,10); a=(W0@z.T).T
    print(f"u_eff={u}: span={t[-1]-t[0]:.0f}  LIM eigenvalues {np.round(ev*1e3,3)} x1e-3   Bloch {[ (round(x*1e3,3),round(y*1e3,3)) for x,y in br[u]['1']]} x1e-3")
    for bidx in (0,1):
        p=np.abs(a[:,bidx])**2; m,e=blocks(p); kurt=np.mean(p**2)/np.mean(p)**2
        if u=="0.0": base[bidx]=m
        G0=-br["0.0"]["1"][bidx][0]; Gu=-br[u]["1"][bidx][0]
        print(f"   branch {bidx} (omega {'<0' if bidx==0 else '>0'}): <|a|^2>={m:.4f}±{e:.4f}  ratio={m/base[bidx]:.2f}±{e/base[bidx]:.2f}  predicted Gamma0/Gamma={G0/Gu:.2f}  kurtosis={kurt:.2f} (Gaussian 2)")
