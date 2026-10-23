import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skx import GD
S1=np.load("S_B0.5.npy"); S0=np.concatenate([S1]*4,axis=0)
Gs=[(0,2),(4,32),(72,32),(0,31),(72,1),(4,1)]
def bragg(S):
    F=np.abs(np.fft.fft2(S[...,2]-S[...,2].mean()))**2
    return sum(F[g] for g in Gs)/F.sum()
def count_cores(S):
    # local minima of m_z below -0.5 (periodic)
    mz=S[...,2]; c=0
    for i in range(mz.shape[0]):
        for j in range(mz.shape[1]):
            v=mz[i,j]
            if v<-0.5 and all(v<=mz[(i+di)%mz.shape[0],(j+dj)%mz.shape[1]] for di in (-1,0,1) for dj in (-1,0,1) if (di,dj)!=(0,0)): c+=1
    return c
fig,ax=plt.subplots(1,3,figsize=(12,2.6))
for a_,(S,t) in zip(ax,[(S0,"static lattice"),(np.load("Sfin_u250.npy"),"u_eff = 0.25, t = 12000"),(np.load("Sfin_u400.npy"),"u_eff = 0.40, t = 8000")]):
    a_.imshow(S[...,2].T,origin="lower",cmap="RdBu_r",vmin=-1,vmax=1); a_.set_title(t,fontsize=10); a_.set_xticks([]); a_.set_yticks([])
    print(f"{t}: Q={GD(S)[2]:.3f}  Bragg weight={bragg(S):.3f}  cores={count_cores(S)}")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/skx_landau/skx_endstate.png",dpi=130)
