import numpy as np, sys, json, time
from fastllg import Integrator
# Finite-T driven SkX in the co-moving frame; records topological-charge centroids of all skyrmions every 10 time units.
J,D=1.0,0.4; B=0.5*D*D; a=0.3
ue=sys.argv[1]; T=float(sys.argv[2]); tmax=float(sys.argv[3]); nx,ny=int(sys.argv[4]),int(sys.argv[5]); tag=sys.argv[6]; seed=int(sys.argv[7])
info=json.load(open(sys.argv[8]))[ue]; w1=np.array(info["w1"]); w2=np.array(info["w2"])
S=np.tile(np.load(f"Scomov_{float(ue)}.npy"),(nx,ny,1)); Lx,Ly=S.shape[:2]
def qdens(S):
    gx=(np.roll(S,-1,0)-np.roll(S,1,0))/2; gy=(np.roll(S,-1,1)-np.roll(S,1,1))/2; return np.sum(S*np.cross(gx,gy),-1)
X,Y=np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij'); mz=S[...,2]; R=[]
for i in range(Lx):
    for j in range(Ly):
        v=mz[i,j]
        if v<0 and all(v<mz[(i+di)%Lx,(j+dj)%Ly] for di in (-1,0,1) for dj in (-1,0,1) if (di,dj)!=(0,0)): R.append((i,j))
R=np.array(R,float); q0=qdens(S); win=[]
for (ri,rj) in R:
    dx=(X-ri+Lx/2)%Lx-Lx/2; dy=(Y-rj+Ly/2)%Ly-Ly/2; msk=(dx**2+dy**2)<64; w=q0[msk]
    cx,cy=(w*dx[msk]).sum()/w.sum(),(w*dy[msk]).sum()/w.sum(); win.append((np.flatnonzero(msk.ravel()),dx[msk]-cx,dy[msk]-cy))
R=R+np.array([[0,0]])  # reference (approx); displacements are measured relative to the static centroid
np.save(f"R_{tag}.npy",R)
I=Integrator(S,J,D,B,a,w1,w2,T,0.15,seed); I.run(int(2000/0.15))
rec=[]; t0=time.time(); t=0.0
while t<tmax:
    I.run(int(round(10/0.15))); t+=10
    q=qdens(I.S).ravel(); rec.append([[(q[ix]*dx).sum()/q[ix].sum(),(q[ix]*dy).sum()/q[ix].sum()] for ix,dx,dy in win])
    if len(rec)%5000==0: np.save(f"disp_{tag}.npy",np.array(rec,np.float32)); print(f"t={t:.0f} wall={time.time()-t0:.0f}s",flush=True)
np.save(f"disp_{tag}.npy",np.array(rec,np.float32)); print("done",time.time()-t0,flush=True)
