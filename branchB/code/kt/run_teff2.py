import numpy as np, sys, json, time
from fastllg import Integrator
from melt import sk_centers
# Finite-T driven SkX (co-moving frame). Tracks every skyrmion with a moving disk window (topological-charge centroid).
J,D=1.0,0.4; B=0.5*D*D; a=0.3
ue=sys.argv[1]; T=float(sys.argv[2]); tmax=float(sys.argv[3]); nx,ny=int(sys.argv[4]),int(sys.argv[5]); tag=sys.argv[6]; seed=int(sys.argv[7])
info=json.load(open(sys.argv[8]))[ue]; w1=np.array(info["w1"]); w2=np.array(info["w2"])
S=np.tile(np.load(f"Scomov_{float(ue)}.npy"),(nx,ny,1)); Lx,Ly=S.shape[:2]
def qdens(S):
    gx=(np.roll(S,-1,0)-np.roll(S,1,0))/2; gy=(np.roll(S,-1,1)-np.roll(S,1,1))/2; return np.sum(S*np.cross(gx,gy),-1)
ox,oy=np.meshgrid(np.arange(-8,9),np.arange(-8,9),indexing='ij'); m=(ox**2+oy**2)<64; ox,oy=ox[m],oy[m]
def centroids(q,P):
    c=np.round(P).astype(int); ix=(c[:,0:1]+ox[None,:])%Lx; iy=(c[:,1:2]+oy[None,:])%Ly; w=q[ix,iy]
    return c+np.stack([(w*ox).sum(1)/w.sum(1),(w*oy).sum(1)/w.sum(1)],1)
P=sk_centers(S)[:,:2]; P=centroids(qdens(S),P); P=centroids(qdens(S),P); R=P.copy(); np.save(f"R_{tag}.npy",R)
print("skyrmions:",len(R),flush=True)
I=Integrator(S,J,D,B,a,w1,w2,T,0.15,seed); rec=[]; t=0.0; t0=time.time()
I.run(int(2000/0.15)); P=centroids(qdens(I.S),P)
while t<tmax:
    I.run(int(round(10/0.15))); t+=10
    Pn=centroids(qdens(I.S),P); d=Pn-P; d[:,0]-=Lx*np.round(d[:,0]/Lx); d[:,1]-=Ly*np.round(d[:,1]/Ly); P=P+d   # unwrapped track
    rec.append(P-R)
    if len(rec)%5000==0: np.save(f"disp_{tag}.npy",np.array(rec,np.float32)); print(f"t={t:.0f} wall={time.time()-t0:.0f}s",flush=True)
np.save(f"disp_{tag}.npy",np.array(rec,np.float32)); print("done",time.time()-t0,flush=True)
