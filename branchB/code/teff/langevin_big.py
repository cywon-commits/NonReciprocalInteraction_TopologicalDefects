from skx import *
import sys, json, time
# Stochastic LLG (Gilbert noise, Stratonovich Heun) in the co-moving frame of the driven SkX; 4x2 supercell (76x66).
J=1.0; D=0.4; a=0.3; b=0.6; B=0.5*D*D/J
ue=float(sys.argv[1]); T=float(sys.argv[2]); tmax=float(sys.argv[3]); seed=int(sys.argv[4]); dt=float(sys.argv[5]) if len(sys.argv)>5 else 0.15
info=json.load(open("bloch_rates.json"))[str(ue)]; w1=np.array(info["w1"]); w2=np.array(info["w2"])
S=np.tile(np.load(f"Scomov_{ue}.npy"),(4,2,1)); Lx,Ly=S.shape[:2]
def field(S):
    xp=np.roll(S,-1,0); xm=np.roll(S,1,0); yp=np.roll(S,-1,1); ym=np.roll(S,1,1)
    H=J*(xp+xm+yp+ym)+B*zh+D*(np.cross(xp,dx_)+np.cross(dx_,xm)+np.cross(yp,dy_)+np.cross(dy_,ym))
    return H,(xp-xm)/2,(yp-ym)/2
def f(S,xi):
    H,gx,gy=field(S); H=H+xi
    Tq=-np.cross(S,H)+w1[0]*gx+w1[1]*gy+np.cross(S,w2[0]*gx+w2[1]*gy)
    Tq-=np.sum(Tq*S,-1)[...,None]*S
    return (Tq+a*np.cross(S,Tq))/(1+a*a)
# reference skyrmion positions = topological-charge maxima regions of the static texture
def qdens(S):
    gx=(np.roll(S,-1,0)-np.roll(S,1,0))/2; gy=(np.roll(S,-1,1)-np.roll(S,1,1))/2; return np.sum(S*np.cross(gx,gy),-1)
X,Y=np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
mz=S[...,2]; R=[]
for i in range(Lx):
    for j in range(Ly):
        v=mz[i,j]
        if v<0 and all(v<mz[(i+di)%Lx,(j+dj)%Ly] for di in (-1,0,1) for dj in (-1,0,1) if (di,dj)!=(0,0)): R.append((i,j))
R=np.array(R,float); q0=qdens(S)
win=[]
for (ri,rj) in R:
    dx=(X-ri+Lx/2)%Lx-Lx/2; dy=(Y-rj+Ly/2)%Ly-Ly/2; msk=(dx**2+dy**2)<8.0**2
    win.append((msk,dx[msk],dy[msk]))
for n,(msk,dx,dy) in enumerate(win):           # refine reference to the static centroid
    w=q0[msk]; R[n]+=[(w*dx).sum()/w.sum(),(w*dy).sum()/w.sum()]
    win[n]=(msk,dx-(w*dx).sum()/w.sum(),dy-(w*dy).sum()/w.sum())
kq=2*np.pi/Lx; rng=np.random.default_rng(seed); sig=np.sqrt(2*a*T/dt)
nsamp=10; t=0.0; traj=[]; t0=time.time(); step=0
while t<tmax:
    xi=sig*rng.standard_normal(S.shape)
    k1=f(S,xi); S1=S+dt*k1; S1/=np.linalg.norm(S1,axis=-1)[...,None]
    S=S+dt/2*(k1+f(S1,xi)); S/=np.linalg.norm(S,axis=-1)[...,None]
    t+=dt; step+=1
    if step%int(round(nsamp/dt))==0 and t>2000:
        q=qdens(S); U=np.array([[(q[m]*dx).sum()/q[m].sum(),(q[m]*dy).sum()/q[m].sum()] for (m,dx,dy) in win])
        traj.append(np.concatenate([[t],U.ravel()]))
    if step%50000==0: np.save(f"big_traj_u{ue}_T{T}_s{seed}.npy",np.array(traj)); np.save("big_Rref.npy",R); print(f"t={t:.0f} wall={time.time()-t0:.0f}s",flush=True)
np.save(f"big_traj_u{ue}_T{T}_s{seed}.npy",np.array(traj)); np.save("big_Rref.npy",R); print("done",time.time()-t0)
