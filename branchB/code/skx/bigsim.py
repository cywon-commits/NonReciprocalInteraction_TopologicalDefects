from skx import *
import sys, time, os
# Large lab-frame run: nx x ny copies of the 19x33 two-skyrmion cell (B=0.5 D^2/J), periodic.
J=1.0; D=0.4; bb=0.5; a=0.3; b=0.6
ueff=float(sys.argv[1]); T=float(sys.argv[2]); tag=sys.argv[3]; nx,ny=int(sys.argv[4]),int(sys.argv[5])
dt=float(sys.argv[6]) if len(sys.argv)>6 else 0.2
u=ueff/(b/a-1)
S1=np.load("S_B0.5.npy"); S=np.tile(S1,(nx,ny,1)); Lx,Ly=S.shape[:2]
m=Model(Lx,Ly,J,D,bb*D*D/J)
rng=np.random.default_rng(7); S=S+1e-2*rng.standard_normal(S.shape); S/=np.linalg.norm(S,axis=-1)[...,None]
w1=np.array([-u,0.]); w2=np.array([b*u,0.])
os.makedirs(f"snap_{tag}",exist_ok=True)
t=0.0; every=250; t0=time.time()
np.save(f"snap_{tag}/S_{0:06d}.npy",S.astype(np.float32))
while t<T-1e-9:
    S=rk4(m,S,w1,w2,a,dt,int(round(every/dt))); t+=every
    np.save(f"snap_{tag}/S_{int(t):06d}.npy",S.astype(np.float32))
    print(f"t={t:.0f} Q={GD(S)[2]:.3f} wall={time.time()-t0:.0f}s",flush=True)
