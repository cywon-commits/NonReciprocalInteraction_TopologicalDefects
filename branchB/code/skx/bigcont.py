from skx import *
import sys, glob, time
# continue a large run from its last snapshot
J=1.0; D=0.4; bb=0.5; a=0.3; b=0.6
ueff=float(sys.argv[1]); T=float(sys.argv[2]); tag=sys.argv[3]; dt=0.2
u=ueff/(b/a-1); fn=sorted(glob.glob(f"snap_{tag}/S_*.npy"))[-1]; t=float(fn.split('_')[-1][:-4])
S=np.load(fn).astype(float); S/=np.linalg.norm(S,axis=-1)[...,None]; Lx,Ly=S.shape[:2]
m=Model(Lx,Ly,J,D,bb*D*D/J); w1=np.array([-u,0.]); w2=np.array([b*u,0.]); every=250; t0=time.time()
while t<T-1e-9:
    S=rk4(m,S,w1,w2,a,dt,int(round(every/dt))); t+=every
    np.save(f"snap_{tag}/S_{int(t):06d}.npy",S.astype(np.float32))
    print(f"t={t:.0f} Q={GD(S)[2]:.3f} wall={time.time()-t0:.0f}s",flush=True)
