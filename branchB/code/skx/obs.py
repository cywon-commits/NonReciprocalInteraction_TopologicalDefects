from skx import *
from melt import analyse
import sys, time
# T=0 lab-frame run from a given state with on-the-fly order parameters
J=1.0; D=0.4; bb=0.5; a=0.3; b=0.6
init=sys.argv[1]; ueff=float(sys.argv[2]); tmax=float(sys.argv[3]); every=float(sys.argv[4]); tag=sys.argv[5]; dt=0.2
u=ueff/(b/a-1); S=np.load(init).astype(float); S/=np.linalg.norm(S,axis=-1)[...,None]; Lx,Ly=S.shape[:2]
m=Model(Lx,Ly,J,D,bb*D*D/J); w1=np.array([-u,0.]); w2=np.array([b*u,0.])
f=open(f"obs_{tag}.txt","w"); t=0.0; t0=time.time()
def rec(S,t):
    A=analyse(S); f.write(f"{t:.0f} {A['N']} {A['Psi6']:.4f} {A['psi6loc']:.4f} {A['fdef']:.4f} {A['Smax_overN']:.4f} {GD(S)[2]:.3f}\n"); f.flush()
rec(S,t)
while t<tmax-1e-9:
    S=rk4(m,S,w1,w2,a,dt,int(round(every/dt))); t+=every; rec(S,t)
    if int(t)%2000==0: np.save(f"obs_{tag}_last.npy",S.astype(np.float32)); print(f"t={t:.0f} wall={time.time()-t0:.0f}",flush=True)
np.save(f"obs_{tag}_last.npy",S.astype(np.float32)); print("done",flush=True)
