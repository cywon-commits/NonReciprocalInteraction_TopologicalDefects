import numpy as np, sys, time
from fastllg import Integrator
from orderpar import energy, order, sideband
# T=0 driven SkX (10x5 cells, 100 skyrmions), lab frame as before; on-the-fly order parameters every 25 time units.
J,D=1.0,0.4; B=0.5*D*D; a=0.3; b=0.6
ue=float(sys.argv[1]); tmax=float(sys.argv[2]); tag=sys.argv[3]; start=sys.argv[4] if len(sys.argv)>4 else None
u=ue/(b/a-1); w1=np.array([-u,0.]); w2=np.array([b*u,0.])
if start: S=np.load(start).astype(float); t=float(sys.argv[5])
else:
    S=np.tile(np.load("S_B0.5.npy"),(10,5,1)); S+=1e-2*np.random.default_rng(7).standard_normal(S.shape); t=0.0
S/=np.linalg.norm(S,axis=-1)[...,None]
I=Integrator(S,J,D,B,a,w1,w2,0.0,0.2)
Gs=[(0,10),(10,-5),(-10,-5)]; rec=[]; t0=time.time(); every=25
while t<tmax-1e-9:
    I.run(int(round(every/0.2))); t+=every
    N,P6,p6l,fd,_=order(I.S); sb=sideband(I.S,Gs)
    rec.append([t,N,P6,p6l,fd,*sb,energy(I.S)])
    if len(rec)%200==0:
        np.save(f"ts_{tag}.npy",np.array(rec)); np.save(f"last_{tag}.npy",I.S.astype(np.float32)); print(f"t={t:.0f} wall={time.time()-t0:.0f}s P6={P6:.2f} fdef={fd:.2f}",flush=True)
np.save(f"ts_{tag}.npy",np.array(rec)); np.save(f"last_{tag}.npy",I.S.astype(np.float32)); print("done",flush=True)
