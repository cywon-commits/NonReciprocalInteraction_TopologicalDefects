import numpy as np, sys, os, time
from fastllg import Integrator
from orderpar import energy, order, sideband
J,D=1.0,0.4; B=0.5*D*D; a=0.3; b=0.6
ue=float(sys.argv[1]); dur=float(sys.argv[2]); tag=sys.argv[3]; start=sys.argv[4]; snapevery=float(sys.argv[5])
u=ue/(b/a-1); w1=np.array([-u,0.]); w2=np.array([b*u,0.])
S=np.load(start).astype(float); S/=np.linalg.norm(S,axis=-1)[...,None]
I=Integrator(S,J,D,B,a,w1,w2,0.0,0.2); os.makedirs(f"snap_{tag}",exist_ok=True)
Gs=[(0,10),(10,-5),(-10,-5)]; rec=[]; t=0.0; every=25; t0=time.time()
while t<dur-1e-9:
    I.run(int(round(every/0.2))); t+=every
    N,P6,p6l,fd,_=order(I.S); rec.append([t,N,P6,p6l,fd,*sideband(I.S,Gs),energy(I.S)])
    if abs(t/snapevery-round(t/snapevery))<1e-9: np.save(f"snap_{tag}/S_{int(t):06d}.npy",I.S.astype(np.float32))
np.save(f"ts_{tag}.npy",np.array(rec)); print("done",time.time()-t0,flush=True)
