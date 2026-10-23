import numpy as np, glob, sys
from fastllg import Integrator
from orderpar import order
J,D=1.0,0.4; B=0.5*D*D; a=0.3; b=0.6
tag,ue,dur,stride=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]); u=ue/(b/a-1)
rows=[]
for fn in sorted(glob.glob(f"snap_{tag}/S_*.npy"))[::stride]:
    S0=np.load(fn).astype(float); S0/=np.linalg.norm(S0,axis=-1)[...,None]
    _,P0,_,f0,_=order(S0); res=[]
    for uu in (u,0.0):
        I=Integrator(S0.copy(),J,D,B,a,np.array([-uu,0.]),np.array([b*uu,0.]),0.0,0.2); I.run(int(dur/0.2)); _,P,_,f,_=order(I.S); res+= [P,f]
    rows.append([P0,f0,*res]); print(f"{fn[-10:-4]}: Psi6 {P0:.2f} -> on {res[0]:.2f} / off {res[2]:.2f} ;  f_def {f0:.2f} -> on {res[1]:.2f} / off {res[3]:.2f}",flush=True)
np.save(f"resp_{tag}.npy",np.array(rows))
