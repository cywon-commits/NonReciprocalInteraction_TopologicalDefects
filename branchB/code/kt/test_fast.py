import numpy as np, time
from skx import Model
from fastllg import *
J,D=1.0,0.4; B=0.5*D*D; a=0.3; w1=np.array([0.01,-0.02]); w2=np.array([0.03,0.005])
S=np.tile(np.load("S_B0.5.npy"),(8,4,1)); S=S+0.05*np.random.default_rng(1).standard_normal(S.shape); S/=np.linalg.norm(S,axis=-1)[...,None]
m=Model(*S.shape[:2],J,D,B); ref=m.rhs(S,w1,w2,a); out=np.empty_like(S)
rhs(S,np.zeros_like(S),out,J,D,B,a,w1[0],w1[1],w2[0],w2[1]); print("max diff vs numpy rhs:",np.abs(out-ref).max())
I=Integrator(S,J,D,B,a,w1,w2,0.01,0.15); I.run(10)
t=time.time(); I.run(2000); el=time.time()-t
print(f"{S.shape[0]}x{S.shape[1]} sites: {el/2000*1e3:.2f} ms/step -> {el/2000/0.15:.4f} s per time unit")
