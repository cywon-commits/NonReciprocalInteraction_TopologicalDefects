import numpy as np, sys, json, time
from fastllg import Integrator
from orderpar import order
J,D=1.0,0.4; B=0.5*D*D; a=0.3
ue,T,init,dur,seed=sys.argv[1],float(sys.argv[2]),sys.argv[3],float(sys.argv[4]),int(sys.argv[5])
info=json.load(open("bloch_t1.json"))[ue]; w1=np.array(info["w1"]); w2=np.array(info["w2"])
S=np.tile(np.load(f"Scomov_{float(ue)}.npy"),(8,4,1)) if init=="cry" else np.load("kt_coarse0_T0.3.npy").astype(float)
S/=np.linalg.norm(S,axis=-1)[...,None]
I=Integrator(S,J,D,B,a,w1,w2,T,0.15,seed); rec=[]; t0=time.time()
for k in range(int(dur/100)):
    I.run(int(100/0.15)); N,P6,p6l,fd,_=order(I.S); rec.append([(k+1)*100,N,P6,p6l,fd])
rec=np.array(rec); np.save(f"kt2_u{ue}_T{T}_{init}_s{seed}.npy",rec); h=rec[len(rec)//2:]
print(f"u={ue} T={T} init={init}: second-half Psi6={h[:,2].mean():.3f} f_def={h[:,4].mean():.3f} N={h[:,1].mean():.1f} wall={time.time()-t0:.0f}s",flush=True)
