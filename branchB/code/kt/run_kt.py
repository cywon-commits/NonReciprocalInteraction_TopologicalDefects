import numpy as np, sys, json, time
from fastllg import Integrator
from orderpar import order
# Heating ladder for an 8x4-cell SkX (64 skyrmions), co-moving frame of the driven lattice.
J,D=1.0,0.4; B=0.5*D*D; a=0.3
ue=sys.argv[1]; Ts=[float(x) for x in sys.argv[2].split(',')]; teq,tme=float(sys.argv[3]),float(sys.argv[4]); tag=sys.argv[5]
info=json.load(open("bloch_t1.json"))[ue]; w1=np.array(info["w1"]); w2=np.array(info["w2"])
S=np.tile(np.load(f"Scomov_{float(ue)}.npy"),(8,4,1)); out=[]; t0=time.time()
for i,T in enumerate(Ts):
    I=Integrator(S,J,D,B,a,w1,w2,T,0.15,100+i); I.run(int(teq/0.15)); rec=[]
    for _ in range(int(tme/100)):
        I.run(int(100/0.15)); N,P6,p6l,fd,_=order(I.S); rec.append([N,P6,p6l,fd])
    rec=np.array(rec); S=I.S.copy(); np.save(f"kt_{tag}_T{T}.npy",S.astype(np.float32)); np.save(f"ktts_{tag}_T{T}.npy",rec)
    m=rec.mean(0); e=rec.std(0)/np.sqrt(len(rec)/5)
    out.append([T,*m,*e]); np.save(f"kt_{tag}.npy",np.array(out))
    print(f"T={T}: N={m[0]:.1f} Psi6={m[1]:.3f}±{e[1]:.3f} |psi6|loc={m[2]:.3f} f_def={m[3]:.3f}±{e[3]:.3f} wall={time.time()-t0:.0f}s",flush=True)
