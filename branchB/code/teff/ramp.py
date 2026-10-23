from skx import *
from melt import analyse
import sys, json, time
# Heating ramp with Gilbert noise (Stratonovich Heun), co-moving frame of the driven SkX, 6x3 cells (114x99, 36 skyrmions)
J=1.0; D=0.4; a=0.3; b=0.6; B=0.5*D*D/J
ue=sys.argv[1]; T0,T1,tmax=float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]); tag=sys.argv[5]; seed=int(sys.argv[6]); dt=0.1
info=json.load(open("bloch_rates.json"))[ue]; w1=np.array(info["w1"]); w2=np.array(info["w2"])
S=np.tile(np.load(f"Scomov_{ue}.npy"),(6,3,1)); Lx,Ly=S.shape[:2]
def f(S,xi):
    xp=np.roll(S,-1,0); xm=np.roll(S,1,0); yp=np.roll(S,-1,1); ym=np.roll(S,1,1)
    H=J*(xp+xm+yp+ym)+B*zh+D*(np.cross(xp,dx_)+np.cross(dx_,xm)+np.cross(yp,dy_)+np.cross(dy_,ym))+xi
    gx=(xp-xm)/2; gy=(yp-ym)/2
    Tq=-np.cross(S,H)+w1[0]*gx+w1[1]*gy+np.cross(S,w2[0]*gx+w2[1]*gy); Tq-=np.sum(Tq*S,-1)[...,None]*S
    return (Tq+a*np.cross(S,Tq))/(1+a*a)
rng=np.random.default_rng(seed); t=0.0; out=open(f"ramp_{tag}.txt","w"); t0=time.time(); every=100; avg=20
while t<tmax-1e-9:
    acc=np.zeros_like(S); nacc=0
    for s in range(int(round(every/dt))):
        T=T0+(T1-T0)*t/tmax; xi=np.sqrt(2*a*T/dt)*rng.standard_normal(S.shape)
        k1=f(S,xi); S1=S+dt*k1; S1/=np.linalg.norm(S1,axis=-1)[...,None]
        S=S+dt/2*(k1+f(S1,xi)); S/=np.linalg.norm(S,axis=-1)[...,None]; t+=dt
        if s>=int(round((every-avg)/dt)): acc+=S; nacc+=1
    Sa=acc/nacc; Sa/=np.linalg.norm(Sa,axis=-1)[...,None]
    try: A=analyse(Sa); line=f"{t:.0f} {T:.4f} {A['N']} {A['Psi6']:.4f} {A['fdef']:.4f} {A['Smax_overN']:.4f} {GD(Sa)[2]:.3f}"
    except Exception as e: line=f"{t:.0f} {T:.4f} nan nan nan nan {GD(Sa)[2]:.3f}"
    out.write(line+"\n"); out.flush()
    if int(t)%2000==0: print(line, f"wall={time.time()-t0:.0f}",flush=True)
np.save(f"ramp_{tag}_last.npy",S.astype(np.float32))
