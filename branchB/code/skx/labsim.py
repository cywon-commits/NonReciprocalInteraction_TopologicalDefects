from skx import *
import sys, time
# Direct real-space time integration in the LAB frame (no co-moving frame, no Bloch reduction).
# Supercell of 4x1 magnetic unit cells (76x33, 8 skyrmions), periodic. Seed with small random noise.
J=1.0; D=0.4; bb=0.5; a=0.3; b=0.6
ueff=float(sys.argv[1]); u=ueff/(b/a-1); T=float(sys.argv[2]); tag=sys.argv[3]
m=Model(76,33,J,D,bb*D*D/J)
S1=np.load(f"S_B{bb}.npy"); S=np.concatenate([S1]*4,axis=0)
rng=np.random.default_rng(1); S=S+1e-3*rng.standard_normal(S.shape); S/=np.linalg.norm(S,axis=-1)[...,None]
w1=np.array([-u,0.]); w2=np.array([b*u,0.])
Gs=[(0,2),(4,32),(72,32),(0,31),(72,1),(4,1)]
def amps(S):
    F=np.fft.fft2(S[...,2]); out=[]
    for d in (1,2):
        num=sum(abs(F[(gx+s*d)%76,gy])**2 for gx,gy in Gs for s in (1,-1)); den=sum(abs(F[gx,gy])**2 for gx,gy in Gs)
        out.append(np.sqrt(num/den))
    return out
dt=0.1; nrec=int(50/dt); t=0.0; f=open(f"lab_{tag}.txt","w"); t0=time.time()
while t<T:
    S=rk4(m,S,w1,w2,a,dt,nrec); t+=50
    A1,A2=amps(S); Q=GD(S)[2]
    f.write(f"{t:.0f} {A1:.4e} {A2:.4e} {Q:.4f}\n"); f.flush()
f.close(); np.save(f"Sfin_{tag}.npy",S); print("done",tag,time.time()-t0)
