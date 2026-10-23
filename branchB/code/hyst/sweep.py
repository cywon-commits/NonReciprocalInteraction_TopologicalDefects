import numpy as np, sys, json, time
from fastllg import Integrator
# T=0 lab-frame runs at a sequence of u_eff; records sideband amplitude A, drift V (least squares dS/dt = -(V.grad)S), topological charge.
# usage: python sweep.py system tag hold "u1,u2,..." [start.npy]
sysname,tag,hold,us=sys.argv[1],sys.argv[2],float(sys.argv[3]),[float(x) for x in sys.argv[4].split(',')]
a,b,J=0.3,0.6,1.0
if sysname=='skx':
    D=0.4; B=0.5*D*D; nx=int(sys.argv[6]) if len(sys.argv)>6 else 4
    S=np.tile(np.load("S_B0.5.npy"),(nx,1,1)); comp=2; Gs=[(nx*g0//4*1,g1) for g0,g1 in [(0,2),(4,32),(-4,32),(0,31),(-4,1),(4,1)]]
    Gs=[((g0*nx)//4,g1) for g0,g1 in [(0,2),(4,32),(-4,32),(0,31),(-4,1),(4,1)]]
else:
    N,n=200,3; D=J*np.tan(2*np.pi*n/N); B=0.0; i=np.arange(N); th=2*np.pi*n/N*i
    S=np.zeros((N,1,3)); S[:,0,0]=np.sin(th); S[:,0,2]=np.cos(th)
if len(sys.argv)>5 and sys.argv[5]!='-': S=np.load(sys.argv[5]).astype(float)
else:
    S=S+1e-3*np.random.default_rng(3).standard_normal(S.shape)
S/=np.linalg.norm(S,axis=-1)[...,None]
Lx,Ly=S.shape[:2]
def grad(S):
    return (np.roll(S,-1,0)-np.roll(S,1,0))/2,(np.roll(S,-1,1)-np.roll(S,1,1))/2
def amp(S):
    if sysname=='skx':
        F=np.fft.fft2(S[...,2]); P=abs(F)**2; Pb=sum(P[g0%Lx,g1%Ly] for g0,g1 in Gs)
        onRL=np.broadcast_to((np.arange(Lx)%(Lx//19)==0)[:,None],P.shape)
        side=P[(~onRL)].sum()
        return np.sqrt(side/Pb)
    else:
        c=S[:,0,2]+1j*S[:,0,0]; F=np.fft.fft(c); P=abs(F)**2
        return np.sqrt((P.sum()-P[n])/P[n])
def topo(S):
    if sysname!='skx':
        ph=np.arctan2(S[:,0,0],S[:,0,2]); d=np.angle(np.exp(1j*np.diff(np.r_[ph,ph[0]])));return float(d.sum()/(2*np.pi))
    gx,gy=grad(S); return float(np.sum(S*np.cross(gx,gy))/(4*np.pi))
out=[]; t0=time.time()
for u_eff in us:
    u=u_eff/(b/a-1)
    S=S+1e-6*np.random.default_rng(int(u_eff*1e4)).standard_normal(S.shape); S/=np.linalg.norm(S,axis=-1)[...,None]
    I=Integrator(S,J,D,B,a,np.array([-u,0.]),np.array([b*u,0.]),0.0,0.2)
    t=0.0; rec=[]
    while t<hold-1e-9:
        S0=I.S.copy(); I.run(5); dSdt=(I.S-S0)/1.0
        gx,gy=grad(I.S); M=np.array([[np.sum(gx*gx),np.sum(gx*gy)],[np.sum(gx*gy),np.sum(gy*gy)]])+1e-12*np.eye(2)
        V=-np.linalg.solve(M,[np.sum(dSdt*gx),np.sum(dSdt*gy)])
        I.run(245); t+=50
        rec.append([t,amp(I.S),V[0],V[1],topo(I.S),float(I.S[...,1].mean())])
    rec=np.array(rec); S=I.S.copy()
    np.save(f"rec_{tag}_{u_eff:.4f}.npy",rec); np.save(f"S_{tag}_{u_eff:.4f}.npy",S)
    last=rec[len(rec)*3//4:]
    print(f"u_eff={u_eff:.4f} A_end={rec[-1,1]:.3e} A_mean_lastQ={last[:,1].mean():.3e} A_max={rec[:,1].max():.3e} Vx={last[:,2].mean():.4f} Vy={last[:,3].mean():.4f} Q={rec[-1,4]:.3f} My={last[:,5].mean():.4f} wall={time.time()-t0:.0f}",flush=True)
