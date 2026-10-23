import numpy as np, glob, sys
from fastllg import rhs
from orderpar import order
from skx import zh, dx_, dy_
J,D=1.0,0.4; B=0.5*D*D; a=0.3; b=0.6
def field(S):
    xp=np.roll(S,-1,0); xm=np.roll(S,1,0); yp=np.roll(S,-1,1); ym=np.roll(S,1,1)
    return J*(xp+xm+yp+ym)+B*zh+D*(np.cross(xp,dx_)+np.cross(dx_,xm)+np.cross(yp,dy_)+np.cross(dy_,ym))
def pump(S,ue):
    u=ue/(b/a-1); z=np.zeros_like(S); fon=np.empty_like(S); foff=np.empty_like(S)
    rhs(S,z,fon,J,D,B,a,-u,0.0,b*u,0.0); rhs(S,z,foff,J,D,B,a,0.,0.,0.,0.)
    H=field(S); return -np.sum(H*(fon-foff)), np.sum(H*foff)   # net current power, dissipation rate (= -dE/dt|off)
if __name__=="__main__":
    tag,ue=sys.argv[1],float(sys.argv[2]); rows=[]
    for fn in sorted(glob.glob(f"snap_{tag}/S_*.npy")):
        S=np.load(fn).astype(float); S/=np.linalg.norm(S,axis=-1)[...,None]
        P,Dd=pump(S,ue); N,P6,p6l,fd,_=order(S); rows.append([int(fn[-10:-4]),P/N,Dd/N,P6,p6l,fd])
    rows=np.array(rows); np.save(f"pump_{tag}.npy",rows)
    c=np.corrcoef(rows[:,1],rows[:,3])[0,1]; c2=np.corrcoef(rows[:,1],rows[:,5])[0,1]
    print(f"{tag}: {len(rows)} snapshots; P_net per skyrmion mean {rows[:,1].mean():+.2e}; corr(P_net,|Psi6|)={c:+.2f}; corr(P_net,f_def)={c2:+.2f}")
    for lo,hi in [(0,0.2),(0.2,0.35),(0.35,0.5),(0.5,1)]:
        m=(rows[:,3]>=lo)&(rows[:,3]<hi)
        if m.sum(): print(f"   |Psi6| in [{lo},{hi}): n={m.sum():3d}  P_net={rows[m,1].mean():+.2e} ± {rows[m,1].std()/np.sqrt(m.sum()):.1e}   dissipation(off)={rows[m,2].mean():.2e}")
