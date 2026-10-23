from skx import *
import sys, json
# Decompose the internal-strain corrector of the relaxed modes into local affine-generated deformation channels of each
# skyrmion: helicity h=(x d_y - y d_x)m, breathing b=(x d_x + y d_y)m, ellipses e1=(x d_x - y d_y)m, e2=(x d_y + y d_x)m.
# Affinity s_ch = (amplitude in relaxed mode)/(affine amplitude): 1 = follows the lattice strain, 0 = rigid.
J,D=1.0,0.4; z=np.zeros(2)
def run(S,bb,nk=40):
    Lx,Ly=S.shape[:2]; m=Model(Lx,Ly,J,D,bb*D*D/J); N=Lx*Ly; kx=2*np.pi/(Lx*nk); k=(kx,0.)
    Jm=bloch_jac(m,S,z,z,0.0,k); Om=np.block([[np.zeros((N,N)),np.eye(N)],[-np.eye(N),np.zeros((N,N))]]); H=-Om@Jm; H=(H+H.conj().T)/2
    e1,e2=basis(S); E1,E2=e1.reshape(N,3),e2.reshape(N,3)
    gx=((np.roll(S,-1,0)-np.roll(S,1,0))/2); gy=((np.roll(S,-1,1)-np.roll(S,1,1))/2)
    X=np.arange(Lx)[:,None]*np.ones((1,Ly)); Y=np.ones((Lx,1))*np.arange(Ly)[None,:]; ph=np.exp(1j*kx*X).reshape(N)
    to_t=lambda v: np.concatenate([np.sum(v.reshape(N,3)*E1,1),np.sum(v.reshape(N,3)*E2,1)])
    Phi=np.stack([to_t(gx)*np.tile(ph,2),to_t(gy)*np.tile(ph,2)],1)
    M=Phi.conj().T@Om@Phi; W=Om.T@Phi@np.linalg.inv(M).conj().T; Pr=np.linalg.solve(H,W); Pr=Pr@np.linalg.inv(W.conj().T@Pr)
    # skyrmion centres (m_z minima, refined) and local coordinates (minimum image to nearest centre)
    mz=S[...,2]; C=[]
    for i in range(Lx):
        for j in range(Ly):
            v=mz[i,j]
            if v<0 and all(v<=mz[(i+di)%Lx,(j+dj)%Ly] for di in (-1,0,1) for dj in (-1,0,1)): C.append((i,j))
    C=np.array(sorted(set(C)),float)
    best=None
    for cx,cy in C:
        dx=(X-cx+Lx/2)%Lx-Lx/2; dy=(Y-cy+Ly/2)%Ly-Ly/2; r2=dx**2+dy**2
        best=(r2,dx,dy) if best is None else (np.where(r2<best[0],r2,best[0]),np.where(r2<best[0],dx,best[1]),np.where(r2<best[0],dy,best[2]))
    _,xl,yl=best
    modes={'helicity':xl[...,None]*gy-yl[...,None]*gx,'breathing':xl[...,None]*gx+yl[...,None]*gy,
           'ellipse1':xl[...,None]*gx-yl[...,None]*gy,'ellipse2':xl[...,None]*gy+yl[...,None]*gx}
    B=np.stack([to_t(v) for v in modes.values()]+[to_t(gx),to_t(gy)],1).astype(complex)
    out={}
    for col,name,aff in ((1,'T',{'helicity':-0.5,'ellipse2':-0.5}),(0,'L',{'breathing':-0.5,'ellipse1':-0.5})):
        cell=np.conj(np.tile(ph,2))*Pr[:,col]                       # cell-periodic part phi + i k chi
        base=Phi[:,col]*np.conj(np.tile(ph,2))
        chi=(cell-base)/(1j*kx)
        coef,res,*_=np.linalg.lstsq(B,chi,rcond=None)
        fit_quality=1-np.linalg.norm(B@coef-chi)**2/np.linalg.norm(chi)**2
        # relaxed local deformation amplitude = affine (-1/2 per channel, from -x d_y m etc.) + corrector; affinity = ratio
        for nm,a in aff.items():
            i=list(modes).index(nm); out[f"{name}:{nm}"]=float(((a+coef[i])/a).real)
        out[f"{name}:fit"]=float(fit_quality)
    return out
if __name__=="__main__":
    g=json.load(open("grid_static.json"))
    for key in ["19_0.3","19_0.5","19_0.7","19_0.85","13_0.5","15_0.5","23_0.5","23_0.7","27_0.6"]:
        Lx,bb=key.split('_'); r=run(np.load(f"Sgrid_{Lx}_{bb}.npy"),float(bb))
        print(f"a={Lx} B={bb}: kappa={g[key]['kt']:.3f} | T: s_hel={r['T:helicity']:+.3f} s_ell2={r['T:ellipse2']:+.3f} (fit {r['T:fit']:.2f}) | L: s_br={r['L:breathing']:+.3f} s_ell1={r['L:ellipse1']:+.3f} (fit {r['L:fit']:.2f})",flush=True)
