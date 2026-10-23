from skx import *
import json, sys
# Ritz theory of the internal-strain mode: Psi = Psi_rigid + sum_ch s_ch Delta_ch, s = -K^{-1} C  (static energy minimisation),
# channels = local affine deformations of each skyrmion (helicity/ellipse for T, breathing/ellipse for L), split core/shell.
J,D=1.0,0.4; z=np.zeros(2)
def setup(S,bb,nk=40):
    Lx,Ly=S.shape[:2]; m=Model(Lx,Ly,J,D,bb*D*D/J); N=Lx*Ly; kx=2*np.pi/(Lx*nk); k=(kx,0.)
    Jm=bloch_jac(m,S,z,z,0.0,k); e=1e-4; O=(bloch_jac(m,S,np.array([e,0.]),z,0.0,k)-Jm)/e
    Om=np.block([[np.zeros((N,N)),np.eye(N)],[-np.eye(N),np.zeros((N,N))]]); H=-Om@Jm; H=(H+H.conj().T)/2
    e1,e2=basis(S); E1,E2=e1.reshape(N,3),e2.reshape(N,3); to_t=lambda v: np.concatenate([np.sum(v.reshape(N,3)*E1,1),np.sum(v.reshape(N,3)*E2,1)])
    gx=((np.roll(S,-1,0)-np.roll(S,1,0))/2); gy=((np.roll(S,-1,1)-np.roll(S,1,1))/2)
    X=np.arange(Lx)[:,None]*np.ones((1,Ly)); Y=np.ones((Lx,1))*np.arange(Ly)[None,:]
    mz=S[...,2]; C=[]
    for i in range(Lx):
        for j in range(Ly):
            v=mz[i,j]
            if v<0 and all(v<=mz[(i+di)%Lx,(j+dj)%Ly] for di in (-1,0,1) for dj in (-1,0,1)): C.append((i,j))
    best=None
    for cx,cy in sorted(set(C)):
        dx=(X-cx+Lx/2)%Lx-Lx/2; dy=(Y-cy+Ly/2)%Ly-Ly/2; r2=dx**2+dy**2
        best=(r2,dx,dy) if best is None else (np.where(r2<best[0],r2,best[0]),np.where(r2<best[0],dx,best[1]),np.where(r2<best[0],dy,best[2]))
    r2,xl,yl=best
    Rm=np.sqrt(np.sum(mz<0)/(2*np.pi))
    return dict(Lx=Lx,Ly=Ly,N=N,kx=kx,Jm=Jm,O=O,Om=Om,H=H,to_t=to_t,gx=gx,gy=gy,X=X,xl=xl,yl=yl,r=np.sqrt(r2),R=Rm)
def ritz(d,shell_split=True,Rfac=1.3):
    kx,X,xl,yl,gx,gy,to_t=d['kx'],d['X'],d['xl'],d['yl'],d['gx'],d['gy'],d['to_t']
    prig=np.tile(np.exp(1j*kx*(X-xl)).reshape(-1),2); paff=np.tile(np.exp(1j*kx*X).reshape(-1),2)
    core=(d['r']<Rfac*d['R'])[...,None]*1.0; masks=[core,1-core] if shell_split else [np.ones_like(core)]
    modes={'T':{'helicity':xl[...,None]*gy-yl[...,None]*gx,'ellipse':xl[...,None]*gy+yl[...,None]*gx,'phi':gy},
           'L':{'breathing':xl[...,None]*gx+yl[...,None]*gy,'ellipse':xl[...,None]*gx-yl[...,None]*gy,'phi':gx}}
    out={}; Psi={}
    for pol in ('L','T'):
        base=prig*to_t(modes[pol]['phi']); Delta=[]; names=[]
        for nm,v in modes[pol].items():
            if nm=='phi': continue
            for mi,msk in enumerate(masks):
                Delta.append(prig*to_t(v*msk)*(1j*kx/2)); names.append(nm+('-core' if mi==0 else '-shell') if shell_split else nm)
        Dm=np.stack(Delta,1); K=Dm.conj().T@d['H']@Dm; Cv=Dm.conj().T@d['H']@base
        s=np.linalg.solve(K,-Cv); Psi[pol]=base+Dm@s; out[pol]={n:float(v.real) for n,v in zip(names,s)}
        out[pol+'_aff_check']=float(((paff*to_t(modes[pol]['phi'])).conj()@d['H']@(paff*to_t(modes[pol]['phi']))).real)
    PL,PT=Psi['L'],Psi['T']; Om,O=d['Om'],d['O']
    out['kappa_LT_ritz']=float(abs(((PL.conj()@Om@O@PT)/(PL.conj()@Om@PT)).imag)/kx)
    rig_L=prig*to_t(gx); rig_T=prig*to_t(gy)
    out['kappa_rigid']=float(abs(((rig_L.conj()@Om@O@rig_T)/(rig_L.conj()@Om@rig_T)).imag)/kx)
    return out
if __name__=="__main__":
    g=json.load(open("grid_static.json")); rows={}
    for key in g:
        Lx,bb=key.split('_'); d=setup(np.load(f"Sgrid_{Lx}_{bb}.npy"),float(bb))
        r1=ritz(d,False); r2=ritz(d,True)
        rows[key]=dict(exact=g[key]['kLT'],rigid=r1['kappa_rigid'],ritz_local=r1['kappa_LT_ritz'],ritz_coreshell=r2['kappa_LT_ritz'],s_local=r1,s_cs=r2,R=d['R'])
        print(f"a={Lx} B={bb}: kappa_LT exact={g[key]['kLT']:.3f} | rigid={r1['kappa_rigid']:.3f} | Ritz(4 local ch.)={r1['kappa_LT_ritz']:.3f} | Ritz(core/shell, 8 ch.)={r2['kappa_LT_ritz']:.3f} | s_T: hel={r1['T']['helicity']:.2f} ell={r1['T']['ellipse']:.2f}",flush=True)
        json.dump(rows,open("ritz.json","w"))
