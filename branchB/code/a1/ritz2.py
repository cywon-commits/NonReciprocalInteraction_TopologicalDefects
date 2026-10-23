from ritz import *
# Ritz with a smooth cell-periodic displacement envelope: Psi = e^{ikx}[phi + i k sum_pq c_pq F_pq(r) phi] (+ optional local channels with smooth window)
def ritz_env(d,P,with_channels=False):
    kx,X,Y,gx,gy,to_t=d['kx'],d['X'],None,d['gx'],d['gy'],d['to_t']; Lx,Ly=d['Lx'],d['Ly']
    Xg=d['X']; Yg=np.ones((Lx,1))*np.arange(Ly)[None,:]
    paff=np.tile(np.exp(1j*kx*Xg).reshape(-1),2)
    F=[np.exp(2j*np.pi*(p*Xg/Lx+q*Yg/Ly)) for p in range(-P,P+1) for q in range(-2*P,2*P+1) if (p,q)!=(0,0)]
    win=np.exp(-(d['r']/(1.3*d['R']))**4)[...,None]
    Psi={}
    for pol,phi,chans in (('L',gx,[d['xl'][...,None]*gx+d['yl'][...,None]*gy, d['xl'][...,None]*gx-d['yl'][...,None]*gy]),
                          ('T',gy,[d['xl'][...,None]*gy-d['yl'][...,None]*gx, d['xl'][...,None]*gy+d['yl'][...,None]*gx])):
        base=paff*to_t(phi)
        Dl=[paff*to_t(f[...,None]*phi)*(1j*kx) for f in F]
        if with_channels: Dl+=[paff*to_t(v*win)*(1j*kx/2) for v in chans]
        Dm=np.stack(Dl,1); K=Dm.conj().T@d['H']@Dm; Cv=Dm.conj().T@d['H']@base
        s=np.linalg.lstsq(K,-Cv,rcond=1e-12)[0]; Psi[pol]=base+Dm@s
    PL,PT=Psi['L'],Psi['T']; Om,O=d['Om'],d['O']
    return float(abs(((PL.conj()@Om@O@PT)/(PL.conj()@Om@PT)).imag)/kx), len(F)
if __name__=="__main__":
    g=json.load(open("grid_static.json"))
    for key in ["19_0.3","19_0.5","19_0.85","13_0.5","15_0.5","23_0.5","23_0.7","27_0.6"]:
        Lx,bb=key.split('_'); d=setup(np.load(f"Sgrid_{Lx}_{bb}.npy"),float(bb))
        res=[ritz_env(d,P) for P in (1,2,3)]; rc=ritz_env(d,2,True)
        print(f"a={Lx} B={bb}: exact={g[key]['kLT']:.3f} | envelope P=1,2,3 (n={[r[1] for r in res]}): {[round(r[0],3) for r in res]} | P=2 + local channels: {rc[0]:.3f}",flush=True)
