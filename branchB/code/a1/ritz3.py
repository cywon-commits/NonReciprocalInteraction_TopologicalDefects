from ritz import *
def two_field(d,P,conj=True):
    kx,gx,gy,to_t=d['kx'],d['gx'],d['gy'],d['to_t']; L_,Ly=d['Lx'],d['Ly']; Xg=d['X']; Yg=np.ones((L_,1))*np.arange(Ly)[None,:]
    paff=np.tile(np.exp(1j*kx*Xg).reshape(-1),2); Om=d['Om']
    F=[np.exp(2j*np.pi*(p*Xg/L_+q*Yg/Ly)) for p in range(-P,P+1) for q in range(-2*P,2*P+1)]
    Psi={}
    for pol,phi in (('L',gx),('T',gy)):
        base=paff*to_t(phi)
        V=[paff*to_t(f[...,None]*g) for f in F for g in (gx,gy) if not (f is F[len(F)//2] and g is phi)]
        if conj: V+= [paff*(Om@to_t(f[...,None]*g)) for f in F for g in (gx,gy)]
        Dm=np.stack(V,1)*(1j*kx); K=Dm.conj().T@d['H']@Dm; Cv=Dm.conj().T@d['H']@base
        s=np.linalg.lstsq(K,-Cv,rcond=1e-12)[0]; Psi[pol]=base+Dm@s
    PL,PT=Psi['L'],Psi['T']
    return float(abs(((PL.conj()@Om@d['O']@PT)/(PL.conj()@Om@PT)).imag)/kx)
g=json.load(open("grid_static.json"))
for key in ["19_0.3","19_0.5","19_0.85","13_0.5","15_0.5","23_0.5","23_0.7","27_0.6"]:
    Lx,bb=key.split('_'); d=setup(np.load(f"Sgrid_{Lx}_{bb}.npy"),float(bb))
    r=[two_field(d,P,False) for P in (1,2)]+[two_field(d,P,True) for P in (0,1,2)]
    print(f"{key}: exact={g[key]['kLT']:.3f} | displacement only P=1,2: {r[0]:.3f} {r[1]:.3f} | + conjugate canting P=0,1,2: {r[2]:.3f} {r[3]:.3f} {r[4]:.3f}",flush=True)
