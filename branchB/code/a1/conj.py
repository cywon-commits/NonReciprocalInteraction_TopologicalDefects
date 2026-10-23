from ritz import *
# Where does the exact corrector live?  V = displacement-type basis (smooth envelopes x phi + local affine channels);
# Omega V = the same fields rotated by 90 deg in the tangent plane (spin-conjugate 'tilt' partners).
def diag(key,P=2):
    Lx,bb=key.split('_'); S=np.load(f"Sgrid_{Lx}_{bb}.npy"); d=setup(S,float(bb))
    kx,gx,gy,to_t=d['kx'],d['gx'],d['gy'],d['to_t']; L_,Ly=d['Lx'],d['Ly']; Xg=d['X']; Yg=np.ones((L_,1))*np.arange(Ly)[None,:]; N=d['N']
    paff=np.tile(np.exp(1j*kx*Xg).reshape(-1),2)
    F=[np.exp(2j*np.pi*(p*Xg/L_+q*Yg/Ly)) for p in range(-P,P+1) for q in range(-2*P,2*P+1)]
    win=np.exp(-(d['r']/(1.3*d['R']))**4)[...,None]; xl,yl=d['xl'],d['yl']
    chans=[xl[...,None]*gy-yl[...,None]*gx, xl[...,None]*gy+yl[...,None]*gx, xl[...,None]*gx+yl[...,None]*gy, xl[...,None]*gx-yl[...,None]*gy]
    V=np.stack([to_t(f[...,None]*phi) for f in F for phi in (gx,gy)]+[to_t(v*win) for v in chans],1)   # cell-periodic fields
    Om=d['Om']; W=np.concatenate([V,Om@V],1)
    # exact relaxed T mode and its corrector
    Phi=np.stack([paff*to_t(gx),paff*to_t(gy)],1); M=Phi.conj().T@Om@Phi; Wc=Om.T@Phi@np.linalg.inv(M).conj().T
    Pr=np.linalg.solve(d['H'],Wc); Pr=Pr@np.linalg.inv(Wc.conj().T@Pr)
    out=[]
    for col in (0,1):
        chi=(np.conj(paff)*Pr[:,col]-to_t((gx,gy)[col]))/(1j*kx)
        def frac(B):
            c=np.linalg.lstsq(B,chi,rcond=1e-10)[0]; return 1-np.linalg.norm(B@c-chi)**2/np.linalg.norm(chi)**2
        out.append((frac(V),frac(Om@V),frac(W)))
    return out
g=json.load(open("grid_static.json"))
for key in ["19_0.3","19_0.5","19_0.85","13_0.5","23_0.5"]:
    o=diag(key); print(f"{key}: kappa={g[key]['kLT']:.3f} | corrector captured by  displacement-type V / conjugate OmegaV / both:  L {[round(x,2) for x in o[0]]}  T {[round(x,2) for x in o[1]]}",flush=True)
