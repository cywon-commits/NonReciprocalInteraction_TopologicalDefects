from ritz import *
# Two-field basis on Fourier envelopes; compare the full Ritz solution with block/diagonal approximations -> which structure yields a series formula?
def build(d,P):
    kx,gx,gy,to_t=d['kx'],d['gx'],d['gy'],d['to_t']; L_,Ly=d['Lx'],d['Ly']; Xg=d['X']; Yg=np.ones((L_,1))*np.arange(Ly)[None,:]
    paff=np.tile(np.exp(1j*kx*Xg).reshape(-1),2); Om=d['Om']
    pq=[(p,q) for p in range(-P,P+1) for q in range(-2*P,2*P+1)]
    F={t:np.exp(2j*np.pi*(t[0]*Xg/L_+t[1]*Ly*0+t[1]*Yg/Ly)) for t in pq}
    out={}
    for pol,phi in (('L',gx),('T',gy)):
        base=paff*to_t(phi); cols=[]; tags=[]
        for t in pq:
            for j,g in enumerate((gx,gy)):
                if not (t==(0,0) and g is phi): cols.append(paff*to_t(F[t][...,None]*g)*(1j*kx)); tags.append(('u',t,j))
            for j,g in enumerate((gx,gy)):
                cols.append(paff*(Om@to_t(F[t][...,None]*g))*(1j*kx)); tags.append(('c',t,j))
        Dm=np.stack(cols,1); out[pol]=(base,Dm,Dm.conj().T@d['H']@Dm,Dm.conj().T@d['H']@base,tags)
    return out
def kappa_from(d,B,solver):
    Psi={}
    for pol,(base,Dm,K,C,tags) in B.items(): Psi[pol]=base+Dm@solver(K,C,tags)
    PL,PT=Psi['L'],Psi['T']; Om=d['Om']
    return float(abs(((PL.conj()@Om@d['O']@PT)/(PL.conj()@Om@PT)).imag)/d['kx'])
full=lambda K,C,t: np.linalg.lstsq(K,-C,rcond=1e-12)[0]
diag=lambda K,C,t: -C/np.diag(K)
def by(keyf):
    def sol(K,C,tags):
        s=np.zeros(len(C),complex); groups={}
        for i,t in enumerate(tags): groups.setdefault(keyf(t),[]).append(i)
        for idx in groups.values():
            idx=np.array(idx); s[idx]=np.linalg.lstsq(K[np.ix_(idx,idx)],-C[idx],rcond=1e-12)[0]
        return s
    return sol
per_G=by(lambda t:(abs(t[1][0]),abs(t[1][1])))             # couple u,c and +-G within a shell member (|p|,|q|)
per_field=by(lambda t:t[0])                                # u block and c block separately
g=json.load(open("grid_static.json"))
for key in ["19_0.3","19_0.5","19_0.85","13_0.5","23_0.5","23_0.7"]:
    Lx,bb=key.split('_'); d=setup(np.load(f"Sgrid_{Lx}_{bb}.npy"),float(bb)); B=build(d,2)
    print(f"{key}: exact={g[key]['kLT']:.3f} | full={kappa_from(d,B,full):.3f} | per-|G| blocks (u+c coupled)={kappa_from(d,B,per_G):.3f} | u,c blocks separate={kappa_from(d,B,per_field):.3f} | fully diagonal={kappa_from(d,B,diag):.3f}",flush=True)
