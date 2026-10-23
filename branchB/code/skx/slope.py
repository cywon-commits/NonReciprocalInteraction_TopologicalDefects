from skx import *
J=1.0; D=0.4; Lx,Ly=19,33; B=0.5*D*D/J
m=Model(Lx,Ly,J,D,B); S0=np.load("S0.npy")
def phon(a,b,u,k):
    uv=np.array([u,0.])
    if u==0: S,w1,w2=S0,np.zeros(2),np.zeros(2)
    else: S,V,w1,w2,r=find_V(m,S0.copy(),uv,a,b)
    lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k))
    p=lam[np.argsort(np.abs(lam))[:2]]
    return p, w1
du=0.004
for (a,b,kd,nk) in [(0.3,0.6,'x',40),(0.3,0.6,'x',20),(0.3,0.6,'x',10),
                    (0.3,0.6,'y',20),(0.1,0.2,'x',20),(0.3,0.0,'x',20),(0.3,0.9,'x',20)]:
    kmag=2*np.pi/(Lx*nk); k=(kmag,0.) if kd=='x' else (0.,kmag)
    p0,_=phon(a,b,0,k); p1,w1=phon(a,b,du,k)
    G0=-p0.real.max(); w0=abs(p0.imag).max()
    ueff=(b/a-1)*du
    sl=(p1.real.max()-p0.real.max())/ueff          # d(maxRe)/d(u_eff)
    ustar=G0/sl if sl!=0 else np.inf
    print(f"a={a} b={b} k||{kd} k={kmag:.5f} | omega0={w0:.3e} Gamma0={G0:.3e} | slope/u_eff={sl:.3e} -> u_eff*~{ustar:.4f} | affine-Landau omega0/k={w0/kmag:.4f} | ratio={(w0/kmag)/ustar if np.isfinite(ustar) else 0:.3f} | w1/du=({w1[0]/du:.3f},{w1[1]/du:.3f})",flush=True)
