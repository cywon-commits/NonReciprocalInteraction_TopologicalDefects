from skx import *
import sys
J=1.0; D=0.4; Lx,Ly=19,33; B=0.5*D*D/J
m=Model(Lx,Ly,J,D,B); S0=np.load("S0.npy")
a=float(sys.argv[1]); b=float(sys.argv[2]); kdir=sys.argv[3]; nk=int(sys.argv[4])
us=[float(x) for x in sys.argv[5:]]
kmag=2*np.pi/(Lx*nk)
k=(kmag,0.) if kdir=='x' else (0.,kmag)
for u in us:
    uv=np.array([u,0.])
    S,V,w1,w2,r=find_V(m,S0.copy(),uv,a,b)
    lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k))
    small=lam[np.argsort(np.abs(lam))[:2]]
    ueff=(b/a-1)*u
    print(f"a={a} b={b} k||{kdir}={kmag:.5f} u={u:.4f} u_eff={ueff:.4f}  V=({V[0]:.4f},{V[1]:.4f}) resid={r:.1e}  phonon lam={np.round(small,7)}  maxRe(all)={lam.real.max():.2e}",flush=True)
