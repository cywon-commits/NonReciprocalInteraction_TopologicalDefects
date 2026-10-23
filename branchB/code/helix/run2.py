from helix import *
import sys
J=1.0; N=int(sys.argv[1]); n=int(sys.argv[2])
th=2*np.pi*n/N; D=J*np.tan(th)
def growth(ueff,a,b):
    u=ueff/(b/a-1)
    M,Om=predict(J,D,th,u,a,b)
    if abs(M)>=1: return np.nan,M
    S,_=cone(N,n,M)
    lam=np.linalg.eigvals(jac(S,(J,D,u,a,b,Om)))
    return lam.real.max(),M
def thresh(a,b):
    lo,hi=0.2*D,0.999*np.sin(th)**-1*2*(np.sqrt(J*J+D*D)-J)
    if growth(hi,a,b)[0]<1e-9: return None
    for _ in range(40):
        mid=(lo+hi)/2
        g,_=growth(mid,a,b)
        if g>1e-10: hi=mid
        else: lo=mid
    return (lo+hi)/2
uunw=2*(np.sqrt(J*J+D*D)-J)/np.sin(th)
print(f"N={N} theta={th:.4f} D={D:.5f}  continuum u_L=D/sqrt2={D/np.sqrt(2):.5f}  u_unw(lattice)={uunw:.5f}")
for a,b in [(0.05,0.15),(0.1,0.3),(0.2,0.5),(0.1,-0.1),(0.3,0.0)]:
    t=thresh(a,b)
    print(f"alpha={a} beta={b}  u_eff*={t:.5f}  ratio to D/sqrt2 = {t/(D/np.sqrt(2)):.4f}  M*={predict(J,D,th,t/(b/a-1),a,b)[0]:.4f}")
