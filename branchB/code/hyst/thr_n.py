from helix import *
J=1.0; N=200; D=J*np.tan(2*np.pi*3/N)
for (a,b) in [(0.3,0.6),(0.1,0.3)]:
  for n in (3,2,1):
    th=2*np.pi*n/N
    G=J*np.cos(th)+D*np.sin(th); uunw=2*(G-J)/np.sin(th)
    def g(ueff):
        u=ueff/(b/a-1); M,Om=predict(J,D,th,u,a,b)
        if abs(M)>=1: return 1.0
        S,_=cone(N,n,M); lam=np.linalg.eigvals(jac(S,(J,D,u,a,b,Om))); return lam.real.max()
    lo,hi=0.3*uunw,0.999*uunw
    if g(hi)<=1e-10: print(a,b,n,"stable up to unwinding",uunw); continue
    for _ in range(25):
        mid=(lo+hi)/2
        if g(mid)>1e-10: hi=mid
        else: lo=mid
    print(f"a={a} b={b} n={n}: u_L={hi:.4f}  u_unw={uunw:.4f}  ratio={hi/uunw:.3f}  M_at_uL={predict(J,D,th,hi/(b/a-1),a,b)[0]:.3f}",flush=True)
