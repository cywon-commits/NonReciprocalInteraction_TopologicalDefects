from helix import *
J=1.0
for N,n in [(200,3),(400,6),(800,12)]:
    th=2*np.pi*n/N; D=J*np.tan(th); a,b=0.1,0.3
    kmin=2*np.pi/N
    def g(ueff):
        u=ueff/(b/a-1); M,Om=predict(J,D,th,u,a,b); S,_=cone(N,n,M)
        Jm=jac(S,(J,D,u,a,b,Om)); lam,V=np.linalg.eig(Jm); i=np.argmax(lam.real)
        return lam[i].real,V[:,i]
    lo,hi=0.6*D,0.9*D
    for _ in range(30):
        mid=(lo+hi)/2
        if g(mid)[0]>1e-10: hi=mid
        else: lo=mid
    t=(lo+hi)/2
    # wavenumber of the unstable mode
    _,v=g(t*1.02); amp=np.fft.fft(v[:N]); k=np.argmax(np.abs(amp[1:N//2]))+1
    print(f"N={N}: u_eff*={t:.5f}  finite-k continuum sqrt((D^2+k^2)/2)={np.sqrt((D*D+kmin**2)/2):.5f}  k->0 limit D/sqrt2={D/np.sqrt(2):.5f}  unstable mode index={k}")
