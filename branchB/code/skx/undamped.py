from skx import *
J=1.0; D=0.4; Lx,Ly=19,33; B=0.5*D*D/J
m=Model(Lx,Ly,J,D,B); S0=np.load("S0.npy"); z=np.zeros(2)
for nk in [40,20]:
    k=2*np.pi/(Lx*nk)
    lam=np.linalg.eigvals(bloch_jac(m,S0,z,z,0.0,(k,0.)))
    p=lam[np.argsort(np.abs(lam))[:2]]
    print(f"alpha=0 k={k:.5f} omega_u={abs(p.imag).max():.4e} Re={p.real.max():.1e}  c2=omega/k^2={abs(p.imag).max()/k**2:.4f}  omega/k={abs(p.imag).max()/k:.4f}")
Gm,Dm,Qn=GD(S0); print("G=",Gm[0,1]," Dxx,Dyy=",Dm[0,0],Dm[1,1]," D/G=",Dm[0,0]/abs(Gm[0,1]))
