from stripe import *
a,b=0.3,0.6
S,w1,w2,r=steady(0,0,a,b,2000); print("u=0 resid",r)
for u in [0.02,0.05]:
    S,w1,w2,r=steady(u,0,a,b)
    Mpred=(b/a-1)*u*np.sin(th)/(2*(np.sqrt(J*J+D*D)-J))
    print(f"u_x={u}: resid={r:.1e}  M_y={S[:,0,1].mean():+.5f} (std {S[:,0,1].std():.1e})  1D-chain formula {Mpred:.5f}   M_x={S[:,0,0].mean():.1e}")
# dispersions at u=0
S,w1,w2,r=steady(0,0,a,b,10)
for ky in [0.01,0.02,0.04,0.08]:
    lam=maxre(S,w1,w2,0.0,(0.,ky)); p=lam[np.argsort(np.abs(lam))[:2]]
    print(f"undulation ky={ky}: omega={abs(p.imag).max():.4e}  omega/ky^2={abs(p.imag).max()/ky**2:.4f}  Re(a=0)={p.real.max():.1e}")
for kx in [0.005,0.01,0.02]:
    lam=maxre(S,w1,w2,0.0,(kx,0.)); p=lam[np.argsort(np.abs(lam))[:2]]
    print(f"compression kx={kx}: omega/kx={abs(p.imag).max()/kx:.4f}   (continuum D/s={D:.4f})")
