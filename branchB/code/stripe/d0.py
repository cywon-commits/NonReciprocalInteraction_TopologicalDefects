from diag_stripe import *
S,w1,w2,r=steady(0,0,0.3,0.6,3000); print("u=0 resid",r,"  D=",D)
for kk in [0.005,0.01,0.02,0.04,0.08,0.16]:
    p=low(jac(m,S,w1,w2,0.0,tuple(kk*eh)))
    print(f"undulation k={kk}: omega={abs(p.imag).max():.4e}  omega/k={abs(p.imag).max()/kk:.4e}  omega/k^2={abs(p.imag).max()/kk**2:.4f}  Re(alpha=0)={p.real.max():.1e}")
for kk in [0.005,0.01,0.02]:
    p=low(jac(m,S,w1,w2,0.0,tuple(kk*qh)))
    print(f"compression k={kk}: omega/k={abs(p.imag).max()/kk:.4f}  (continuum D/s={D/np.sqrt(2):.4f} -- per-bond D projected)")
