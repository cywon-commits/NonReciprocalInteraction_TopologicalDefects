from diag_stripe import *
a,b=0.3,0.6; fac=(b/a-1); z=np.zeros(2)
k=tuple(0.005*eh)
S0,_,_,_=steady(0,0,a,b,10)
for ue in [0.0,0.02,0.04,0.08,0.12,0.16]:
    S,w1,w2,r=steady(ue/fac,0.0,a,b,4000)          # u || q : pure parallel drive
    lam=jac(m,S,w1,z,0.0,k); p=lam[np.argsort(np.abs(lam))[:2]]
    w0=abs(p[0].imag-p[1].imag)/2; M=S[:,:2].mean(0)@eh
    print(f"u_eff||={ue:.2f}  M_e={M:.4f}  c_rot=omega0/k={w0/0.005:.4f}  c_rot^2-c0^2={(w0/0.005)**2-0.0629**2:.5f}  /M^2={((w0/0.005)**2-0.0629**2)/max(M*M,1e-12):.4f}")
