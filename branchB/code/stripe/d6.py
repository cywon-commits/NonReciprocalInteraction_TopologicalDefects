from diag_stripe import *
a,b=0.3,0.6; fac=(b/a-1); z=np.zeros(2)
phi=np.radians(30)
for ue in [0.10,0.134]:
    S,w1,w2,r=steady(ue/fac,phi,a,b,4000); M=S[:,:2].mean(0)@eh
    row=[]
    for kk in [0.005,0.01,0.02,0.04,0.08,0.16]:
        k=tuple(kk*eh); lam=jac(m,S,w1,z,0.0,k); p=lam[np.argsort(np.abs(lam))[:2]]
        row.append(abs(p[0].imag-p[1].imag)/2/kk)
    print(f"phi=30 u_eff={ue} M={M:.3f}: omega0/k vs k(.005..0.16) =",np.round(row,4)," Doppler u_eff*sin(phi)=",round(ue*np.sin(phi),4))
