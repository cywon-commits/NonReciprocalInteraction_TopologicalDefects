from diag_stripe import *
a,b=0.3,0.6; fac=(b/a-1); z=np.zeros(2)
def w0_driven(ue,phi,k):
    S,w1,w2,r=steady(ue/fac,phi,a,b,3000)
    lam=jac(m,S,w1,z,0.0,k)                    # alpha=0, reactive Doppler only
    p=lam[np.argsort(np.abs(lam))[:2]]
    return abs(p[0].imag-p[1].imag)/2, S[:,:2].mean(0)@eh
for phi_deg,kk,ue in [(30,0.01,0.20),(30,0.01,0.30),(30,0.01,0.378),(60,0.01,0.0812),(30,0.04,0.1344)]:
    phi=np.radians(phi_deg); k=tuple(kk*eh)
    w0,M=w0_driven(ue,phi,k)
    print(f"phi={phi_deg} k={kk} u_eff={ue}: cone M_e={M:.3f}  omega0(driven)/(k sin phi)={w0/(kk*np.sin(phi)):.4f}  (vs u_eff {ue})")
