from diag_stripe import *
import time
a,b=0.3,0.6; fac=(b/a-1); z=np.zeros(2)
S00,_,_,_=steady(0,0,a,b,10)
def omega_u(k):
    p=low(jac(m,S00,z,z,0.0,k)); return abs(p.imag).max()
def g(ue,phi,k):
    u=ue/fac
    if abs(np.cos(phi))<1e-12: S,w1,w2=S00,fac*u*eh,z
    else: S,w1,w2,r=steady(u,phi,a,b,3000)
    lam=jac(m,S,w1,w2,a,k); lam=lam[np.abs(lam)>1e-9]; return lam.real.max()
def thresh(phi,k,lo,hi,it=14):
    for _ in range(it):
        mid=(lo+hi)/2
        if g(mid,phi,k)>0: hi=mid
        else: lo=mid
    return (lo+hi)/2
with open("d3.txt","w") as f:
  for phi_deg in [90,60,30]:
    phi=np.radians(phi_deg)
    for kk in [0.01,0.04,0.16]:
        k=tuple(kk*eh); pred=omega_u(k)/(kk*np.sin(phi))
        t=thresh(phi,k,0.3*pred,3*pred)
        f.write(f"phi={phi_deg:3d} k={kk:.2f}: u_eff*={t:.4f}   omega_u/(k sin phi)={pred:.4f}   ratio={t/pred:.3f}\n"); f.flush()
