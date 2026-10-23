from skx import *
import sys, time
from scipy.sparse.linalg import eigs
# kappa_tau, kappa_P, P for a relaxed SkX in an Lx x Ly cell (2 skyrmions), field B (units D^2/J)
J=1.0; D=0.4
Lx=int(sys.argv[1]); Ly=int(sys.argv[2]); bb=float(sys.argv[3])
B=bb*D*D/J; m=Model(Lx,Ly,J,D,B); z=np.zeros(2)
x,y=np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
G1=2*np.pi/Lx; G2=2*np.pi/Ly
ks=[np.array([0,2*G2]),np.array([G1,-G2]),np.array([-G1,-G2])]
S=np.zeros((Lx,Ly,3)); S[...,2]=0.6
for kk in ks:
    ph=kk[0]*x+kk[1]*y; kh=kk/np.linalg.norm(kk)
    S[...,2]+=-np.cos(ph); S[...,0]+=kh[0]*np.sin(ph); S[...,1]+=kh[1]*np.sin(ph)
S/=np.linalg.norm(S,axis=-1)[...,None]
t0=time.time()
for it in range(40):
    S=rk4(m,S,z,z,0.5,0.1,1000); r=np.abs(m.rhs(S,z,z,0.5)).max()
    if r<1e-10: break
Q=GD(S)[2]
Rsk=np.sqrt(np.sum(S[...,2]<0)/(2*np.pi))
kx=2*np.pi/(Lx*20); k=(kx,0.)
def ph(w1,w2,a,sig):
    lam=eigs(bloch_jac(m,S,w1,w2,a,k),k=4,sigma=sig,return_eigenvectors=False)
    return lam[np.argmin(np.abs(lam-sig))]
lam_all=eigs(bloch_jac(m,S,z,z,0.0,k),k=6,sigma=1e-9,return_eigenvectors=False)
pos=lam_all[lam_all.imag>0]; l0=pos[np.argmin(np.abs(pos))]; wn=l0.imag
e=1e-4; ex=np.array([1.,0])
Kt=-(ph(e*ex,z,0.0,l0).imag-wn)/e
KP=(ph(z,e*ex,0.0,l0).real-l0.real)/e
P=-(ph(z,z,e,l0).real-l0.real)/e/wn
ell=np.sqrt(J/B)
line=f"{Lx} {Ly} {bb} Q={Q:.3f} resid={r:.0e} R={Rsk:.3f} a={Lx} ell={ell:.3f} gap/ell={(Lx-2*Rsk)/ell:.3f} omega/k^2={wn/kx**2:.4f} P={P:.3f} kappa_tau={abs(Kt)/kx:.4f} kappa_P={abs(KP)/(P*kx):.4f} time={time.time()-t0:.0f}"
print(line,flush=True); open("kgrid.txt","a").write(line+"\n")
np.save(f"Sgrid_{Lx}_{bb}.npy",S)
