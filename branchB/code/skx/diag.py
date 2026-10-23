from skx import *
J=1.0; D=0.4; Lx,Ly=19,33; B=0.5*D*D/J
m=Model(Lx,Ly,J,D,B); S0=np.load("S0.npy")
a,b=0.3,0.6; du=0.002; k=(2*np.pi/(Lx*20),0.)
uv=np.array([du,0.])
S,V,w1,w2,r=find_V(m,S0.copy(),uv,a,b)
def mre(Sx,w1,w2):
    lam=np.linalg.eigvals(bloch_jac(m,Sx,w1,w2,a,k)); p=lam[np.argsort(np.abs(lam))[:2]]; return p.real.max()
z=np.zeros(2)
base=mre(S0,z,z)
print("full (deformed background + Doppler):", (mre(S,w1,w2)-base)/du)
print("Doppler only (frozen u=0 background):", (mre(S0,w1,w2)-base)/du)
print("deformation only (deformed background, w=0):", (mre(S,z,z)-base)/du)
print("affine Landau prediction Gamma0*k/omega_u:", -base*k[0]/1.6834e-04)
