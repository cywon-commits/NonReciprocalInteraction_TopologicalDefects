from skx import *
# Neel cycloid, q || x, uniform along y. Cell = one period (Lx sites) x 1 site; Bloch k=(kx,ky) exact.
J=1.0; Lx=30; th=2*np.pi/Lx; D=J*np.tan(th)
m=Model(Lx,1,J,D,0.0)
i=np.arange(Lx)
def cycloid(M=0.0):
    r=np.sqrt(1-M*M)
    S=np.zeros((Lx,1,3)); S[:,0,0]=r*np.sin(th*i); S[:,0,1]=M; S[:,0,2]=r*np.cos(th*i); return S
def steady(u,phi,a,b,nstep=20000,dt=0.1):
    uv=u*np.array([np.cos(phi),np.sin(phi)])
    w1=(b/a-1)*uv; w2=np.zeros(2)          # frame V=(b/a)u  (G=0 pattern)
    S=rk4(m,cycloid(),w1,w2,a,dt,nstep)
    return S,w1,w2,np.abs(m.rhs(S,w1,w2,a)).max()
def maxre(S,w1,w2,a,k):
    lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k)); return lam
