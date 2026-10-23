from skx import *
import sys, re
from scipy.sparse.linalg import eigs
# Affine (Thiele-continuum) phonon via symplectic Galerkin projection onto e^{ik r}(c_x d_x S + c_y d_y S)
J=1.0; D=0.4; z=np.zeros(2)
def affine_c2(S,bb,nk=20):
    Lx,Ly=S.shape[:2]; m=Model(Lx,Ly,J,D,bb*D*D/J)
    kx=2*np.pi/(Lx*nk); k=(kx,0.)
    Jm=bloch_jac(m,S,z,z,0.0,k); N=Lx*Ly
    e1,e2=basis(S); E=[e1.reshape(N,3),e2.reshape(N,3)]
    gx=((np.roll(S,-1,0)-np.roll(S,1,0))/2).reshape(N,3); gy=((np.roll(S,-1,1)-np.roll(S,1,1))/2).reshape(N,3)
    X=np.arange(Lx)[:,None]*np.ones((1,Ly)); ph=np.exp(1j*kx*X).reshape(N)
    Psi=np.zeros((2*N,2),complex)
    for c,g in enumerate((gx,gy)):
        Psi[:N,c]=np.sum(g*E[0],1)*ph; Psi[N:,c]=np.sum(g*E[1],1)*ph
    Om=np.block([[np.zeros((N,N)),np.eye(N)],[-np.eye(N),np.zeros((N,N))]])
    A=Psi.conj().T@Om@Psi; Bm=Psi.conj().T@Om@Jm@Psi
    lam=np.linalg.eigvals(np.linalg.solve(A,Bm))
    wa=np.abs(lam.imag).max()
    lt=eigs(Jm,k=6,sigma=1e-9,return_eigenvectors=False); pos=lt[lt.imag>0]; wt=pos[np.argmin(np.abs(pos))].imag
    return wt/kx**2, wa/kx**2, lam
if __name__=="__main__":
    for l in open("kgrid.txt"):
        r=l.split(); Lx=int(r[0]); bb=r[2]
        if 'resid=1e-03' in l: continue
        S=np.load(f"Sgrid_{Lx}_{bb}.npy")
        ct,ca,lam=affine_c2(S,float(bb))
        kt=float(re.search(r"kappa_tau=([\d.]+)",l).group(1)); kp=float(re.search(r"kappa_P=([\d.]+)",l).group(1))
        print(f"a={Lx:3d} B={bb:>5}: c2_true={ct:.4f} c2_aff={ca:.4f} ratio={ct/ca:.3f} | kappa_tau={kt:.3f} kappa_P={kp:.3f} | Re(lam_aff)={np.abs(lam.real).max():.1e}",flush=True)
