from skx import *
import sys, re
# Type-B phonon: at small k the Bloch eigenvector is the STATIC relaxed (internal-strain) mode, with the polarization
# fixed by the 2x2 gyrotropic problem. Test: Doppler overlap of the static relaxed mode == dynamic kappa_tau.
J,D=1.0,0.4; z=np.zeros(2)
def analyse(S,bb,nk=20):
    Lx,Ly=S.shape[:2]; m=Model(Lx,Ly,J,D,bb*D*D/J); N=Lx*Ly; kx=2*np.pi/(Lx*nk); k=(kx,0.)
    Jm=bloch_jac(m,S,z,z,0.0,k)
    e=1e-4; O=(bloch_jac(m,S,np.array([e,0.]),z,0.0,k)-Jm)/e          # reactive Doppler operator (d Jm / d w1x)
    Om=np.block([[np.zeros((N,N)),np.eye(N)],[-np.eye(N),np.zeros((N,N))]])
    H=-Om@Jm; H=(H+H.conj().T)/2                                       # Hessian (symplectic form Om, dynamics Om x' = -H x)
    e1,e2=basis(S); E=[e1.reshape(N,3),e2.reshape(N,3)]
    gx=((np.roll(S,-1,0)-np.roll(S,1,0))/2).reshape(N,3); gy=((np.roll(S,-1,1)-np.roll(S,1,1))/2).reshape(N,3)
    ph=np.exp(1j*kx*(np.arange(Lx)[:,None]*np.ones((1,Ly))).reshape(N))
    Phi=np.zeros((2*N,2),complex)
    for c,g in enumerate((gx,gy)): Phi[:N,c]=np.sum(g*E[0],1)*ph; Phi[N:,c]=np.sum(g*E[1],1)*ph
    M=Phi.conj().T@Om@Phi                                              # gyro matrix (antisymmetric)
    W=Om.T@Phi@np.linalg.inv(M).conj().T                               # constraint vectors: displacement c = W^dag Psi
    HiW=np.linalg.solve(H,W); Krel=np.linalg.inv(W.conj().T@HiW)       # relaxed stiffness 2x2
    Psi_rel=HiW@Krel                                                   # relaxed static modes (columns: unit u_x, unit u_y)
    Kaff=Phi.conj().T@H@Phi
    # polarization from the 2x2 gyro problem:  omega * (i M) c = Krel c  ->  use Jm-Galerkin on Psi_rel
    A=Psi_rel.conj().T@Om@Psi_rel; Bm=Psi_rel.conj().T@Om@Jm@Psi_rel; lam,C=np.linalg.eig(np.linalg.solve(A,Bm))
    i=np.argmax(lam.imag); c=C[:,i]; Psi=Psi_rel@c
    shift_static=(Psi.conj()@Om@O@Psi)/(Psi.conj()@Om@Psi)            # d lambda / d w1x for the static relaxed mode
    lt=np.linalg.eigvals(Jm); pos=lt[lt.imag>0]; l0=pos[np.argmin(np.abs(pos))]
    lt2=np.linalg.eigvals(Jm+e*O); l1=lt2[np.argmin(np.abs(lt2-l0))]; shift_dyn=(l1-l0)/e
    # affine mode for comparison
    A2=Phi.conj().T@Om@Phi; B2=Phi.conj().T@Om@Jm@Phi; lam2,C2=np.linalg.eig(np.linalg.solve(A2,B2)); c2=C2[:,np.argmax(lam2.imag)]; Pa=Phi@c2
    shift_aff=(Pa.conj()@Om@O@Pa)/(Pa.conj()@Om@Pa)
    rL=(Krel[0,0]/Kaff[0,0]).real; rT=(Krel[1,1]/Kaff[1,1]).real
    return dict(kt_dyn=abs(shift_dyn.imag)/kx, kt_static=abs(shift_static.imag)/kx, kt_aff=abs(shift_aff.imag)/kx,
                w_dyn=l0.imag/kx**2, w_rel=lam[i].imag/kx**2, rL=rL, rT=rT, pol=abs(c[0]/c[1]))
if __name__=="__main__":
    for Lx,bb in [(19,'0.5'),(13,'0.5'),(19,'0.3'),(19,'0.85'),(23,'0.6')]:
        r=analyse(np.load(f"Sgrid_{Lx}_{bb}.npy"),float(bb))
        print(f"a={Lx} B={bb}: kappa_tau dynamic={r['kt_dyn']:.3f} | static-relaxed mode={r['kt_static']:.3f} | affine mode={r['kt_aff']:.3f} || c2 dyn={r['w_dyn']:.3f} rel={r['w_rel']:.3f} | r_L={r['rL']:.3f} r_T={r['rT']:.3f} |u_x/u_y|={r['pol']:.2f}",flush=True)
