from sot import *
# Static test of the strain-dependent-force picture: evaluate the SOT linearisation on (i) the exact mode, (ii) the static
# relaxed (internal-strain) mode, (iii) the affine mode -- all at alpha=0 (symplectic first-order formula).
J,D=1.0,0.4; B=0.5*D*D; S=np.load("S_B0.5.npy"); c=1e-3; z=np.zeros(2); Lx,Ly=19,33; N=Lx*Ly
m0=SOTModel(Lx,Ly,J,D,B,0.,0.,(0,1,0)); mc=SOTModel(Lx,Ly,J,D,B,0.,c,(0,1,0))
Om=np.block([[np.zeros((N,N)),np.eye(N)],[-np.eye(N),np.zeros((N,N))]])
e1,e2=basis(S); E=[e1.reshape(N,3),e2.reshape(N,3)]
gx=((np.roll(S,-1,0)-np.roll(S,1,0))/2).reshape(N,3); gy=((np.roll(S,-1,1)-np.roll(S,1,1))/2).reshape(N,3)
for nk in (40,20,10):
    kx=2*np.pi/(Lx*nk); k=(kx,0.)
    Jm=bloch_jac(m0,S,z,z,0.0,k); dJ=(bloch_jac(mc,S,z,z,0.0,k)-Jm)/c
    H=-Om@Jm; H=(H+H.conj().T)/2
    ph=np.exp(1j*kx*(np.arange(Lx)[:,None]*np.ones((1,Ly))).reshape(N)); Phi=np.zeros((2*N,2),complex)
    for cc,g in enumerate((gx,gy)): Phi[:N,cc]=np.sum(g*E[0],1)*ph; Phi[N:,cc]=np.sum(g*E[1],1)*ph
    M=Phi.conj().T@Om@Phi; W=Om.T@Phi@np.linalg.inv(M).conj().T; HiW=np.linalg.solve(H,W); Kr=np.linalg.inv(W.conj().T@HiW); Pr=HiW@Kr
    def mode(P):
        A=P.conj().T@Om@P; Bm=P.conj().T@Om@Jm@P; lam,C=np.linalg.eig(np.linalg.solve(A,Bm)); return [P@C[:,i] for i in np.argsort(lam.imag)], np.sort(lam.imag)
    def shift(x): return (x.conj()@Om@dJ@x)/(x.conj()@Om@x)
    ev,X=np.linalg.eig(Jm); idx=np.argsort(np.abs(ev))[:2]; idx=idx[np.argsort(ev[idx].imag)]
    ex=[shift(X[:,i]) for i in idx]; rel,_=mode(Pr); aff,_=mode(Phi)
    print(f"k={kx:.4f}: Re dlambda/dc (branch -,+)  exact {[round(s.real,4) for s in ex]} | relaxed {[round(shift(x).real,4) for x in rel]} | affine {[round(shift(x).real,4) for x in aff]} | omega={abs(ev[idx[0]].imag):.2e}",flush=True)
