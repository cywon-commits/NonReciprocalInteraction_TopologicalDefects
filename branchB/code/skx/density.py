from skx import *
J=1.0; D=0.4; Lx,Ly=19,33; a,b=0.3,0.6; du=0.002
kx=2*np.pi/(Lx*20); k=(kx,0.); z=np.zeros(2)
S0=np.load("S0.npy")
for bb in [0.3,0.5,0.7,0.85]:
    m=Model(Lx,Ly,J,D,bb*D*D/J)
    S=rk4(m,S0.copy(),z,z,0.5,0.1,12000)
    res=np.abs(m.rhs(S,z,z,0.5)).max(); Q=GD(S)[2]
    core=np.mean(S[...,2]<0)   # area fraction of reversed core
    lam=np.linalg.eigvals(bloch_jac(m,S,z,z,0.0,k)); wu=abs(lam[np.argsort(np.abs(lam))[:2]].imag).max()
    lam=np.linalg.eigvals(bloch_jac(m,S,z,z,a,k)); p0=lam[np.argsort(np.abs(lam))[:2]]
    Sd,V,w1,w2,r=find_V(m,S.copy(),np.array([du,0.]),a,b)
    lam=np.linalg.eigvals(bloch_jac(m,Sd,w1,w2,a,k)); p1=lam[np.argsort(np.abs(lam))[:2]]
    ueff=(b/a-1)*du; sl=(p1.real.max()-p0.real.max())/ueff; ust=-p0.real.max()/sl
    print(f"B/(D^2/J)={bb}: resid={res:.0e} Q={Q:.2f} core fraction={core:.3f} | omega_u/k={wu/kx:.4f} u_eff*={ust:.4f} lambda=(omega_u/k)/u*={wu/kx/ust:.3f}",flush=True)
