from skx import *
import re, json, sys
J,D=1.0,0.4; z=np.zeros(2)
def static(S,bb,nk=20):
    Lx,Ly=S.shape[:2]; m=Model(Lx,Ly,J,D,bb*D*D/J); N=Lx*Ly; kx=2*np.pi/(Lx*nk); k=(kx,0.)
    Jm=bloch_jac(m,S,z,z,0.0,k); e=1e-4; O=(bloch_jac(m,S,np.array([e,0.]),z,0.0,k)-Jm)/e
    Om=np.block([[np.zeros((N,N)),np.eye(N)],[-np.eye(N),np.zeros((N,N))]]); H=-Om@Jm; H=(H+H.conj().T)/2
    e1,e2=basis(S); E=[e1.reshape(N,3),e2.reshape(N,3)]
    gx=((np.roll(S,-1,0)-np.roll(S,1,0))/2).reshape(N,3); gy=((np.roll(S,-1,1)-np.roll(S,1,1))/2).reshape(N,3)
    ph=np.exp(1j*kx*(np.arange(Lx)[:,None]*np.ones((1,Ly))).reshape(N)); Phi=np.zeros((2*N,2),complex)
    for c,g in enumerate((gx,gy)): Phi[:N,c]=np.sum(g*E[0],1)*ph; Phi[N:,c]=np.sum(g*E[1],1)*ph
    M=Phi.conj().T@Om@Phi; W=Om.T@Phi@np.linalg.inv(M).conj().T; HiW=np.linalg.solve(H,W); Kr=np.linalg.inv(W.conj().T@HiW); Pr=HiW@Kr
    Ka=Phi.conj().T@H@Phi
    A=Pr.conj().T@Om@Pr; Bm=Pr.conj().T@Om@Jm@Pr; lam,C=np.linalg.eig(np.linalg.solve(A,Bm)); c=C[:,np.argmax(lam.imag)]; P=Pr@c
    kt=abs(((P.conj()@Om@O@P)/(P.conj()@Om@P)).imag)/kx
    # per-polarisation Doppler overlaps of the relaxed modes (L = x, T = y)
    kL=abs(((Pr[:,0].conj()@Om@O@Pr[:,1])/(Pr[:,0].conj()@Om@Pr[:,1])).imag)/kx
    return dict(kt=kt,kLT=kL,rL=(Kr[0,0]/Ka[0,0]).real,rT=(Kr[1,1]/Ka[1,1]).real,pol=abs(c[0]/c[1]))
out={}
for l in open("../skx/kgrid.txt"):
    r=l.split(); Lx=int(r[0]); bb=r[2]
    res=float(re.search(r"resid=([\de.+-]+)",l).group(1)); P=float(re.search(r"P=([-\d.]+)",l).group(1))
    if res>1e-8 or not (1.3<P<2.5): continue
    S=np.load(f"Sgrid_{Lx}_{bb}.npy"); R=float(re.search(r"R=([\d.]+)",l).group(1))
    d=static(S,float(bb)); d.update(a=Lx,B=float(bb),R=R,Ly=S.shape[1]); out[f"{Lx}_{bb}"]=d
    print(f"a={Lx} B={bb}: kt={d['kt']:.3f} rL={d['rL']:.3f} rT={d['rT']:.3f} pol={d['pol']:.2f} R={R}",flush=True)
    json.dump(out,open("grid_static.json","w"))
