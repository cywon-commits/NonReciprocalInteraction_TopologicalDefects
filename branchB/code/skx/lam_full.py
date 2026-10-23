from skx import *
# First-order theory in the co-moving frame:
#   i tau_z (d_t - w1.grad) Psi = H Psi + alpha d_t Psi + (w2.grad) Psi
#   => Im omega ~ -alpha P (omega_n - w1.K_tau) + w2.K_P
# K_tau: reactive overlap (tau_z metric), K_P: dissipative overlap (identity metric), P: ellipticity.
# All three read off from eigenvalue derivatives at alpha=0, u=0.
J=1.0; D=0.4; Lx,Ly=19,33; z=np.zeros(2)
def mode(m,S,w1,w2,a,k,sel=None):
    lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k)); p=lam[np.argsort(np.abs(lam))[:2]]
    return p[np.argmax(p.imag)]            # branch with positive frequency  (e^{lam t}, omega = Im lam)
for bb in [0.3,0.5,0.7,0.85]:
    m=Model(Lx,Ly,J,D,bb*D*D/J); S=np.load(f"S_B{bb}.npy")
    kx=2*np.pi/(Lx*20); k=(kx,0.); e=1e-4
    l0=mode(m,S,z,z,0.0,k); wn=l0.imag
    Kt=[-(mode(m,S,e*v,z,0.0,k).imag-mode(m,S,-e*v,z,0.0,k).imag)/(2*e) for v in (np.array([1.,0]),np.array([0,1.]))]
    KP=[(mode(m,S,z,e*v,0.0,k).real-mode(m,S,z,-e*v,0.0,k).real)/(2*e) for v in (np.array([1.,0]),np.array([0,1.]))]
    P=-(mode(m,S,z,z,e,k).real-mode(m,S,z,z,-e,k).real)/(2*e)/wn
    Kt=np.array(Kt); KP=np.array(KP)
    # Thiele mapping for a=0.3: w1=c1 u, w2=c2 u (u along x), predicted threshold in u_eff
    a=0.3
    Gm,Dm,_=GD(S)
    out=[]
    for b in [0.6,0.9,0.0]:
        u=np.array([1.,0]); V=np.linalg.solve(Gm-a*Dm, Gm@u-b*Dm@u); w1=V-u; w2=b*u-a*V
        # growth(u) = u*(w2.KP + a P w1.Kt) - a P wn  -> threshold u*
        s=w2@KP + a*P*(w1@Kt); ust=a*P*wn/s
        out.append(f"b={b}: u_eff*={ (b/a-1)*ust:+.4f}")
    print(f"B={bb}: omega_n/k={wn/kx:.4f} P={P:.3f}  K_tau/k=({Kt[0]/kx:+.3f},{Kt[1]/kx:+.3f})  K_P/(Pk)=({KP[0]/(P*kx):+.3f},{KP[1]/(P*kx):+.3f})  | "+"  ".join(out),flush=True)
