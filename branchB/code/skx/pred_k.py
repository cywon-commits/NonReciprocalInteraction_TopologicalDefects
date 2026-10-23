from skx import *
J=1.0; D=0.4; Lx,Ly=19,33; z=np.zeros(2); bb=0.5
m=Model(Lx,Ly,J,D,bb*D*D/J); S=np.load(f"S_B{bb}.npy")
def mode(w1,w2,a,k):
    lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k)); p=lam[np.argsort(np.abs(lam))[:2]]
    return p[np.argmax(p.imag)]
Gm,Dm,_=GD(S); a=0.3; b=0.6
u=np.array([1.,0]); V=np.linalg.solve(Gm-a*Dm, Gm@u-b*Dm@u); w1=V-u; w2=b*u-a*V
print("w1/u",w1,"w2/u",w2,"V/u",V)
for n in [1,2,3]:
    kx=2*np.pi*n/76; k=(kx,0.); e=1e-4; ex=np.array([1.,0])
    wn=mode(z,z,0.0,k).imag
    Kt=-(mode(e*ex,z,0.0,k).imag-mode(-e*ex,z,0.0,k).imag)/(2*e)
    KP=(mode(z,e*ex,0.0,k).real-mode(z,-e*ex,0.0,k).real)/(2*e)
    P=-(mode(z,z,e,k).real-mode(z,z,-e,k).real)/(2*e)/wn
    s=w2[0]*KP+a*P*w1[0]*Kt; ust=abs(a*P*wn/s)*(b/a-1)
    print(f"n={n} k={kx:.4f}: omega_n={wn:.3e} kappa_tau={Kt/kx:+.3f} kappa_P={KP/(P*kx):+.3f} P={P:.2f} | u_eff* overlap={ust:.4f}  affine(omega/k)={wn/kx:.4f}",flush=True)
