from sot import *
# Exact first-order attribution of the damping-like-SOT pumping:  d lambda = y^H dJ x / y^H x, dJ split into
#  (a) co-moving advection (w1=V, w2=-alpha V), (b) direct SOT linearisation on the frozen texture, (c) static texture deformation
J,D=1.0,0.4; B=0.5*D*D; a=0.3; S0=np.load("S_B0.5.npy"); c=1e-3
m0=SOTModel(19,33,J,D,B,0.,0.,(0,1,0)); mc=SOTModel(19,33,J,D,B,0.,c,(0,1,0))
Sc,V,r=find_drift(mc,S0.copy(),a,iters=12,n=1500); print("drift V/c =",V/c," resid",r)
z=np.zeros(2)
for ang in (0,90):
    kk=2*np.pi/(19*20); k=(kk*np.cos(np.radians(ang)),kk*np.sin(np.radians(ang)))
    J0=bloch_jac(m0,S0,z,z,a,k); ev,X=np.linalg.eig(J0); evl,Y=np.linalg.eig(J0.conj().T)
    idx=np.argsort(np.abs(ev))[:2]
    parts={"(a) advection":(bloch_jac(m0,S0,V,-a*V,a,k)-J0)/c,
           "(b) SOT on frozen texture":(bloch_jac(mc,S0,z,z,a,k)-J0)/c,
           "(c) texture deformation":(bloch_jac(m0,Sc,z,z,a,k)-J0)/c}
    Jt=bloch_jac(mc,Sc,V,-a*V,a,k); evt=np.linalg.eigvals(Jt)
    print(f"k angle {ang}:")
    for i in idx:
        l=ev[i]; x=X[:,i]; j=np.argmin(np.abs(evl-l.conj())); y=Y[:,j]
        out={nm:(y.conj()@dJ@x)/(y.conj()@x) for nm,dJ in parts.items()}
        exact=(evt[np.argmin(np.abs(evt-l))]-l)/c
        print(f"   branch Im={l.imag:+.2e}: Re dlambda/dc  " + "  ".join(f"{nm}={v.real:+.4f}" for nm,v in out.items()) + f"  | sum={sum(v.real for v in out.values()):+.4f} exact={exact.real:+.4f}")
