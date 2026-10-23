from csl import *
import sys
J=1.0; D=0.3; N=int(sys.argv[1]); Kp=float(sys.argv[2])
k=2*np.pi/(N*40)
print(f"N={N} Kp={Kp}  (continuum critical field pi^2 D^2/16J = {np.pi**2*D**2/16:.4f})")
for B in [0.02,0.035]:
    m,S=relax(N,J,D,B,Kp)
    kt,c2=kappa(m,S,k); f1,f2=formula(S)
    print(f" B={B:.3f}: kappa_tau={kt:.4f}  formula 1/(<psi^2><psi^-2>)={f1:.4f} (bond version {f2:.4f})  c2={c2:.4f}  min/max density={(np.min(np.diff(np.unwrap(np.arctan2(S[:,1],S[:,2]))))):.4f}",flush=True)
