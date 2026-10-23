from skx import *
import time
J=1.0; D=0.4; Lx,Ly=19,33; B=0.5*D*D/J
m=Model(Lx,Ly,J,D,B); S=np.load("S0.npy"); a=0.3; z=np.zeros(2)
for kx in [0.0, 2*np.pi/(Lx*40), 2*np.pi/(Lx*20), 2*np.pi/(Lx*10)]:
    t=time.time()
    lam=np.linalg.eigvals(bloch_jac(m,S,z,z,a,(kx,0.)))
    i=np.argsort(np.abs(lam))[:4]
    print(f"kx={kx:.5f}  smallest |lam|: ", np.round(lam[i],7), f"  maxRe={lam.real.max():.2e}  ({time.time()-t:.1f}s)")
