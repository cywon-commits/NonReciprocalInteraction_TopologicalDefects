from csl import *
J=1.0; D=0.3; N=40; k=2*np.pi/(N*40)
for Kp in [0.2,1.0]:
    for B in [0.01,0.02,0.035]:
        m,S=relax(N,J,D,B,Kp); kt,_=kappa(m,S,k)
        th=np.unwrap(np.arctan2(S[:,1],S[:,2])); tp=np.diff(np.concatenate([th,[th[0]+2*np.pi]])); psi=(tp+np.roll(tp,1))/2
        M=2*Kp+B*np.cos(th)+2*D*psi-2*J*psi**2
        f0=1/(np.mean(psi**2)*np.mean(1/psi**2))
        f1=np.mean(1/M)/(np.mean(psi**-2)*np.mean(psi**2/M))
        print(f"Kp={Kp} B={B}: kappa={abs(kt):.4f}  scalar formula={f0:.4f}  with conjugate stiffness M(x)={f1:.4f}  M range=({M.min():.3f},{M.max():.3f})",flush=True)
