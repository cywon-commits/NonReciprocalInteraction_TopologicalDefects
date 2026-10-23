import numpy as np, json, os
from fitspec import spectrum, seeds, br
# amplitudes of the two branches with (omega, Gamma) fixed to the Bloch values; flat background free
def L(w,G,w0): return G/np.pi/((w-w0)**2+G**2)
out={}
for u in seeds:
    if not os.path.exists(f"traj_u{u}_T0.01_s{seeds[u]}.npy"): continue
    w,P,span,nseg=spectrum(u); m=np.abs(w)<0.03
    (Ga,wa),(Gb,wb)=[(-x[0],x[1]) for x in br[u]['1']]
    X=np.stack([L(w[m],Ga,wa),L(w[m],Gb,wb),np.ones(m.sum())],1); Wt=1/P[m]
    c,*_=np.linalg.lstsq(X*Wt[:,None],P[m]*Wt,rcond=None)
    # bootstrap over welch segments is costly; use residual-based error
    r=(X@c-P[m])*Wt; cov=np.linalg.pinv((X*Wt[:,None]).T@(X*Wt[:,None]))*np.mean(r**2)
    out[u]=(c,np.sqrt(np.diag(cov)),span)
    print(f"u_eff={u} span={span:.0f}: var(-)={c[0]:.4f}±{np.sqrt(cov[0,0]):.4f} var(+)={c[1]:.4f}±{np.sqrt(cov[1,1]):.4f} bg={c[2]:.3g}")
for u in out:
    c,e,_=out[u]; c0,e0,_=out['0.0']
    for i,nm in ((0,'(-) pumped'),(1,'(+) other ')):
        pred=br['0.0']['1'][i][0]/br[u]['1'][i][0]; r=c[i]/c0[i]; er=r*np.hypot(e[i]/c[i],e0[i]/c0[i])
        print(f"u_eff={u} {nm}: T_eff/T = {r:.2f} ± {er:.2f}   predicted 1/(1-+x) = Gamma0/Gamma = {pred:.2f}")
json.dump({u:[list(map(float,out[u][0])),list(map(float,out[u][1]))] for u in out},open("fitamp.json","w"))
