import numpy as np, json, os
from scipy.signal import welch
from scipy.optimize import least_squares
seeds={'0.0':2,'0.05':3,'0.07':1}; br=json.load(open("branches.json"))
def spectrum(u,nper=4096):
    r=np.load(f"traj_u{u}_T0.01_s{seeds[u]}.npy"); t=r[:,0].real; z=r[:,1:]; dt=t[1]-t[0]
    P=0
    for c in range(2):
        f,p=welch(z[:,c],fs=1/dt,nperseg=nper,return_onesided=False,detrend=False,scaling='density'); P=P+p
    o=np.argsort(f); w=2*np.pi*f[o]; P=P[o]/(2*np.pi)          # density per unit angular frequency
    return w,P,t[-1]-t[0],len(z)//nper
def model(p,w):
    s1,G1,w1,s2,G2,w2,bg=p
    L=lambda s,G,w0: s*G/np.pi/((w-w0)**2+G**2)
    return L(s1,G1,w1)+L(s2,G2,w2)+bg
def fit(w,P,nseg):
    m=np.abs(w)<0.03
    p0=[P[m].sum()*(w[1]-w[0])/2,1.5e-3,-3.4e-3,P[m].sum()*(w[1]-w[0])/2,1.5e-3,3.4e-3,np.median(P[np.abs(w)>0.05])]
    res=least_squares(lambda p:(np.log(model(p,w[m]))-np.log(P[m])),p0,bounds=([0,1e-5,-0.02,0,1e-5,0,0],[np.inf,0.02,0,np.inf,0.02,0.02,np.inf]))
    J=res.jac; cov=np.linalg.pinv(J.T@J)*np.mean(res.fun**2); return res.x,np.sqrt(np.diag(cov))
out={}
for u in seeds:
    if not os.path.exists(f"traj_u{u}_T0.01_s{seeds[u]}.npy"): continue
    w,P,span,nseg=spectrum(u); p,e=fit(w,P,nseg); out[u]=(p,e,span)
    print(f"u_eff={u} span={span:.0f}: branch(-): var={p[0]:.4f}±{e[0]:.4f} Gamma={p[1]*1e3:.2f}e-3 omega={p[2]*1e3:+.2f}e-3 | branch(+): var={p[3]:.4f}±{e[3]:.4f} Gamma={p[4]*1e3:.2f}e-3 omega={p[5]*1e3:+.2f}e-3 | bg={p[6]:.3g}")
    print(f"      Bloch: (-) Gamma={-br[u]['1'][0][0]*1e3:.2f}e-3 omega={br[u]['1'][0][1]*1e3:+.2f}e-3 | (+) Gamma={-br[u]['1'][1][0]*1e3:.2f}e-3 omega={br[u]['1'][1][1]*1e3:+.2f}e-3")
if '0.0' in out:
    for u in out:
        p,e,_=out[u]; p0,e0,_=out['0.0']
        for i,nm in ((0,'(-) pumped'),(3,'(+) other ')):
            j=0 if i==0 else 1; pred=br['0.0']['1'][j][0]/br[u]['1'][j][0]
            r=p[i]/p0[i]; er=r*np.hypot(e[i]/p[i],e0[i]/p0[i])
            print(f"u_eff={u} {nm}: T_eff/T = {r:.2f} ± {er:.2f}   predicted Gamma0/Gamma = {pred:.2f}")
json.dump({u:[list(map(float,out[u][0])),list(map(float,out[u][1]))] for u in out},open("fitspec.json","w"))
