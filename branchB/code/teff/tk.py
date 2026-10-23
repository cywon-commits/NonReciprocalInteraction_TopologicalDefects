import numpy as np, json, os, glob
from scipy.signal import welch
br=json.load(open("branches_big.json")); R=np.load("big_Rref.npy")
runs={'0.0':'big_traj_u0.0_T0.01_s12.npy','0.07':'big_traj_u0.07_T0.01_s11.npy'}
def L(w,G,w0): return G/np.pi/((w-w0)**2+G**2)
def zq(d,q):
    t=d[:,0]; U=d[:,1:].reshape(len(d),-1,2); ph=np.exp(-1j*(R@q))/np.sqrt(len(R))
    return t, np.einsum('n,tnc->tc',ph,U)
def fitamp(t,z,u,key,nper=4096):
    dt=t[1]-t[0]; P=0
    for c in range(2):
        f,p=welch(z[:,c],fs=1/dt,nperseg=min(nper,len(z)),return_onesided=False,detrend=False); P=P+p
    o=np.argsort(f); w=2*np.pi*f[o]; P=P[o]/(2*np.pi); m=np.abs(w)<0.04
    (Ga,wa),(Gb,wb)=[(-x[0],x[1]) for x in br[u][key]]
    X=np.stack([L(w[m],Ga,wa),L(w[m],Gb,wb),np.ones(m.sum())],1); W=1/P[m]
    c,*_=np.linalg.lstsq(X*W[:,None],P[m]*W,rcond=None)
    r=(X@c-P[m])*W; cov=np.linalg.pinv((X*W[:,None]).T@(X*W[:,None]))*np.mean(r**2)
    return c,np.sqrt(np.diag(cov))
res={}
for u,fn in runs.items():
    if not os.path.exists(fn): continue
    d=np.load(fn); res[u]={}
    for key in br['0.0']:
        mm,ll=map(int,key.split(',')); q=np.array([2*np.pi*mm/76,2*np.pi*ll/66])
        t,z=zq(d,q); res[u][key]=fitamp(t,z,u,key)
    print(u,"span",d[-1,0]-d[0,0])
if '0.0' in res and '0.07' in res:
    print(" q (m,l)   |q|     angle |  branch(-) T_eff/T  pred |  branch(+) T_eff/T  pred")
    for key in br['0.0']:
        mm,ll=map(int,key.split(',')); q=np.array([2*np.pi*mm/76,2*np.pi*ll/66])
        c,e=res['0.07'][key]; c0,e0=res['0.0'][key]; s=[]
        for i in (0,1):
            r=c[i]/c0[i]; er=r*np.hypot(e[i]/c[i],e0[i]/c0[i]); pred=br['0.0'][key][i][0]/br['0.07'][key][i][0]; s.append(f"{r:5.2f}±{er:4.2f} {pred:5.2f}")
        print(f" ({mm:+d},{ll})  {np.linalg.norm(q):.3f}  {np.degrees(np.arctan2(q[1],q[0])):5.0f}° | {s[0]} | {s[1]}")
