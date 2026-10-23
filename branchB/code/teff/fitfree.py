import numpy as np, json, os, sys
from scipy.signal import welch
from scipy.optimize import least_squares
seeds={'0.0':2,'0.05':3,'0.07':1}; br=json.load(open("branches.json"))
def spectrum(z,dt,nper=4096):
    P=0
    for c in range(2):
        f,p=welch(z[:,c],fs=1/dt,nperseg=nper,return_onesided=False,detrend=False); P=P+p
    o=np.argsort(f); return 2*np.pi*f[o],P[o]/(2*np.pi)
def L(w,G,w0): return G/np.pi/((w-w0)**2+G**2)
def fit(w,P,u):
    (Ga,wa),(Gb,wb)=[(-x[0],x[1]) for x in br[u]['1']]; m=np.abs(w)<0.03
    mod=lambda p:p[0]*L(w[m],p[1],p[2])+p[3]*L(w[m],p[4],p[5])+p[6]
    p0=[0.05,Ga,wa,0.05,Gb,wb,np.median(P[np.abs(w)>0.05])]
    lo=[0,0.5*Ga,1.4*wa,0,0.5*Gb,0.6*wb,0]; hi=[np.inf,2*Ga,0.6*wa,np.inf,2*Gb,1.4*wb,np.inf]
    r=least_squares(lambda p:np.log(mod(p))-np.log(P[m]),p0,bounds=(lo,hi)); return r.x
res={}
for u in seeds:
    fn=f"traj_u{u}_T0.01_s{seeds[u]}.npy"
    if not os.path.exists(fn): continue
    d=np.load(fn); t=d[:,0].real; z=d[:,1:]; dt=t[1]-t[0]
    w,P=spectrum(z,dt); p=fit(w,P,u)
    # error by splitting the run into 4 parts
    parts=[fit(*spectrum(z[i*len(z)//4:(i+1)*len(z)//4],dt,nper=2048),u) for i in range(4)]
    e=np.std(parts,0)/np.sqrt(4); res[u]=(p,e,t[-1]-t[0])
for u in res:
    p,e,span=res[u]; p0,e0,_=res['0.0']
    print(f"u_eff={u} (span {span:.0f}): fitted Gamma(-)={p[1]*1e3:.2f} [Bloch {-br[u]['1'][0][0]*1e3:.2f}]  Gamma(+)={p[4]*1e3:.2f} [Bloch {-br[u]['1'][1][0]*1e3:.2f}]  (1e-3)")
    for i,j,nm in ((0,0,'pumped (omega<0)'),(3,1,'other  (omega>0)')):
        r=p[i]/p0[i]; er=r*np.hypot(e[i]/p[i],e0[i]/p0[i]); pred=br['0.0']['1'][j][0]/br[u]['1'][j][0]
        print(f"     {nm}: T_eff/T = {r:.2f} ± {er:.2f}   predicted {pred:.2f}")
json.dump({u:[list(map(float,res[u][0])),list(map(float,res[u][1]))] for u in res},open("fitfree.json","w"))
