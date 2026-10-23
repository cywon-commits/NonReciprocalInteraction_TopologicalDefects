import numpy as np, json, os
from scipy.signal import welch
seeds={'0.0':2,'0.05':3,'0.07':1}; br=json.load(open("branches.json"))
def load(u): r=np.load(f"traj_u{u}_T0.01_s{seeds[u]}.npy"); return r[:,0].real, r[:,1:]
res={}
for u in seeds:
    if not os.path.exists(f"traj_u{u}_T0.01_s{seeds[u]}.npy"): continue
    t,z=load(u); dtau=t[1]-t[0]
    P=0; 
    for c in range(2):
        f,p=welch(z[:,c],fs=1/dtau,nperseg=4096,return_onesided=False,detrend=False); P=P+p
    o=np.argsort(f); f,P=f[o],P[o]; df=f[1]-f[0]
    neg=P[f<0].sum()*df; pos=P[f>0].sum()*df
    # block errors on the split
    nb=8; L=len(z)//nb; bn=[];bp=[]
    for i in range(nb):
        q=0
        for c in range(2):
            ff,pp=welch(z[i*L:(i+1)*L,c],fs=1/dtau,nperseg=2048,return_onesided=False,detrend=False); q=q+pp
        bn.append(q[ff<0].sum()*(ff[1]-ff[0])); bp.append(q[ff>0].sum()*(ff[1]-ff[0]))
    # peak frequencies (angular)
    wn=2*np.pi*f[f<0][np.argmax(P[f<0])]; wp=2*np.pi*f[f>0][np.argmax(P[f>0])]
    res[u]=(neg,np.std(bn)/np.sqrt(nb),pos,np.std(bp)/np.sqrt(nb),wn,wp,t[-1]-t[0])
for u in res:
    neg,en,pos,ep,wn,wp,span=res[u]; n0,_,p0,_,_,_,_=res['0.0']
    G0=[-x[0] for x in br['0.0']['1']]; Gu=[-x[0] for x in br[u]['1']]
    print(f"u_eff={u} span={span:.0f}: peaks omega=({wn*1e3:+.2f},{wp*1e3:+.2f})e-3 [Bloch ({br[u]['1'][0][1]*1e3:+.2f},{br[u]['1'][1][1]*1e3:+.2f})e-3]")
    print(f"   pumped branch (omega<0): ratio {neg/n0:.2f} ± {en/n0:.2f}   predicted Gamma0/Gamma = {G0[0]/Gu[0]:.2f}")
    print(f"   other  branch (omega>0): ratio {pos/p0:.2f} ± {ep/p0:.2f}   predicted Gamma0/Gamma = {G0[1]/Gu[1]:.2f}")
