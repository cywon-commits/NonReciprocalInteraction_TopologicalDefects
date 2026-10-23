import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fitfree import spectrum, L, seeds, br
amp=json.load(open("fitamp.json"))
fig,ax=plt.subplots(1,2,figsize=(11.5,4.3))
cols={'0.0':'#8a8a85','0.05':'#2a78d6','0.07':'#eb6834'}
for u in ['0.0','0.07']:
    d=np.load(f"traj_u{u}_T0.01_s{seeds[u]}.npy"); w,P=spectrum(d[:,1:],d[1,0].real-d[0,0].real)
    m=np.abs(w)<0.015; ax[0].semilogy(w[m]*1e3,P[m],color=cols[u],lw=1.4,label=f'simulation, u_eff = {u}')
    (Ga,wa),(Gb,wb)=[(-x[0],x[1]) for x in br[u]['1']]; c=amp[u][0]
    ax[0].semilogy(w[m]*1e3,c[0]*L(w[m],Ga,wa)+c[1]*L(w[m],Gb,wb)+c[2],'--',color=cols[u],lw=1.2)
ax[0].set_xlabel('frequency ω (10⁻³)'); ax[0].set_ylabel('displacement spectrum at k₁'); ax[0].legend(frameon=False,fontsize=9)
ax[0].set_title('k₁ = 2π/76 phonon: pumped branch (ω<0) sharpens and grows',fontsize=10)
G0=-br['0.0']['1'][0][0]; ust=0.098
uu=np.linspace(0,0.08,100); x=uu/ust
ax[1].plot(uu,1/(1-x),color='#eb6834',lw=2,label='T/(1−x): pumped branch')
ax[1].plot(uu,1/(1+x),color='#2a78d6',lw=2,label='T/(1+x): other branch')
for u in ['0.0','0.05','0.07']:
    c,e=np.array(amp[u][0]),np.array(amp[u][1]); c0,e0=np.array(amp['0.0'][0]),np.array(amp['0.0'][1])
    for i,col in ((0,'#eb6834'),(1,'#2a78d6')):
        r=c[i]/c0[i]; er=r*np.hypot(e[i]/c[i],e0[i]/c0[i])
        ax[1].errorbar(float(u),r,er,fmt='o',color=col,ms=8,mec='white',mew=1.5,capsize=3)
ax[1].set_xlabel('u_eff'); ax[1].set_ylabel('T_eff / T'); ax[1].set_title('Moving-Gibbs prediction vs stochastic LLG (T = 0.01)',fontsize=10)
ax[1].legend(frameon=False,fontsize=9); ax[1].set_ylim(0,5)
for a in ax: a.grid(alpha=0.25,lw=0.6); a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/skx_landau/teff_phonon.png",dpi=140)
