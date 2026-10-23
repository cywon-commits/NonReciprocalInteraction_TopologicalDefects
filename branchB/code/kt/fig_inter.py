import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cols={'u015':'#2a78d6','u020':'#eb6834','u030':'#1baf7a'}; lab={'u015':'u_eff = 0.15','u020':'u_eff = 0.2','u030':'u_eff = 0.3'}
g={'u015':8.36e-4,'u020':1.39e-3,'u030':2.78e-3}
fig=plt.figure(figsize=(13,7.5)); gs=fig.add_gridspec(2,3)
ax=fig.add_subplot(gs[0,:])
for tag,c in cols.items():
    r=np.load(f"ts_{tag}.npy"); ax.plot(r[:,0],r[:,2],color=c,lw=1,label=lab[tag])
r=np.load("ts_anneal015.npy"); ax.plot(80000+r[:,0],r[:,2],color='#8a8a85',lw=1.5,label='current switched off at t = 80000 (from u_eff = 0.15)')
ax.set_xlabel('t'); ax.set_ylabel('global hexatic order |Ψ6|'); ax.legend(frameon=False,fontsize=9,ncol=4); ax.set_xlim(0,110000)
ax=fig.add_subplot(gs[1,0]); R=np.load("resp_u015snap.npy"); P0,f0,Pon,fon,Poff,foff=R.T
ax.scatter(P0,Pon-P0,color='#eb6834',s=30,edgecolor='white',label='current on'); ax.scatter(P0,Poff-P0,color='#2a78d6',s=30,edgecolor='white',label='current off')
ax.axhline(0,color='#8a8a85',lw=0.8); ax.set_xlabel('initial |Ψ6|'); ax.set_ylabel('Δ|Ψ6| after 800'); ax.legend(frameon=False,fontsize=9)
ax=fig.add_subplot(gs[1,1])
taus=[]
for tag in g:
    r=np.load(f"ts_{tag}.npy"); t,P6=r[:,0],r[:,2]; m=t>(16000 if tag!='u030' else 6000); x=P6[m]-P6[m].mean(); n=len(x)
    f=np.fft.rfft(np.concatenate([x,np.zeros(n)])); ac=np.fft.irfft(f*f.conj())[:n]/np.arange(n,0,-1); ac/=ac[0]; taus.append(np.argmax(ac<np.exp(-1))*25)
ax.plot([1/g[k] for k in g],taus,'o',ms=9,color='#1baf7a',mec='white',mew=1.5); xx=np.linspace(0,1300,10); ax.plot(xx,xx,'--',color='#8a8a85',lw=1,label='τ = 1/γ_max')
ax.set_xlabel('1/γ_max (linear Landau growth)'); ax.set_ylabel('correlation time of |Ψ6|'); ax.legend(frameon=False,fontsize=9)
ax=fig.add_subplot(gs[1,2])
for tag,c in cols.items():
    r=np.load(f"ts_{tag}.npy"); m=r[:,0]>(16000 if tag!='u030' else 6000); ax.hist(r[m,2],bins=np.linspace(0,0.8,33),color=c,alpha=0.6,density=True,label=lab[tag])
ax.set_xlabel('|Ψ6|'); ax.set_ylabel('probability density'); ax.legend(frameon=False,fontsize=9)
for a in fig.axes: a.grid(alpha=0.25,lw=0.6); a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/skx_landau/intermittency.png",dpi=130)
