import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(4,1,figsize=(12,9),sharex=True)
cols={'u015':'#2a78d6','u020':'#eb6834','u030':'#1baf7a'}
for tag,c in cols.items():
    try: r=np.load(f"ts_{tag}.npy")
    except: continue
    t,N,P6,p6l,fd,s1,s2,s3,E=r.T
    ax[0].plot(t,P6,color=c,lw=1,label=tag); ax[1].plot(t,fd,color=c,lw=1); ax[2].semilogy(t,s1,color=c,lw=1); ax[3].plot(t,(E-E.min())/100,color=c,lw=1)
ax[0].set_ylabel('|Ψ6|'); ax[1].set_ylabel('f_def'); ax[2].set_ylabel('sideband k1'); ax[3].set_ylabel('E − E_min (per skyrmion)'); ax[3].set_xlabel('t'); ax[0].legend()
plt.tight_layout(); plt.savefig("ts_T0.png",dpi=110)
