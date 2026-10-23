import numpy as np, glob, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11.5,4.2))
for ue,c in (('0.0','#2a78d6'),('0.03','#eb6834')):
    for init,mk,ls in (('cry','o','-'),('liq','s','--')):
        Ts=[0.17,0.19,0.21,0.23]; P=[];F=[]
        for T in Ts:
            r=np.load(f"kt2_u{ue}_T{T}_{init}.npy"); h=r[len(r)//2:]; P.append(h[:,2].mean()); F.append(h[:,4].mean())
        lab=f"u_eff = {float(ue):g}, {'crystal' if init=='cry' else 'liquid'} start"
        ax[0].plot(Ts,P,ls,marker=mk,color=c,lw=1.8,ms=7,mec='white',label=lab); ax[1].plot(Ts,F,ls,marker=mk,color=c,lw=1.8,ms=7,mec='white')
ax[0].set_ylabel('|Ψ6| (second-half mean)'); ax[1].set_ylabel('fraction with coordination ≠ 6')
for a in ax: a.set_xlabel('k_B T / J'); a.grid(alpha=0.25,lw=0.6); a.spines[['top','right']].set_visible(False)
ax[0].legend(frameon=False,fontsize=8.5); ax[0].set_title('64 skyrmions, 25000 time units per point',fontsize=10)
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/skx_landau/kthny.png",dpi=130)
