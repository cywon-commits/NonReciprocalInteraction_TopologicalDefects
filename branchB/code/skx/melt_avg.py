import numpy as np, glob, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from melt import analyse, sk_centers
cols={'big010':'#2a78d6','big020':'#eb6834','big040':'#1baf7a'}; lab={'big010':'u_eff = 0.1','big020':'u_eff = 0.2','big040':'u_eff = 0.4'}
res={}
for tag in cols:
    fs=sorted(glob.glob(f"snap_{tag}/S_*.npy")); ts=np.array([int(f.split('_')[-1][:-4]) for f in fs])
    TS=[];G6=[];GG=[];GR=[]
    for f,t in zip(fs,ts):
        A=analyse(np.load(f).astype(float)); TS.append((t,A['N'],A['Psi6'],A['psi6loc'],A['fdef'],A['Smax_overN']))
        if t>=ts.max()-3000: G6.append(A['g6']); GG.append(A['gG']); GR.append(A['gr'])
    res[tag]=dict(ts=np.array(TS),rc=A['rc'],g6=np.nanmean(G6,0),gG=np.nanmean(GG,0),gr=np.nanmean(GR,0),a0=A['a0'],last=fs[-1])
    T=res[tag]['ts']; late=T[T[:,0]>=ts.max()-3000]
    print(f"{tag}: t_end={ts.max()} N={int(T[-1,1])}  late avg: Psi6={late[:,2].mean():.3f} |psi6|loc={late[:,3].mean():.3f} f_def={late[:,4].mean():.3f} S(G)/N={late[:,5].mean():.3f}")
    rc,g6,gG,a0=res[tag]['rc'],res[tag]['g6'],res[tag]['gG'],res[tag]['a0']
    for nm,g in (('g6',g6),('gG',gG)):
        m=(rc>0.8*a0)&(rc<4.2*a0)&(g>0.02)
        if m.sum()>=3:
            p=np.polyfit(rc[m],np.log(g[m]),1); print(f"   {nm}: xi = {-1/p[0]:.1f} sites = {-1/p[0]/a0:.2f} a0  (fit range {rc[m].min():.0f}-{rc[m].max():.0f})" if p[0]<0 else f"   {nm}: no decay")
np.save("melt_res.npy",res,allow_pickle=True)
fig=plt.figure(figsize=(13,7.2)); gs=fig.add_gridspec(2,3,height_ratios=[1,1])
for i,tag in enumerate(cols):
    ax=fig.add_subplot(gs[0,i]); S=np.load(res[tag]['last']).astype(float)
    ax.imshow(S[...,2].T,origin='lower',cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_title(f"{lab[tag]},  t = {int(res[tag]['ts'][-1,0])}",fontsize=10); ax.set_xticks([]); ax.set_yticks([])
ax=fig.add_subplot(gs[1,0])
for tag in cols:
    T=res[tag]['ts']; ax.plot(T[:,0],T[:,2],color=cols[tag],lw=2,label=lab[tag])
ax.set_xlabel('t'); ax.set_ylabel('global hexatic order |Ψ6|'); ax.legend(frameon=False,fontsize=9)
ax=fig.add_subplot(gs[1,1])
for tag in cols:
    T=res[tag]['ts']; ax.plot(T[:,0],T[:,4],color=cols[tag],lw=2,label=lab[tag])
ax.set_xlabel('t'); ax.set_ylabel('fraction with coordination ≠ 6')
ax=fig.add_subplot(gs[1,2])
for tag in cols:
    r=res[tag]; m=r['g6']>0; ax.semilogy(r['rc'][m]/r['a0'],r['g6'][m],'-',color=cols[tag],lw=2,label='g6 '+lab[tag])
    m=r['gG']>0; ax.semilogy(r['rc'][m]/r['a0'],r['gG'][m],'--',color=cols[tag],lw=1.2)
ax.set_xlabel('r / a0'); ax.set_ylabel('g6 (solid), g_G (dashed)'); ax.set_ylim(5e-3,1.5)
for a in fig.axes[3:]: a.grid(alpha=0.25,lw=0.6); a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/skx_landau/melting.png",dpi=130)
