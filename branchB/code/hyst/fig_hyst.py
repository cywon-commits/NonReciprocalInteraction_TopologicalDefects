import numpy as np, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C1,C2,INK,MUTED='#2a78d6','#eb6834','#222222','#8a8a85'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':MUTED,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED})
def chain(tag):
    fs=sorted(glob.glob(f"rec_{tag}_*.npy")); out=[]
    for f in fs:
        u=float(f.split('_')[-1][:-4]); r=np.load(f); out.append((u,np.median(r[len(r)*3//4:,1]),r[-1,4]))
    return np.array(sorted(out))
fig,ax=plt.subplots(1,3,figsize=(11,3.3))
up,dn=chain('sup'),chain('sdn')
a=ax[0]; a.plot(up[:,0],np.minimum(up[:,1],6),'o-',color=C1,lw=2,ms=6,label='increasing u')
a.plot(dn[:,0],dn[:,1],'s--',color=C2,lw=2,ms=6,label='decreasing u')
fx=[(0.096,0),(0.098,0.0),(0.099,0.069),(0.100,0.312),(0.105,1.186)]
a.plot([p[0] for p in fx],[p[1] for p in fx],'D',color=INK,ms=5,label='long fixed-u run')
a.axvline(0.0994,color=MUTED,lw=1,ls=':'); a.text(0.0255,4.4,'',color=MUTED)
a.set_xlabel('u_eff'); a.set_ylabel('deformation amplitude A'); a.set_title('(a) SkX, 4x1 cells: no hysteresis',fontsize=9,loc='left'); a.legend(frameon=False,fontsize=8)
a=ax[1]; st=[(0.100,0.312),(0.105,1.186),(0.110,dn[np.isclose(dn[:,0],0.11)][0,1])]
uc=0.0994; x=np.array([p[0]-uc for p in st]); y=np.array([p[1] for p in st])
a.plot(x,y**2,'o',color=C2,ms=7); xx=np.linspace(0,0.011,50); s=np.sum(x*y**2)/np.sum(x*x); a.plot(xx,s*xx,color=MUTED,lw=1,ls='--')
a.set_xlabel('u_eff - u_c   (u_c = 0.0994)'); a.set_ylabel('A²'); a.set_title('(b) SkX steady branch: A² roughly ∝ (u - u_c)',fontsize=9,loc='left')
hu,hd=chain('st_up'),chain('st_dn')
a=ax[2]; a.step(hu[:,0],hu[:,2],where='post',color=C1,lw=2,label='increasing u'); a.step(hd[:,0],hd[:,2],where='pre',color=C2,lw=2,ls='--',label='decreasing u')
for v,l,yy in [(0.0707,'u_L(n=3)',2.6),(0.1094,'u_L(n=2)',1.7),(0.1576,'u_unw(n=1)',0.7)]:
    a.axvline(v,color=MUTED,lw=1,ls=':'); a.text(v+0.003,yy,l,color=MUTED,fontsize=8,bbox=dict(fc='white',ec='none',pad=1))
a.set_xlabel('u_eff'); a.set_ylabel('helix winding n'); a.set_yticks([0,1,2,3]); a.set_title('(c) 1D helix: phase-slip staircase',fontsize=9,loc='left'); a.legend(frameon=False,fontsize=8,loc='lower left')
plt.tight_layout(); plt.savefig('hysteresis.png',dpi=160)
print(up[:,:2].round(4).tolist()); print(dn[:,:2].round(3).tolist()); print(hu[:,[0,2]].tolist()); print(hd[:,[0,2]].tolist())
