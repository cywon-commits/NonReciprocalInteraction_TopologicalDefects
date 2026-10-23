import numpy as np, re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from homog import sigma_eff, dens
cols=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']
rows=[]
for l in open("kgrid.txt"):
    r=l.split(); Lx=int(r[0]); B=float(r[2])
    res=float(re.search(r"resid=([\de.+-]+)",l).group(1)); P=float(re.search(r"P=([-\d.]+)",l).group(1))
    if res>1e-8 or not (1.3<P<2.5): continue
    kt=float(re.search(r"kappa_tau=([\d.]+)",l).group(1)); kp=float(re.search(r"kappa_P=([\d.]+)",l).group(1))
    S=np.load(f"Sgrid_{Lx}_{r[2]}.npy"); sx,sy,q=dens(S); s=sx+sy
    rows.append((Lx,B,kt,kp,sigma_eff(s)/s.mean()))
rows=np.array(rows); np.savetxt("kappa_table.txt",rows,fmt="%.4f",header="a B kappa_tau kappa_P scalar_hom")
C=np.load("/home/claude/csl/csl_curve.npy")
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
a0=ax[0]; a0.plot(C[0]/(np.pi**2*0.09/16),C[2],color='#8a8a85',lw=2,label='closed form  1/(<ψ0²><ψ0⁻²>)')
a0.plot(C[0]/(np.pi**2*0.09/16),C[1],'o',ms=8,color='#2a78d6',mec='white',mew=1.5,label='Bloch eigenvalue (numerical)')
a0.set_xlabel('B / B_c  (B_c = π²D²/16J)'); a0.set_ylabel('Doppler efficiency λ'); a0.set_title('1D chiral soliton lattice (D = 0.3, period 40, K⊥ = 5)',fontsize=10)
a0.legend(frameon=False,fontsize=9); a0.set_ylim(0,1.05)
a1=ax[1]
for i,a in enumerate(sorted(set(rows[:,0]))):
    m=rows[:,0]==a; d=rows[m]; d=d[np.argsort(d[:,1])]
    a1.plot(d[:,1],d[:,2],'-o',color=cols[i],lw=2,ms=7,mec='white',mew=1.5,label=f'a = {int(a)}')
    a1.plot(d[:,1],d[:,4],'--',color=cols[i],lw=1.2,alpha=0.7)
a1.set_yscale('log'); a1.set_xlabel('B / (D²/J)'); a1.set_ylabel('κ_τ  (solid)   scalar estimate (dashed)')
a1.set_title('Néel SkX: reactive Doppler overlap κ_τ',fontsize=10); a1.legend(frameon=False,fontsize=9,ncol=2)
for x in ax: x.grid(alpha=0.25,lw=0.6); x.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/skx_landau/kappa_maps.png",dpi=140)
print(rows)
