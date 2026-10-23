import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rates=json.load(open("rates_T0.json")) if __import__('os').path.exists("rates_T0.json") else {}
tags={'u015':0.15,'u020':0.2,'u030':0.3}
for tag,u in tags.items():
    try: r=np.load(f"ts_{tag}.npy")
    except: continue
    t,N,P6,p6l,fd,s1,s2,s3,E=r.T
    # recrystallized episodes: maximal runs with f_def==0 lasting >= 500
    ep=[]; i=0
    while i<len(t):
        if fd[i]==0:
            j=i
            while j<len(t) and fd[j]==0: j+=1
            if t[j-1]-t[i]>=500 and t[i]>2500: ep.append((i,j))
            i=j
        else: i+=1
    print(f"{tag}: t_end={t[-1]:.0f}  N range {N.min():.0f}-{N.max():.0f}  defect-free episodes after melting: {len(ep)}  mean |Psi6|={P6[t>5000].mean():.2f}")
    for (i,j) in ep:
        seg=slice(i,j); best=[]
        for nm,s in (('k1',s1),('k2',s2),('k3',s3)):
            p=np.polyfit(t[seg],np.log(s[seg]),1); best.append(p[0])
        print(f"   episode t={t[i]:.0f}-{t[j-1]:.0f} (len {t[j-1]-t[i]:.0f}): peak Psi6={P6[seg].max():.2f}; sideband log-slopes k1,k2,k3 = {np.round(np.array(best)*1e3,2)} e-3   [linear theory {np.round(np.array(rates.get(str(u),[np.nan]*3))*1e3,2)} e-3]")
    # energy derivative by phase
    dE=np.gradient(E,t); liq=fd>0.15; cry=fd==0
    print(f"   dE/dt: liquid (f_def>0.15) {dE[liq&(t>3000)].mean():+.3f}   defect-free {dE[cry&(t>3000)].mean():+.3f}")
