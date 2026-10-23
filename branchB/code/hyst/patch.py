s=open('sweep.py').read()
if "rng(int(u_eff" not in s:
    s=s.replace("    u=u_eff/(b/a-1)\n","    u=u_eff/(b/a-1)\n    S=S+1e-6*np.random.default_rng(int(u_eff*1e4)).standard_normal(S.shape); S/=np.linalg.norm(S,axis=-1)[...,None]\n")
    s=s.replace("def topo(S):\n","def topo(S):\n    if sysname!='skx':\n        ph=np.arctan2(S[:,0,0],S[:,0,2]); d=np.angle(np.exp(1j*np.diff(np.r_[ph,ph[0]])));return float(d.sum()/(2*np.pi))\n")
open('sweep.py','w').write(s)
