from skx import *
J=1.0; D=0.4; bb=0.5; a=0.3; b=0.6
m=Model(19,33,J,D,bb*D*D/J); S0=np.load(f"S_B{bb}.npy")
k=(2*np.pi/76,0.)
for ue in [0.07,0.09,0.115,0.14]:
    u=ue/(b/a-1); S,V,w1,w2,r=find_V(m,S0.copy(),np.array([u,0.]),a,b,iters=6,n=800)
    lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k)); lam=lam[np.abs(lam)>1e-9]
    print(f"u_eff={ue}: Bloch growth rate={lam.real.max():+.3e}  resid={r:.0e}",flush=True)
