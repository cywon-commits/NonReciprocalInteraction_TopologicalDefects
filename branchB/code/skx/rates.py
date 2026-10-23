from skx import *
J=1.0; D=0.4; a=0.3; b=0.6
m=Model(19,33,J,D,0.5*D*D/J); S0=np.load("S_B0.5.npy")
ks=[(2*np.pi/190*n,0.) for n in (1,2,3)]+[(0.,2*np.pi/165*n) for n in (1,2)]+[(2*np.pi/190,2*np.pi/165)]
for ue in [0.1,0.2,0.4]:
    S,V,w1,w2,r=find_V(m,S0.copy(),np.array([ue,0.]),a,b,iters=6,n=800)
    rr=[]
    for k in ks:
        lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k)); lam=lam[np.abs(lam)>1e-9]; rr.append(lam.real.max())
    print(f"u_eff={ue}: max growth over allowed k = {max(rr):.2e} at k={ks[int(np.argmax(rr))]}  (all: {np.round(rr,5)})",flush=True)
