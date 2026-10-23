import numpy as np, re
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import cg, spsolve
import scipy.sparse as sp
# Scalar homogenization: sigma(r) = Goldstone-mode density. kappa_hom = sigma_eff,xx / <sigma>
def sigma_eff(sig):
    Lx,Ly=sig.shape; N=Lx*Ly; idx=np.arange(N).reshape(Lx,Ly)
    # face conductivities (harmonic mean of neighbours)
    sxp=2*sig*np.roll(sig,-1,0)/(sig+np.roll(sig,-1,0)); syp=2*sig*np.roll(sig,-1,1)/(sig+np.roll(sig,-1,1))
    rows=[];cols=[];vals=[]
    def add(i,j,v): rows.append(i);cols.append(j);vals.append(v)
    I=idx.ravel(); Ixp=np.roll(idx,-1,0).ravel(); Iyp=np.roll(idx,-1,1).ravel()
    for (Ia,Ib,s) in [(I,Ixp,sxp.ravel()),(I,Iyp,syp.ravel())]:
        rows+= [Ia,Ib,Ia,Ib]; cols+=[Ia,Ib,Ib,Ia]; vals+=[s,s,-s,-s]
    A=sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N,N))
    # rhs: -div(sigma xhat) -> flux of unit gradient through x faces
    b=np.zeros(N); np.add.at(b,I,sxp.ravel()); np.add.at(b,Ixp,-sxp.ravel())
    A=A+sp.identity(N)*1e-12*sig.mean()
    g=spsolve(A.tocsc(),b)
    g=g.reshape(Lx,Ly)
    flux=sxp*(1+np.roll(g,-1,0)-g)
    return flux.mean()
def dens(S):
    gx=(np.roll(S,-1,0)-np.roll(S,1,0))/2; gy=(np.roll(S,-1,1)-np.roll(S,1,1))/2
    return np.sum(gx*gx,-1), np.sum(gy*gy,-1), np.sum(S*np.cross(gx,gy),-1)
if __name__=="__main__":
    print("  a     B | k_tau  k_P | hom(|dx m|^2) hom(|grad m|^2) hom(|q|) | harm(|grad|^2) geo(|grad|^2)")
    for l in open("kgrid.txt"):
        r=l.split(); Lx=int(r[0]); bb=r[2]
        if re.search(r"resid=(1e-03|2e-03|2e-06)",l): continue
        S=np.load(f"Sgrid_{Lx}_{bb}.npy"); sx,sy,q=dens(S); s=sx+sy
        kt=float(re.search(r"kappa_tau=([\d.]+)",l).group(1)); kp=float(re.search(r"kappa_P=([\d.]+)",l).group(1))
        hx=sigma_eff(sx)/sx.mean(); hs=sigma_eff(s)/s.mean(); hq=sigma_eff(np.abs(q)+1e-12)/np.abs(q).mean()
        harm=1/(s.mean()*np.mean(1/s)); geo=np.exp(np.mean(np.log(s)))/s.mean()
        print(f"{Lx:3d} {bb:>5} | {kt:.3f} {kp:.3f} | {hx:.3f} {hs:.3f} {hq:.3f} | {harm:.3f} {geo:.3f}",flush=True)
