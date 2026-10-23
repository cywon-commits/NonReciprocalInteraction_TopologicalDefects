import numpy as np, glob, sys
from scipy import ndimage
from scipy.spatial import Delaunay
def sk_centers(S):
    mz=S[...,2]; Lx,Ly=mz.shape
    lab,n=ndimage.label(mz<0,structure=np.ones((3,3)))
    parent=list(range(n+1))
    def f(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for A,B in [(lab[0,:],lab[-1,:]),(lab[:,0],lab[:,-1])]:
        for a,b in zip(A,B):
            if a and b: parent[f(a)]=f(b)
    for dA,dB in [((0,0),(-1,-1)),((0,-1),(-1,0))]:
        a,b=lab[dA],lab[dB]
        if a and b: parent[f(a)]=f(b)
    roots=np.array([f(i) for i in range(n+1)]); lab2=roots[lab]
    X,Y=np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij'); w=-mz
    C=[]
    for r in np.unique(lab2[lab2>0]):
        msk=lab2==r; ww=w[msk]
        if msk.sum()<4: continue
        cx=np.angle(np.sum(ww*np.exp(2j*np.pi*X[msk]/Lx)))*Lx/(2*np.pi)%Lx
        cy=np.angle(np.sum(ww*np.exp(2j*np.pi*Y[msk]/Ly)))*Ly/(2*np.pi)%Ly
        C.append([cx,cy,msk.sum()])
    return np.array(C)
def analyse(S, nb=40):
    Lx,Ly=S.shape[:2]; C=sk_centers(S); P=C[:,:2]; N=len(P)
    imgs=np.concatenate([P+[dx*Lx,dy*Ly] for dx in (-1,0,1) for dy in (-1,0,1)]); own=np.tile(np.arange(N),9)
    tri=Delaunay(imgs); indptr,ind=tri.vertex_neighbor_vertices
    base=4*N  # (dx,dy)=(0,0) block index
    psi=np.zeros(N,complex); z=np.zeros(N,int)
    for i in range(N):
        nb_=ind[indptr[base+i]:indptr[base+i+1]]; d=imgs[nb_]-P[i]
        z[i]=len(nb_); psi[i]=np.mean(np.exp(6j*np.arctan2(d[:,1],d[:,0])))
    # pair statistics (minimum image)
    D=P[:,None,:]-P[None,:,:]; D[...,0]-=Lx*np.round(D[...,0]/Lx); D[...,1]-=Ly*np.round(D[...,1]/Ly)
    R=np.hypot(D[...,0],D[...,1]); iu=np.triu_indices(N,1); r=R[iu]
    rmax=min(Lx,Ly)/2; bins=np.linspace(0,rmax,nb+1); rc=(bins[1:]+bins[:-1])/2
    g6=np.zeros(nb); cnt=np.zeros(nb)
    prod=(psi[:,None]*psi[None,:].conj()).real[iu]
    k=np.digitize(r,bins)-1; ok=(k>=0)&(k<nb)
    np.add.at(g6,k[ok],prod[ok]); np.add.at(cnt,k[ok],1)
    g6=np.where(cnt>0,g6/np.maximum(cnt,1),np.nan)
    gr=cnt/(2*np.pi*rc*(bins[1]-bins[0])*N*(N-1)/2/(Lx*Ly))
    # translational: max structure factor near first shell
    a0=np.sqrt(2*Lx*Ly/(np.sqrt(3)*N)); G0=4*np.pi/(np.sqrt(3)*a0)
    best=(0,None)
    for gm in np.linspace(0.85,1.15,31)*G0:
        th=np.linspace(0,np.pi/3,121,endpoint=False)
        q=np.stack([gm*np.cos(th),gm*np.sin(th)],1)
        Sq=np.abs(np.exp(1j*P@q.T).sum(0))**2/N
        j=np.argmax(Sq)
        if Sq[j]>best[0]: best=(Sq[j],q[j])
    Smax,Gv=best
    # translational correlation g_G(r) with the best G (and its 60/120 deg partners)
    gG=np.zeros(nb)
    for t in (0,np.pi/3,2*np.pi/3):
        Rm=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]); G=Rm@Gv
        c=np.cos(D[...,0]*G[0]+D[...,1]*G[1])[iu]; tmp=np.zeros(nb); np.add.at(tmp,k[ok],c[ok]); gG+=tmp/np.maximum(cnt,1)/3
    return dict(N=N,Psi6=abs(psi.mean()),psi6loc=np.mean(np.abs(psi)),fdef=np.mean(z!=6),Smax_overN=Smax/N,
                rc=rc,g6=g6,gr=gr,gG=gG,a0=a0)
def xi_fit(rc,g,a0,rmin=1.0):
    m=(rc>rmin*a0)&np.isfinite(g)&(g>1e-3)
    if m.sum()<3: return np.nan
    p=np.polyfit(rc[m],np.log(g[m]),1); return -1/p[0] if p[0]<0 else np.inf
if __name__=="__main__":
    tag=sys.argv[1]; files=sorted(glob.glob(f"snap_{tag}/S_*.npy"))
    for fn in files[::int(sys.argv[2]) if len(sys.argv)>2 else 1]:
        S=np.load(fn).astype(float); A=analyse(S)
        print(f"{fn.split('_')[-1][:-4]} N={A['N']} Psi6={A['Psi6']:.3f} |psi6|loc={A['psi6loc']:.3f} f_def={A['fdef']:.3f} S(G)/N={A['Smax_overN']:.3f} xi6={xi_fit(A['rc'],A['g6'],A['a0']):.1f} xiT={xi_fit(A['rc'],A['gG'],A['a0']):.1f}",flush=True)
