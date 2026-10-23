import numpy as np
from scipy.spatial import Delaunay
from melt import sk_centers
def energy(S,J=1.0,D=0.4,B=0.08):
    xp=np.roll(S,-1,0); yp=np.roll(S,-1,1)
    e=-J*np.sum(S*xp)-J*np.sum(S*yp)
    e-=D*np.sum(np.cross(S,xp)[...,1]); e+=D*np.sum(np.cross(S,yp)[...,0]); e-=B*np.sum(S[...,2])
    return e
def order(S):
    Lx,Ly=S.shape[:2]; C=sk_centers(S); P=C[:,:2]; N=len(P)
    imgs=np.concatenate([P+[dx*Lx,dy*Ly] for dx in (-1,0,1) for dy in (-1,0,1)])
    tri=Delaunay(imgs); ip,ind=tri.vertex_neighbor_vertices; base=4*N
    psi=np.zeros(N,complex); z=np.zeros(N,int)
    for i in range(N):
        nb=ind[ip[base+i]:ip[base+i+1]]; d=imgs[nb]-P[i]; z[i]=len(nb); psi[i]=np.mean(np.exp(6j*np.arctan2(d[:,1],d[:,0])))
    return N,abs(psi.mean()),np.mean(np.abs(psi)),np.mean(z!=6),P
def sideband(S,Gs):
    F=np.fft.fft2(S[...,2]); Lx,Ly=F.shape
    den=sum(abs(F[g[0]%Lx,g[1]%Ly])**2 for g in Gs)
    out=[]
    for n in (1,2,3):
        out.append(np.sqrt(max(sum(abs(F[(g[0]+dx)%Lx,(g[1]+dy)%Ly])**2 for g in Gs) for dx,dy in [(n,0),(-n,0)])/den))
    return out
