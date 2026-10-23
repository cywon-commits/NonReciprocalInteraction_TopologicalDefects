import numpy as np, glob, re
from scipy.spatial import Voronoi
from scipy.ndimage import map_coordinates
def dens(S):
    gx=(np.roll(S,-1,0)-np.roll(S,1,0))/2; gy=(np.roll(S,-1,1)-np.roll(S,1,1))/2
    q=np.sum(S*np.cross(gx,gy),-1); g=np.sum(gx*gx+gy*gy,-1); return q,g
def centers(S):
    Lx,Ly=S.shape[:2]; mz=S[...,2]; c=[]
    for i in range(Lx):
        for j in range(Ly):
            v=mz[i,j]
            if v<0 and all(v<mz[(i+di)%Lx,(j+dj)%Ly] for di in (-1,0,1) for dj in (-1,0,1) if (di,dj)!=(0,0)):
                # sub-grid refinement: weighted centroid of (1-mz) in 5x5 box
                ii,jj=np.meshgrid(np.arange(-3,4),np.arange(-3,4),indexing='ij')
                w=np.clip(-mz[(i+ii)%Lx,(j+jj)%Ly],0,None)
                c.append([i+(w*ii).sum()/w.sum(), j+(w*jj).sum()/w.sum()])
    return np.array(c)
def face_avg(S,f):
    Lx,Ly=S.shape[:2]; C=centers(S)
    imgs=np.array([C+[dx*Lx,dy*Ly] for dx in (-1,0,1) for dy in (-1,0,1)]).reshape(-1,2)
    vor=Voronoi(imgs); vals=[]; lens=[]
    for (p,q),rv in zip(vor.ridge_points,vor.ridge_vertices):
        if -1 in rv: continue
        if not (p<len(C) or q<len(C)): continue      # faces touching an original cell
        A,B=vor.vertices[rv[0]],vor.vertices[rv[1]]; L=np.linalg.norm(B-A)
        t=np.linspace(0,1,60); pts=A[None,:]+t[:,None]*(B-A)[None,:]
        v=map_coordinates(f,[pts[:,0],pts[:,1]],order=1,mode='grid-wrap')
        vals.append(v.mean()*L); lens.append(L)
    return np.sum(vals)/np.sum(lens)
if __name__=="__main__":
    rows=[l.split() for l in open("kgrid.txt")]
    print(" a     B   kt_num  kt_geo | kP_num  kP_geo")
    for r in rows:
        Lx,bb=int(r[0]),r[2]; S=np.load(f"Sgrid_{Lx}_{bb}.npy")
        q,g=dens(S)
        kt=face_avg(S,q)/q.mean(); kp=face_avg(S,g)/g.mean()
        kn=float(r[-3].split('=')[1]); kpn=float(r[-2].split('=')[1])
        print(f"{Lx:3d} {bb:>5} {kn:7.3f} {kt:7.3f} | {kpn:7.3f} {kp:7.3f}")
