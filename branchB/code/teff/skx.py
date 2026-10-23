import numpy as np
# 2D square lattice, |S|=1, gamma=1, interfacial (Neel) DMI, field B along z.
# Moving-frame (velocity V) Gilbert equation:
#   dS/dt = -S x H + a S x dS/dt + (w1.grad)S + S x (w2.grad)S ,  w1=V-u , w2=b*u-a*V
# Arrays shape (Lx,Ly,3). Bloch phases (kx,ky) are applied only to the perturbation part.
zh=np.array([0,0,1.]); xh=np.array([1.,0,0]); yh=np.array([0,1.,0])
dx_=yh; dy_=-xh            # DMI vectors z x e_ij for bonds +x, +y

class Model:
    def __init__(s,Lx,Ly,J,D,B):
        s.Lx,s.Ly,s.J,s.D,s.B=Lx,Ly,J,D,B
    def nb(s,S0,d,eps,shift,axis,k):
        # neighbour S_{i+shift e_axis} of S0+eps*d, Bloch phase on d
        out0=np.roll(S0,-shift,axis)
        if d is None: return out0
        dd=np.roll(d,-shift,axis).astype(complex)
        L=S0.shape[axis]; ph=np.exp(1j*k[axis]*L)
        idx=[slice(None)]*3
        if shift==1: idx[axis]=-1; dd[tuple(idx)]*=ph
        else:        idx[axis]=0;  dd[tuple(idx)]*=np.conj(ph)
        return out0+eps*dd
    def rhs(s,S0,w1,w2,a,d=None,eps=0.0,k=(0.,0.)):
        S=S0 if d is None else S0+eps*d
        xp=s.nb(S0,d,eps,1,0,k); xm=s.nb(S0,d,eps,-1,0,k)
        yp=s.nb(S0,d,eps,1,1,k); ym=s.nb(S0,d,eps,-1,1,k)
        H=s.J*(xp+xm+yp+ym)+s.B*zh \
          +s.D*(np.cross(xp,dx_)+np.cross(dx_,xm)+np.cross(yp,dy_)+np.cross(dy_,ym))
        gx=(xp-xm)/2; gy=(yp-ym)/2
        T=-np.cross(S,H)+w1[0]*gx+w1[1]*gy+np.cross(S,w2[0]*gx+w2[1]*gy)
        T=T-np.sum(T*S,-1)[...,None]*S
        return (T+a*np.cross(S,T))/(1+a*a)
    def energy_grad_ok(s): pass

def rk4(m,S,w1,w2,a,dt,n):
    for _ in range(n):
        k1=m.rhs(S,w1,w2,a); k2=m.rhs(S+dt/2*k1,w1,w2,a)
        k3=m.rhs(S+dt/2*k2,w1,w2,a); k4=m.rhs(S+dt*k3,w1,w2,a)
        S=S+dt/6*(k1+2*k2+2*k3+k4); S/=np.linalg.norm(S,axis=-1)[...,None]
    return S

def GD(S):
    gx=(np.roll(S,-1,0)-np.roll(S,1,0))/2; gy=(np.roll(S,-1,1)-np.roll(S,1,1))/2
    G=np.sum(S*np.cross(gx,gy))                      # G_xy
    Dxx=np.sum(gx*gx); Dyy=np.sum(gy*gy); Dxy=np.sum(gx*gy)
    Gm=np.array([[0,G],[-G,0]]); Dm=np.array([[Dxx,Dxy],[Dxy,Dyy]])
    return Gm,Dm,G/(4*np.pi)

def find_V(m,S,u,a,b,iters=12,dt=0.1,n=1500):
    # Thiele projection (exact for the stationary texture): G w1 + D w2 = 0
    for it in range(iters):
        Gm,Dm,_=GD(S)
        V=np.linalg.solve(Gm-a*Dm, Gm@u-b*Dm@u)
        w1=V-u; w2=b*u-a*V
        S=rk4(m,S,w1,w2,a,dt,n)
    r=np.abs(m.rhs(S,w1,w2,a)).max()
    return S,V,w1,w2,r

def basis(S):
    a=np.where((np.abs(S[...,0])<0.9)[...,None],xh,yh)
    e1=np.cross(S,a); e1/=np.linalg.norm(e1,axis=-1)[...,None]
    e2=np.cross(S,e1); return e1,e2

def bloch_jac(m,S,w1,w2,a,k,eps=1e-6):
    e1,e2=basis(S); N=S.shape[0]*S.shape[1]
    E=[e1.reshape(N,3),e2.reshape(N,3)]
    Jm=np.zeros((2*N,2*N),complex)
    for c in range(2):
        for j in range(N):
            d=np.zeros((N,3)); d[j]=E[c][j]; d=d.reshape(S.shape)
            f=(m.rhs(S,w1,w2,a,d,eps,k)-m.rhs(S,w1,w2,a,d,-eps,k))/(2*eps)
            f=f.reshape(N,3)
            Jm[:N,c*N+j]=np.sum(f*E[0],1); Jm[N:,c*N+j]=np.sum(f*E[1],1)
    return Jm
