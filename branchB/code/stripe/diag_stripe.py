import numpy as np
# Neel cycloid with q along the square-lattice diagonal (stable orientation on this lattice).
# Pattern depends on i+j only. Cell = P sites n (positions (n,0)); exact Bloch k for any k.
zh=np.array([0,0,1.]); xh=np.array([1.,0,0]); yh=np.array([0,1.,0])
dvec={(1,0):yh,(0,1):-xh}          # interfacial DMI z x e_ij
class DModel:
    def __init__(s,P,J,D,B=0.0):
        s.P,s.J,s.D,s.B=P,J,D,B
        n=np.arange(P); s.map={}
        for (dx,dy) in [(1,0),(-1,0),(0,1),(0,-1)]:
            nn=n+dx+dy; tgt=nn%P
            T=np.stack([n+dx-tgt, np.full(P,dy)],1)   # lattice translation to the actual neighbour
            s.map[(dx,dy)]=(tgt,T)
    def nb(s,S0,d,eps,dxy,k):
        tgt,T=s.map[dxy]; out=S0[tgt]
        if d is None: return out
        ph=np.exp(1j*(T@np.array(k)))
        return out+eps*d[tgt]*ph[:,None]
    def rhs(s,S0,w1,w2,a,d=None,eps=0.0,k=(0.,0.)):
        S=S0 if d is None else S0+eps*d
        xp=s.nb(S0,d,eps,(1,0),k); xm=s.nb(S0,d,eps,(-1,0),k)
        yp=s.nb(S0,d,eps,(0,1),k); ym=s.nb(S0,d,eps,(0,-1),k)
        dx_,dy_=dvec[(1,0)],dvec[(0,1)]
        H=s.J*(xp+xm+yp+ym)+s.B*zh+s.D*(np.cross(xp,dx_)+np.cross(dx_,xm)+np.cross(yp,dy_)+np.cross(dy_,ym))
        gx=(xp-xm)/2; gy=(yp-ym)/2
        T=-np.cross(S,H)+w1[0]*gx+w1[1]*gy+np.cross(S,w2[0]*gx+w2[1]*gy)
        T=T-np.sum(T*S,-1)[:,None]*S
        return (T+a*np.cross(S,T))/(1+a*a)
def rk4(m,S,w1,w2,a,dt,n):
    for _ in range(n):
        k1=m.rhs(S,w1,w2,a); k2=m.rhs(S+dt/2*k1,w1,w2,a); k3=m.rhs(S+dt/2*k2,w1,w2,a); k4=m.rhs(S+dt*k3,w1,w2,a)
        S=S+dt/6*(k1+2*k2+2*k3+k4); S/=np.linalg.norm(S,axis=-1)[:,None]
    return S
def jac(m,S,w1,w2,a,k,eps=1e-6):
    P=len(S); A=np.where((np.abs(S[:,0])<0.9)[:,None],xh,yh)
    e1=np.cross(S,A); e1/=np.linalg.norm(e1,axis=1)[:,None]; e2=np.cross(S,e1); E=[e1,e2]
    Jm=np.zeros((2*P,2*P),complex)
    for c in range(2):
        for j in range(P):
            d=np.zeros((P,3)); d[j]=E[c][j]
            f=(m.rhs(S,w1,w2,a,d,eps,k)-m.rhs(S,w1,w2,a,d,-eps,k))/(2*eps)
            Jm[:P,c*P+j]=np.sum(f*e1,1); Jm[P:,c*P+j]=np.sum(f*e2,1)
    return np.linalg.eigvals(Jm)
J=1.0; P=30; th=2*np.pi/P; D=np.sqrt(2)*J*np.tan(th)
m=DModel(P,J,D)
qh=np.array([1,1.])/np.sqrt(2); eh=np.array([-1,1.])/np.sqrt(2)   # stripe normal, stripe direction
def cycloid():
    n=np.arange(P); S=np.zeros((P,3))
    S[:,0]=qh[0]*np.sin(th*n); S[:,1]=qh[1]*np.sin(th*n); S[:,2]=np.cos(th*n); return S
def steady(u,phi,a,b,nstep=20000,dt=0.1):
    ang=np.pi/4+phi; uv=u*np.array([np.cos(ang),np.sin(ang)])   # phi = angle between u and q
    w1=(b/a-1)*uv; w2=np.zeros(2)
    S=rk4(m,cycloid(),w1,w2,a,dt,nstep)
    return S,w1,w2,np.abs(m.rhs(S,w1,w2,a)).max()
def low(lam,n=2): return lam[np.argsort(np.abs(lam))[:n]]
