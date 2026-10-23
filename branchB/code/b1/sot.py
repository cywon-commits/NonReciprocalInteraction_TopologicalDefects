from skx import *
# Néel SkX + uniform SOT-type torque (leading-order reduction of nonreciprocal DMI delta D with n_ij = n):
#   field-like (DL-channel dD): T += -cF S x n ;  damping-like (FL-channel dD): T += -cD S x (S x n)
# co-moving frame with drift V: w1 = V, w2 = -alpha V   (u = 0)
class SOTModel(Model):
    def __init__(s,Lx,Ly,J,D,B,cF,cD,n): super().__init__(Lx,Ly,J,D,B); s.cF,s.cD,s.n=cF,cD,np.array(n,float)
    def rhs(s,S0,w1,w2,a,d=None,eps=0.0,k=(0.,0.)):
        S=S0 if d is None else S0+eps*d
        xp=s.nb(S0,d,eps,1,0,k); xm=s.nb(S0,d,eps,-1,0,k); yp=s.nb(S0,d,eps,1,1,k); ym=s.nb(S0,d,eps,-1,1,k)
        H=s.J*(xp+xm+yp+ym)+s.B*zh+s.D*(np.cross(xp,dx_)+np.cross(dx_,xm)+np.cross(yp,dy_)+np.cross(dy_,ym))
        gx=(xp-xm)/2; gy=(yp-ym)/2
        T=-np.cross(S,H)+w1[0]*gx+w1[1]*gy+np.cross(S,w2[0]*gx+w2[1]*gy)
        T=T-s.cF*np.cross(S,s.n)-s.cD*np.cross(S,np.cross(S,s.n))
        T=T-np.sum(T*S,-1)[...,None]*S
        return (T+a*np.cross(S,T))/(1+a*a)
def find_drift(m,S,a,iters=10,n=800):
    V=np.zeros(2)
    for _ in range(iters):
        f0=m.rhs(S,V,-a*V,a).ravel()
        A=np.stack([(m.rhs(S,V+e,-a*(V+e),a).ravel()-f0)/1e-3 for e in (np.array([1e-3,0]),np.array([0,1e-3]))],1)
        V=V+np.linalg.lstsq(A,-f0,rcond=None)[0]
        S=rk4(m,S,V,-a*V,a,0.1,n)
    return S,V,np.abs(m.rhs(S,V,-a*V,a)).max()
