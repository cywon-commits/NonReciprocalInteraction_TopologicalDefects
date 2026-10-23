import numpy as np
# 1D monoaxial chiral chain: E=-J S_i.S_i+1 - D x.(S_i x S_i+1) - B S_z + Kp S_x^2 ; spins rotate in the y-z plane (q || x).
xh=np.array([1.,0,0]); zh=np.array([0,0,1.])
class Chain:
    def __init__(s,N,J,D,B,Kp): s.N,s.J,s.D,s.B,s.Kp=N,J,D,B,Kp
    def nb(s,S0,d,eps,sh,k):
        o=np.roll(S0,-sh,0)
        if d is None: return o
        dd=np.roll(d,-sh,0).astype(complex)
        if sh==1: dd[-1]*=np.exp(1j*k*s.N)
        else: dd[0]*=np.exp(-1j*k*s.N)
        return o+eps*dd
    def rhs(s,S0,w1,w2,a,d=None,eps=0.0,k=0.0):
        S=S0 if d is None else S0+eps*d
        p=s.nb(S0,d,eps,1,k); m=s.nb(S0,d,eps,-1,k)
        H=s.J*(p+m)+s.D*(np.cross(p,xh)+np.cross(xh,m))+s.B*zh-2*s.Kp*S[:,0:1]*xh
        g=(p-m)/2
        T=-np.cross(S,H)+w1*g+np.cross(S,w2*g)
        T=T-np.sum(T*S,-1)[:,None]*S
        return (T+a*np.cross(S,T))/(1+a*a)
def rk4(m,S,w1,w2,a,dt,n):
    for _ in range(n):
        k1=m.rhs(S,w1,w2,a);k2=m.rhs(S+dt/2*k1,w1,w2,a);k3=m.rhs(S+dt/2*k2,w1,w2,a);k4=m.rhs(S+dt*k3,w1,w2,a)
        S=S+dt/6*(k1+2*k2+2*k3+k4); S/=np.linalg.norm(S,axis=1)[:,None]
    return S
def jac(m,S,w1,w2,a,k,eps=1e-6):
    N=len(S); A=np.where((np.abs(S[:,0])<0.9)[:,None],xh,np.array([0,1.,0]))
    e1=np.cross(S,A); e1/=np.linalg.norm(e1,axis=1)[:,None]; e2=np.cross(S,e1); E=[e1,e2]
    Jm=np.zeros((2*N,2*N),complex)
    for c in range(2):
        for j in range(N):
            d=np.zeros((N,3)); d[j]=E[c][j]
            f=(m.rhs(S,w1,w2,a,d,eps,k)-m.rhs(S,w1,w2,a,d,-eps,k))/(2*eps)
            Jm[:N,c*N+j]=np.sum(f*e1,1); Jm[N:,c*N+j]=np.sum(f*e2,1)
    return Jm
def relax(N,J,D,B,Kp):
    m=Chain(N,J,D,B,Kp); th=2*np.pi*np.arange(N)/N
    S=np.stack([0*th,np.sin(th),np.cos(th)],1)
    for _ in range(60):
        S=rk4(m,S,0,0,1.0,0.1,1000)
        if np.abs(m.rhs(S,0,0,1.0)).max()<1e-12: break
    return m,S
def phonon(m,S,w1,w2,a,k):
    lam=np.linalg.eigvals(jac(m,S,w1,w2,a,k)); lam=lam[np.argsort(np.abs(lam))[:2]]
    return lam[np.argmax(lam.imag)]
def kappa(m,S,k,e=1e-5):
    l0=phonon(m,S,0,0,0.0,k)
    lt=phonon(m,S,e,0,0.0,k)
    return -(lt.imag-l0.imag)/e/k, l0.imag/k**2
def formula(S):
    th=np.unwrap(np.arctan2(S[:,1],S[:,2])); N=len(S)
    tp=np.diff(np.concatenate([th,[th[0]+2*np.pi]]))       # bond angle increments
    psi=(tp+np.roll(tp,1))/2                              # site density
    return 1/(np.mean(psi**2)*np.mean(1/psi**2)), 1/(np.mean(tp**2)*np.mean(1/tp**2))
