import numpy as np
# 1D chain, |S|=1, gamma=1. E = -J sum S_i.S_{i+1} - D sum x.(S_i x S_{i+1})
# LLG (Gilbert) + Zhang-Li (central difference):
#  dS/dt = -S x H + a S x dS/dt - u dS + b S x (u dS),  dS_i=(S_{i+1}-S_{i-1})/2
xh=np.array([1.,0,0])
def field(S,J,D):
    Sp=np.roll(S,-1,0); Sm=np.roll(S,1,0)
    return J*(Sp+Sm)+D*(np.cross(Sp,xh)+np.cross(xh,Sm))
def rhs(S,J,D,u,a,b,Om):
    H=field(S,J,D)
    dS=(np.roll(S,-1,0)-np.roll(S,1,0))/2
    T=-np.cross(S,H)-u*dS+b*u*np.cross(S,dS)
    T-= (np.sum(T*S,1))[:,None]*S       # keep tangent
    F=(T+a*np.cross(S,T))/(1+a*a)
    return F-Om*np.cross(xh,S)          # rotating frame about x
def cone(N,n,M,psi=0):
    th=2*np.pi*n/N; i=np.arange(N)
    r=np.sqrt(1-M*M)
    return np.stack([M*np.ones(N),r*np.cos(th*i+psi),r*np.sin(th*i+psi)],1),th
def predict(J,D,th,u,a,b):
    ueff=(b/a-1)*u
    G=J*np.cos(th)+D*np.sin(th)
    M=ueff*np.sin(th)/(2*(G-J))
    Om=-b*u*np.sin(th)/a
    return M,Om
def tangent(S):
    e1=np.cross(S,xh); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(S,e1)
    return e1,e2
def jac(S,args,eps=1e-6):
    N=len(S); e1,e2=tangent(S); E=[e1,e2]
    Jm=np.zeros((2*N,2*N))
    for c in range(2):
        for j in range(N):
            Sp=S.copy(); Sm=S.copy()
            Sp[j]+=eps*E[c][j]; Sp[j]/=np.linalg.norm(Sp[j])
            Sm[j]-=eps*E[c][j]; Sm[j]/=np.linalg.norm(Sm[j])
            d=(rhs(Sp,*args)-rhs(Sm,*args))/(2*eps)
            Jm[0:N,c*N+j]=np.sum(d*e1,1); Jm[N:,c*N+j]=np.sum(d*e2,1)
    return Jm
