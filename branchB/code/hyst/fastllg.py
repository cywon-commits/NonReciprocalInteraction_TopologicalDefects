import numpy as np
from numba import njit, prange
# f(S) for: -S x (H+xi) + (w1.grad)S + S x (w2.grad)S, Gilbert->LL with alpha. Neel DMI, square lattice, periodic.
@njit(parallel=True,fastmath=True,cache=True)
def rhs(S,xi,out,J,D,B,a,w1x,w1y,w2x,w2y):
    Lx,Ly=S.shape[0],S.shape[1]; c=1.0/(1+a*a)
    for i in prange(Lx):
        ip=(i+1)%Lx; im=(i-1)%Lx
        for j in range(Ly):
            jp=(j+1)%Ly; jm=(j-1)%Ly
            sx,sy,sz=S[i,j,0],S[i,j,1],S[i,j,2]
            xpx,xpy,xpz=S[ip,j,0],S[ip,j,1],S[ip,j,2]; xmx,xmy,xmz=S[im,j,0],S[im,j,1],S[im,j,2]
            ypx,ypy,ypz=S[i,jp,0],S[i,jp,1],S[i,jp,2]; ymx,ymy,ymz=S[i,jm,0],S[i,jm,1],S[i,jm,2]
            # exchange + Zeeman + noise
            hx=J*(xpx+xmx+ypx+ymx)+xi[i,j,0]; hy=J*(xpy+xmy+ypy+ymy)+xi[i,j,1]; hz=J*(xpz+xmz+ypz+ymz)+B+xi[i,j,2]
            # DMI: D[ xp x yhat + yhat x xm + yp x (-xhat) + (-xhat) x ym ]
            # a x yhat = (-a_z, 0, a_x); yhat x a = (a_z, 0, -a_x); a x (-xhat) = (0, -a_z, a_y); (-xhat) x a = (0, a_z, -a_y)
            hx+=D*(-xpz+xmz); hz+=D*(xpx-xmx)
            hy+=D*(-ypz+ymz); hz+=D*(ypy-ymy)
            gxx=(xpx-xmx)*0.5; gxy=(xpy-xmy)*0.5; gxz=(xpz-xmz)*0.5
            gyx=(ypx-ymx)*0.5; gyy=(ypy-ymy)*0.5; gyz=(ypz-ymz)*0.5
            # -S x H
            tx=-(sy*hz-sz*hy); ty=-(sz*hx-sx*hz); tz=-(sx*hy-sy*hx)
            tx+=w1x*gxx+w1y*gyx; ty+=w1x*gxy+w1y*gyy; tz+=w1x*gxz+w1y*gyz
            vx=w2x*gxx+w2y*gyx; vy=w2x*gxy+w2y*gyy; vz=w2x*gxz+w2y*gyz
            tx+=sy*vz-sz*vy; ty+=sz*vx-sx*vz; tz+=sx*vy-sy*vx
            d=tx*sx+ty*sy+tz*sz; tx-=d*sx; ty-=d*sy; tz-=d*sz
            out[i,j,0]=c*(tx+a*(sy*tz-sz*ty)); out[i,j,1]=c*(ty+a*(sz*tx-sx*tz)); out[i,j,2]=c*(tz+a*(sx*ty-sy*tx))
@njit(parallel=True,fastmath=True,cache=True)
def axpy_norm(S,f,dt,out):
    Lx,Ly=S.shape[0],S.shape[1]
    for i in prange(Lx):
        for j in range(Ly):
            x=S[i,j,0]+dt*f[i,j,0]; y=S[i,j,1]+dt*f[i,j,1]; z=S[i,j,2]+dt*f[i,j,2]; n=1/np.sqrt(x*x+y*y+z*z)
            out[i,j,0]=x*n; out[i,j,1]=y*n; out[i,j,2]=z*n
@njit(parallel=True,fastmath=True,cache=True)
def heun_finish(S,f1,f2,dt):
    Lx,Ly=S.shape[0],S.shape[1]
    for i in prange(Lx):
        for j in range(Ly):
            x=S[i,j,0]+0.5*dt*(f1[i,j,0]+f2[i,j,0]); y=S[i,j,1]+0.5*dt*(f1[i,j,1]+f2[i,j,1]); z=S[i,j,2]+0.5*dt*(f1[i,j,2]+f2[i,j,2])
            n=1/np.sqrt(x*x+y*y+z*z); S[i,j,0]=x*n; S[i,j,1]=y*n; S[i,j,2]=z*n
class Integrator:
    def __init__(s,S,J,D,B,a,w1,w2,T,dt,seed=0):
        s.S=np.ascontiguousarray(S,dtype=np.float64); s.p=(J,D,B,a,w1[0],w1[1],w2[0],w2[1])
        s.T=T; s.dt=dt; s.sig=np.sqrt(2*a*T/dt) if T>0 else 0.0
        s.f1=np.empty_like(s.S); s.f2=np.empty_like(s.S); s.S1=np.empty_like(s.S); s.xi=np.zeros_like(s.S)
        s.rng=np.random.default_rng(seed)
    def run(s,n):
        for _ in range(n):
            if s.sig>0: s.xi[...]=s.sig*s.rng.standard_normal(s.S.shape)
            rhs(s.S,s.xi,s.f1,*s.p); axpy_norm(s.S,s.f1,s.dt,s.S1)
            rhs(s.S1,s.xi,s.f2,*s.p); heun_finish(s.S,s.f1,s.f2,s.dt)
