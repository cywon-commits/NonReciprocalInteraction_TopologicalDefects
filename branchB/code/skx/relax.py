from skx import *
J=1.0; D=0.4; Lx,Ly=19,33; B=0.5*D*D/J
m=Model(Lx,Ly,J,D,B)
x,y=np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
Q=4*np.pi/(np.sqrt(3)*Lx)
qs=[np.array([0,1.]),np.array([np.sqrt(3)/2,-0.5]),np.array([-np.sqrt(3)/2,-0.5])]
# triple-q: wavevectors commensurate with the rectangular cell
G1=2*np.pi/Lx; G2=2*np.pi/Ly
ks=[np.array([0,2*G2]),np.array([G1,-G2]),np.array([-G1,-G2])]
S=np.zeros((Lx,Ly,3)); S[...,2]=0.6
for kk in ks:
    ph=kk[0]*x+kk[1]*y; kh=kk/np.linalg.norm(kk)
    S[...,2]+= -np.cos(ph)
    S[...,0]+= kh[0]*np.sin(ph); S[...,1]+= kh[1]*np.sin(ph)
S/=np.linalg.norm(S,axis=-1)[...,None]
z=np.zeros(2)
for r in range(6):
    S=rk4(m,S,z,z,0.5,0.1,2000)
    print(r,"residual",np.abs(m.rhs(S,z,z,0.5)).max(),"Q",GD(S)[2],"mz mean",S[...,2].mean())
np.save("S0.npy",S)
