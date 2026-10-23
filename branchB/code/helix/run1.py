from helix import *
J=1.0; N=200; n=3
th=2*np.pi*n/N; D=J*np.tan(th)       # helix pitch commensurate
print("theta",th,"D",D)
for (a,b) in [(0.1,0.3),(0.2,0.5)]:
  for u in [0.0,0.02,0.05]:
    M,Om=predict(J,D,th,u,a,b)
    S,_=cone(N,n,M)
    r=rhs(S,J,D,u,a,b,Om)
    print(a,b,u,"M_pred",M,"residual",np.abs(r).max())
