from csl import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
J=1.0; D=0.3; N=40; k=2*np.pi/(N*40); Kp=5.0
Bs=[0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04]
num=[];form=[]
for B in Bs:
    m,S=relax(N,J,D,B,Kp); kt,_=kappa(m,S,k); f1,_=formula(S); num.append(abs(kt)); form.append(f1)
    print(B,abs(kt),f1,flush=True)
np.save("csl_curve.npy",np.array([Bs,num,form]))
