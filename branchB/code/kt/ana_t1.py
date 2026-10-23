import numpy as np, json
from scipy.signal import welch
br=json.load(open("bloch_t1.json"))
def L(w,G,w0): return G/np.pi/((w-w0)**2+G**2)
def spec(z,dt,nper):
    P=0
    for c in range(2):
        f,p=welch(z[:,c],fs=1/dt,nperseg=nper,return_onesided=False,detrend=False); P=P+p
    o=np.argsort(f); return 2*np.pi*f[o],P[o]/(2*np.pi)
def amps(tag,u,nper=8192):
    U=np.load(f"disp_{tag}.npy").astype(float); R=np.load(f"R_{tag}.npy"); N=U.shape[1]; dt=10.0; out=[]
    for kx,ky,bb in br[u]["k"]:
        ph=np.exp(-1j*(R[:,0]*kx+R[:,1]*ky))/np.sqrt(N); z=np.einsum('tnc,n->tc',U,ph)
        w,P=spec(z,dt,min(nper,len(z))); m=np.abs(w)<0.03
        (Ga,wa),(Gb,wb)=[(-x[0],x[1]) for x in bb]
        X=np.stack([L(w[m],Ga,wa),L(w[m],Gb,wb),np.ones(m.sum())],1); Wt=1/P[m]
        c,*_=np.linalg.lstsq(X*Wt[:,None],P[m]*Wt,rcond=None)
        # error from 4 sub-blocks
        sub=[]
        for i in range(4):
            zz=z[i*len(z)//4:(i+1)*len(z)//4]; w2,P2=spec(zz,dt,min(nper//2,len(zz))); m2=np.abs(w2)<0.03
            X2=np.stack([L(w2[m2],Ga,wa),L(w2[m2],Gb,wb),np.ones(m2.sum())],1); W2=1/P2[m2]
            sub.append(np.linalg.lstsq(X2*W2[:,None],P2[m2]*W2,rcond=None)[0])
        out.append((kx,ky,c,np.std(sub,0)/2,(Ga,Gb)))
    return out
if __name__=="__main__":
    A0=amps("t2u00","0.0"); A1=amps("t2u03","0.03")
    print(" k_x     k_y   | branch a: T_eff/T meas / pred | branch b: meas / pred")
    rows=[]
    for (kx,ky,c0,e0,G0),(_,_,c1,e1,G1) in zip(A0,A1):
        r=c1[:2]/c0[:2]; er=r*np.hypot(e1[:2]/c1[:2],e0[:2]/c0[:2]); pr=np.array(G0)/np.array(G1)
        rows.append([kx,ky,*r,*er,*pr])
        print(f"{kx:+.4f} {ky:+.4f} | {r[0]:.2f} ± {er[0]:.2f} / {pr[0]:.2f} | {r[1]:.2f} ± {er[1]:.2f} / {pr[1]:.2f}")
    np.save("t1_result.npy",np.array(rows))
