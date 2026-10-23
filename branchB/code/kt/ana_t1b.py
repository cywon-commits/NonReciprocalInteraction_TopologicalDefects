import numpy as np, json
br=json.load(open("bloch_t1.json"))
def zk(tag,kx,ky):
    U=np.load(f"disp_{tag}.npy").astype(float); R=np.load(f"R_{tag}.npy"); U=U-U.mean(1,keepdims=True)
    ph=np.exp(-1j*(R[:,0]*kx+R[:,1]*ky))/np.sqrt(len(R)); return np.einsum('tnc,n->tc',U,ph)
def var_blocks(z,nb=10):
    p=(np.abs(z)**2).sum(1); L=len(p)//nb; b=[p[i*L:(i+1)*L].mean() for i in range(nb)]; return np.mean(b),np.std(b)/np.sqrt(nb)
print(" k_x     k_y   | total <|u_k|^2> ratio (meas)  | predicted (R_a+R_b)/2 | R_a, R_b")
for (kx,ky,b0),(_,_,b1) in zip(br["0.0"]["k"],br["0.03"]["k"]):
    v0,e0=var_blocks(zk("t2u00",kx,ky)); v1,e1=var_blocks(zk("t2u03",kx,ky))
    Ra=b0[0][0]/b1[0][0]; Rb=b0[1][0]/b1[1][0]; r=v1/v0; er=r*np.hypot(e0/v0,e1/v1)
    print(f"{kx:+.4f} {ky:+.4f} | {r:.2f} ± {er:.2f} | {(Ra+Rb)/2:.2f} | {Ra:.2f}, {Rb:.2f}")
