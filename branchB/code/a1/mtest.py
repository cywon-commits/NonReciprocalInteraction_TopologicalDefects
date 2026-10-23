import numpy as np, json
from homog import sigma_eff
from skx import zh, dx_, dy_
J,D=1.0,0.4
def field(S,B):
    xp=np.roll(S,-1,0); xm=np.roll(S,1,0); yp=np.roll(S,-1,1); ym=np.roll(S,1,1)
    return J*(xp+xm+yp+ym)+B*zh+D*(np.cross(xp,dx_)+np.cross(dx_,xm)+np.cross(yp,dy_)+np.cross(dy_,ym))
g=json.load(open("grid_static.json"))
print(" key      kappa | scalar sigma_eff/<sigma> | with conjugate stiffness M=m.H : sigma_eff*<1/M>/<sigma/M>")
for key,v in g.items():
    Lx,bb=key.split('_'); S=np.load(f"Sgrid_{Lx}_{bb}.npy"); B=float(bb)*D*D
    gx=(np.roll(S,-1,0)-np.roll(S,1,0))/2; gy=(np.roll(S,-1,1)-np.roll(S,1,1))/2; sig=np.sum(gx*gx+gy*gy,-1)
    M=np.sum(S*field(S,B),-1)
    se=sigma_eff(sig); k0=se/sig.mean(); k1=se*np.mean(1/M)/np.mean(sig/M)
    print(f"{key:8s} {v['kLT']:.3f} | {k0:.3f} | {k1:.3f}   (M range {M.min():.2f}-{M.max():.2f})")
