import numpy as np, sys
sys.path.insert(0,'.')
from linear_nu import evolve
def summary(r, r0):
    t=r['t']; D=r['Delta']; m=t<8
    i=np.argmax(np.abs(D[m])); i0=np.argmax(np.abs(r0['Delta'][r0['t']<8]))
    return (r['phi'][0]/r0['phi'][0], abs(D[i])/abs(r0['Delta'][i0]), t[i])
# fluid check
r0=evolve(0.0); t=r0['t']; x=t/np.sqrt(3)
an=-6*(2/3)*(np.sin(x)/x-np.cos(x)); print('fluid check max|diff| =',np.max(np.abs(r0['Delta']-an)))
for R in [0.409,0.488]:
    for kw in [dict(), dict(lmax=80), dict(tau0=1e-4), dict(lmax=80,tau0=1e-4)]:
        r=evolve(R,**kw); rr0=evolve(0.0,**{k:v for k,v in kw.items()})
        s=summary(r,rr0); print(f"R={R} {kw}: superhorizon ratio {s[0]:.5f}, first-peak ratio {s[1]:.5f} at tau={s[2]:.3f}")
