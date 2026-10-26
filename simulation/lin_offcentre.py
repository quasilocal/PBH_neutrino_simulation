import os
"""Linear-regime validation away from the centre (r ~ 0.4 r_m): simulation vs linear Boltzmann prediction."""
import numpy as np, sys, time
from run_nu import run
from linear_pred import mode
r0 = 0.4
taus = np.linspace(0.25, 4.0, 31)
def lin_at(Rnu, tau0, taus, r0, kmax=10.0, nk=240):
    ks = np.linspace(kmax/nk, kmax, nk); out = np.zeros((3, len(taus)))
    for k in ks:
        zt = 0.25*np.pi**1.5*np.exp(-k*k/4)*1.5*np.e        # per unit delta_m
        res = mode(k, tau0, taus, Rnu)
        for j in range(3):
            out[j] += k*k*zt*res[j]*np.sin(k*r0)/(k*r0)
    return -out*(ks[1]-ks[0])/(2*np.pi**2)
L = lin_at(0.409, 0.02, taus, r0)
res = {}
for K in [32, 64]:
    for dm in [0.005, 0.0025]:
        def diag(t, epsn, r, R, U, e, I, M, mue, C, K=K):
            rc = 0.5*(r[1:]+r[:-1]); i = np.argmin(abs(rc-r0)); dmu = mue[1]-mue[0]
            return (epsn, rc[i], e[i]/e[-1]-1, I[i].sum()/I[-1].sum()-1)
        t1 = time.time()
        o = run(dm, fnu=0.409, N=400, K=K, eps0=0.02, eps_max=4.05, stop_rule=False, hist_every=5, cH=0.001, diag=diag)
        h = np.array(o['hist2']); res[(K, dm)] = h
        print(f"K={K} dm={dm} done ({time.time()-t1:.0f}s), r_cell={h[0,1]:.3f}", flush=True)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figs', 'lin_offcentre.npz'), taus=taus, L=L, **{f"K{K}_{dm}": v for (K, dm), v in res.items()})
for K in [32, 64]:
    a, b = res[(K, 0.005)], res[(K, 0.0025)]
    print(f"K={K}: zero-amplitude sim/lin at tau=1,2,3,4 (plasma | nu):")
    for tau in [1.0, 2.0, 3.0, 4.0]:
        ia = np.argmin(abs(a[:,0]-tau)); ib = np.argmin(abs(b[:,0]-tau)); j = np.argmin(abs(taus-tau))
        fl = (2*b[ib,2]/0.0025 - a[ia,2]/0.005)/L[0,j]; nu = (2*b[ib,3]/0.0025 - a[ia,3]/0.005)/L[1,j]
        print(f"   tau={tau}: {fl:.4f} | {nu:.4f}   (lin plasma {L[0,j]:.3f}, nu {L[1,j]:.3f} per unit delta_m)")
