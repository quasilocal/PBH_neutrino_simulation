import os
"""Data for the dynamics figure: same initial amplitude, fluid-only vs fluid + free-streaming neutrinos."""
import numpy as np, sys
from run_nu import run
def diag(t, epsn, r, R, U, e, I, M, mue, C):
    dmu = mue[1]-mue[0]; mu = 0.5*(mue[1:]+mue[:-1]); mu2 = (mue[1:]**3-mue[:-1]**3)/(3*dmu)
    eps = 2*np.pi*I.sum(axis=1)*dmu; q = 2*np.pi*(I*mu).sum(axis=1)*dmu; Pr = 2*np.pi*(I*mu2).sum(axis=1)*dmu
    rc = 0.5*(r[1:]+r[:-1]); core = rc < 0.3
    w = (R[1:]**3-R[:-1]**3)[core]
    epsc = np.sum(eps[core]*w)/np.sum(w); Prc = np.sum(Pr[core]*w)/np.sum(w)
    twoMR = 2*M[1:]/R[1:]; coll = U[1:] < 0
    m2 = twoMR[coll].max() if coll.any() else 0.0
    return (epsn, C[r[1:] < 4].max(), e[0]/e[-1]-1, (eps[0]/eps[-1]-1) if eps[-1] > 0 else 0.0, (Prc/epsc) if epsc > 0 else 0.0, m2)
out = {}
for tag, kw in [('fluid', dict(fnu=0.0, K=2)), ('nu', dict(fnu=0.409, K=32))]:
    o = run(0.51, N=400, eps0=0.1, eps_max=9.0, hist_every=10, diag=diag, disperse_rule=False, **kw)
    out[tag] = np.array(o['hist2']); print(tag, o['status'], round(o['eps'], 2), flush=True)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figs', 'dyn_data.npz'), **out)
