import numpy as np, time
from run_nu import run
from ms_nu import rho_b
for eps0 in [0.1, 0.02]:
    t1 = time.time()
    out = run(0.0, fnu=0.409, N=400, K=32, eps0=eps0, eps_max=3.0, stop_rule=False)
    t = out['t']; I = out['I']; Im = I.mean(axis=1)
    print(f"eps0={eps0}: {out['status']} eps={out['eps']:.2f} steps={out['steps']} ({time.time()-t1:.0f}s): "
          f"anisotropy {np.abs(I/Im[:,None]-1).max():.2e}, inhomogeneity I {np.ptp(Im)/Im.mean():.2e}, e {np.ptp(out['e'])/out['e'].mean():.2e}, "
          f"max|Gam-1| {np.abs(out['Gam']-1).max():.2e}, drift e/eb-1 {out['e'].mean()/(0.591*rho_b(t))-1:.2e}")
