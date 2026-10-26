import numpy as np, time, sys
from run_nu import run
from linear_pred import centre
taus = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
lin = {f: centre(f, 0.02, taus, 1.0) for f in [0.0, 0.409]}      # linear: proportional to delta_m
for fnu, dm in [(0.0, 0.005), (0.409, 0.01), (0.409, 0.005)]:
    t1 = time.time()
    out = run(dm, fnu=fnu, N=400, K=32, eps0=0.02, eps_max=3.6, stop_rule=False, hist_every=5, cH=0.001)
    h = out['hist']
    dfl = h[:,4]/h[:,6] - 1
    dnu = h[:,5]/h[:,7] - 1 if fnu > 0 else np.zeros_like(dfl)
    dtot = ((1-fnu)*h[:,4] + fnu*h[:,5])/((1-fnu)*h[:,6] + fnu*h[:,7]) - 1 if fnu > 0 else dfl
    L = lin[fnu]*dm
    print(f"fnu={fnu} dm={dm}: {out['status']} {out['steps']} steps, {time.time()-t1:.0f}s; max Hamiltonian-constraint violation {h[:,8].max():.1e}")
    for j, tau in enumerate(taus):
        i = np.argmin(abs(h[:,1]-tau))
        print(f"  tau={h[i,1]:.3f}: fluid sim/lin {dfl[i]/L[0,j]:.4f} | nu sim/lin {(dnu[i]/L[1,j] if fnu>0 else 0):.4f} | total sim/lin {dtot[i]/L[2,j]:.4f}   (lin total {L[2,j]: .5f})")
    sys.stdout.flush()
