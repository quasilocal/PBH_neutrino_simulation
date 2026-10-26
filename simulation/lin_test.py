import numpy as np, time, sys
from run_nu import run
from linear_pred import centre
taus = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
for fnu in [0.0, 0.409]:
    t1 = time.time()
    out = run(0.01, fnu=fnu, N=400, K=32, eps0=0.02, eps_max=3.6, stop_rule=False, hist_every=5)
    h = out['hist']
    dfl = h[:,4]/h[:,6] - 1
    dnu = (h[:,5]/h[:,7] - 1) if fnu > 0 else np.zeros_like(dfl)
    dtot = ((1-fnu)*h[:,4] + fnu*h[:,5]*(fnu>0))/((1-fnu)*h[:,6] + fnu*h[:,7]*(fnu>0)) - 1 if fnu>0 else dfl
    lin = centre(fnu, 0.02, taus, 0.01)
    print(f"fnu={fnu}: sim took {time.time()-t1:.0f}s, {out['steps']} steps")
    for j, tau in enumerate(taus):
        i = np.argmin(abs(h[:,1]-tau))
        print(f"  tau={h[i,1]:.3f}: fluid sim {dfl[i]: .5f} lin {lin[0,j]: .5f} | nu sim {dnu[i]: .5f} lin {lin[1,j]: .5f} | total sim {dtot[i]: .5f} lin {lin[2,j]: .5f}")
    sys.stdout.flush()
