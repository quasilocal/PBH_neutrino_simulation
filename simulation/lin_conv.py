import numpy as np, time, sys
from run_nu import run
from linear_pred import centre
taus = np.array([1.0, 2.0, 2.5, 3.0, 3.5])
L = centre(0.409, 0.02, taus, 1.0)
for N, K in [(400, 64), (800, 32)]:
    res = {}
    for dm in [0.005, 0.0025]:
        t1 = time.time()
        out = run(dm, fnu=0.409, N=N, K=K, eps0=0.02, eps_max=3.6, stop_rule=False, hist_every=5, cH=0.001)
        h = out['hist']; dnu = h[:,5]/h[:,7]-1; dfl = h[:,4]/h[:,6]-1
        res[dm] = [(dfl[np.argmin(abs(h[:,1]-tau))]/(L[0,j]*dm), dnu[np.argmin(abs(h[:,1]-tau))]/(L[1,j]*dm)) for j, tau in enumerate(taus)]
        print(f"  N={N} K={K} dm={dm}: {time.time()-t1:.0f}s"); sys.stdout.flush()
    ex = [(2*res[0.0025][j][0]-res[0.005][j][0], 2*res[0.0025][j][1]-res[0.005][j][1]) for j in range(len(taus))]
    print(f"N={N} K={K}: zero-amplitude sim/lin at tau={list(taus)}: fluid", ' '.join(f"{x[0]:.4f}" for x in ex), "| nu", ' '.join(f"{x[1]:.4f}" for x in ex))
    sys.stdout.flush()
