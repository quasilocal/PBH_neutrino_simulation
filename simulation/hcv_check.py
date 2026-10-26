import numpy as np, time
from run_nu import run
for dm in [0.530, 0.526]:
    t1 = time.time()
    out = run(dm, fnu=0.409, N=400, K=32, eps0=0.1, hist_every=10)
    h = out['hist']
    print(f"dm={dm}: {out['status']} at eps={out['eps']:.2f} ({time.time()-t1:.0f}s); max |Gam^2-(1+U^2-2M/R)| (r<3): "
          + ' '.join(f"{x[1]:.1f}:{x[8]:.1e}" for x in h[::max(1, len(h)//10)]))
