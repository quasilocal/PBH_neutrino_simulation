import numpy as np, time
from run_nu import run
for dm in [0.52812, 0.5265]:
    t1 = time.time()
    out = run(dm, fnu=0.409, N=400, K=32, eps0=0.1, eps_max=15.0, disperse_rule=False, hist_every=50)
    h = out['hist']
    print(f"dm={dm}: {out['status']} at eps={out['eps']:.2f} ({time.time()-t1:.0f}s); C_max at eps:",
          ' '.join(f"{x[1]:.1f}:{x[2]:.3f}" for x in h[::max(1, len(h)//15)]))
