import numpy as np, time, sys
from bisect_fluid import bisect
for kw in [dict(N=800), dict(N=1600), dict(N=3200), dict(N=1600, eps0=0.05), dict(N=1600, eps0=0.2), dict(N=1600, cH=0.0025)]:
    t1 = time.time(); dc, log = bisect(lo=0.49, hi=0.51, tol=1e-4, **kw)
    print(kw, f"delta_c = {dc:.5f}  ({time.time()-t1:.0f}s)"); sys.stdout.flush()
