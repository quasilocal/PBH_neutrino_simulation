import time, sys
from bisect_fluid import bisect
for kw in [dict(N=400), dict(N=800), dict(N=1600), dict(N=3200)]:
    t1 = time.time(); dc, log = bisect(lo=0.49, hi=0.51, tol=1e-4, **kw)
    print(kw, f"delta_c = {dc:.5f} +- 0.00005 ({time.time()-t1:.0f}s)", flush=True)
