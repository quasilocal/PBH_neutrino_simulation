import numpy as np, time, sys
from run_fluid import run

def bisect(lo=0.45, hi=0.55, tol=2e-4, **kw):
    log = []
    while hi - lo > tol:
        mid = 0.5*(lo+hi); out = run(mid, **kw)
        log.append((mid, out['status'], round(out['eps'],2)))
        if out['status'] == 'BH': hi = mid
        elif out['status'] == 'dispersed': lo = mid
        else:
            print('  undecided/crash at', mid, out['status'], out['eps']); break
    return 0.5*(lo+hi), log

if __name__ == '__main__':
    for kw in [dict(N=800), dict(N=1600), dict(N=800, cQ=0.0), dict(N=800, rmax=15.0)]:
        t1 = time.time(); dc, log = bisect(**kw)
        print(kw, f"delta_c = {dc:.4f}  ({time.time()-t1:.0f}s)", log[-3:])
        sys.stdout.flush()
