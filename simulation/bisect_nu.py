import numpy as np, time, sys, json
from run_nu import run

def bisect(lo, hi, tol=5e-4, tag='', **kw):
    log = []
    while hi - lo > tol:
        mid = 0.5*(lo+hi); t1 = time.time(); out = run(mid, **kw)
        log.append((round(mid, 6), out['status'] + ('*' if out.get('fallback') else ''), round(float(out['eps']), 2), round(time.time()-t1)))
        print(f"   {tag} delta={mid:.5f}: {out['status']}{'*' if out.get('fallback') else ''} at eps={out['eps']:.2f} ({time.time()-t1:.0f}s)"); sys.stdout.flush()
        if out['status'] == 'BH': hi = mid
        elif out['status'] == 'dispersed': lo = mid
        else: print('   stop:', out['status']); break
    return 0.5*(lo+hi), hi-lo, log

if __name__ == '__main__':
    job = sys.argv[1]
    if job == 'fluid':
        dc, w, log = bisect(0.49, 0.51, tol=2e-4, tag='fluid', fnu=0.0, N=400, K=2)
    elif job.startswith('nu'):
        K = int(job[2:])
        dc, w, log = bisect(0.45, 0.65, tol=5e-4, tag=f'nu K={K}', fnu=0.409, N=400, K=K)
    print(f"RESULT {job}: delta_c = {dc:.5f} +- {w/2:.5f}")
    json.dump(dict(job=job, dc=dc, w=w, log=log), open(f'res_{job}.json', 'w'))
