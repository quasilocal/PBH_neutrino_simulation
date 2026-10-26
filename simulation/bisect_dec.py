import sys, json, time
from run_nuc import run
epsd = float(sys.argv[1]); lo, hi = float(sys.argv[2]), float(sys.argv[3]); tol = float(sys.argv[4])
log = []; t0 = time.time()
while hi - lo > tol:
    mid = 0.5*(lo+hi); t1 = time.time(); out = run(mid, fnu=0.409, N=400, K=32, eps0=0.1, eps_dec=epsd, eps_max=15.0)
    print(f"   eps_dec={epsd} delta={mid:.5f}: {out['status']} eps={out['eps']:.2f} ({time.time()-t1:.0f}s)", flush=True)
    log.append((mid, out['status']))
    if out['status'] == 'BH': hi = mid
    elif out['status'] == 'dispersed': lo = mid
    else: break
print(f"RESULT eps_dec={epsd}: delta_c = {0.5*(lo+hi):.5f} +- {0.5*(hi-lo):.5f} ({(time.time()-t0)/60:.1f} min)", flush=True)
json.dump(dict(eps_dec=epsd, dc=0.5*(lo+hi), w=hi-lo, log=log), open(f'res_dec_{epsd}.json', 'w'))
