"""Resolution check of the full model (reviewer 2, C5): single-rate model, T_dec = 1 MeV, T0 = 10 MeV (T_H = 1 MeV),
N = 400 instead of the default 300. Writes review_tests/res_eosc_N400_1.0_10.0.json (archived N = 300 file untouched)."""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
from run_eosc import run
lo, hi, tol = 0.518, 0.524, 2e-4; log = []; TH = None; t0 = time.time()
while hi - lo > tol:
    mid = 0.5*(lo+hi); out = run(mid, 10.0, Tdec=1.0, N=400); TH = out['TH'] or TH
    st = out['status'] + ('*' if out.get('fallback') else '')
    log.append((mid, st, round(float(out['eps']), 2)))
    print(f'   N=400 delta={mid:.5f}: {st} eps={out["eps"]:.2f}', flush=True)
    if out['status'] == 'BH': hi = mid
    elif out['status'] == 'dispersed': lo = mid
    else: break
print(f'RESULT N=400 Tdec=1 T0=10: TH={TH:.4f} delta_c={0.5*(lo+hi):.5f} +- {0.5*(hi-lo):.5f} ({(time.time()-t0)/60:.1f} min)', flush=True)
json.dump(dict(Tdec=1.0, T0=10.0, N=400, TH=TH, dc=0.5*(lo+hi), w=hi-lo, log=log), open(os.path.join(HERE, 'res_eosc_N400_1.0_10.0.json'), 'w'))
