"""Threshold bisection with the multigroup code.
usage: bisect_mg.py <tag> <T0> <lo> <hi> <tol> [key=value ...]   (keys passed to run_mg.run, e.g. G=6 K=16 N=200 nunu=0)"""
import sys, json, time
from run_mg import run
tag, T0, lo, hi, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
kw = {}
for a in sys.argv[6:]:
    k, v = a.split('=')
    kw[k] = (v if k == 'mode' else (int(v) if k in ('G', 'N', 'K') else float(v)))
if 'nunu' in kw: kw['nunu'] = bool(kw['nunu'])
log = []; TH = None; t0 = time.time()
while hi - lo > tol:
    mid = 0.5*(lo+hi); out = run(mid, T0, **kw); TH = out['TH'] or TH
    st = out['status'] + ('*' if out.get('fallback') else '')
    log.append((mid, st, round(float(out['eps']), 2)))
    print(f"   {tag} T0={T0} delta={mid:.5f}: {st} eps={out['eps']:.2f} ({out['seconds']:.0f}s)", flush=True)
    if out['status'] == 'BH': hi = mid
    elif out['status'] == 'dispersed': lo = mid
    else: break
print(f"RESULT {tag} T0={T0}: TH={TH:.4f} delta_c={0.5*(lo+hi):.5f} +- {0.5*(hi-lo):.5f} ({(time.time()-t0)/60:.1f} min)", flush=True)
json.dump(dict(tag=tag, T0=T0, TH=TH, dc=0.5*(lo+hi), w=hi-lo, log=log, kw={k: (v if not isinstance(v, bool) else int(v)) for k, v in kw.items()}),
          open(f'res_mg_{tag}_{T0}.json', 'w'))
