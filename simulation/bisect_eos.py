import sys, json, time
from run_eos import run
def bisect(lo, hi, tol, **kw):
    log = []; TH = None
    while hi - lo > tol:
        mid = 0.5*(lo+hi); t1 = time.time(); out = run(mid, **kw); TH = out['TH'] or TH
        log.append((round(mid, 6), out['status'], round(float(out['eps']), 2)))
        print(f"   {kw.get('kind')} T0={kw.get('T0')} delta={mid:.5f}: {out['status']} eps={out['eps']:.2f} TH={out['TH']} ({time.time()-t1:.0f}s)", flush=True)
        if out['status'] == 'BH': hi = mid
        elif out['status'] == 'dispersed': lo = mid
        else: break
    return 0.5*(lo+hi), hi-lo, TH, log
if __name__ == '__main__':
    kind = sys.argv[1]; T0s = [float(x) for x in sys.argv[2].split(',')]
    lo, hi, tol = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    extra = dict(K=2) if kind == 'pf' else dict(K=32)
    if len(sys.argv) > 6: extra['eps_dec'] = float(sys.argv[6])
    for T0 in T0s:
        dc, w, TH, log = bisect(lo, hi, tol, kind=kind, T0=T0, **extra)
        print(f"RESULT {kind} T0={T0}: TH={TH:.4f} MeV delta_c={dc:.5f} +- {w/2:.5f}", flush=True)
        json.dump(dict(kind=kind, T0=T0, TH=TH, dc=dc, w=w, log=log), open(f'res_eos_{kind}_{T0}.json', 'w'))
