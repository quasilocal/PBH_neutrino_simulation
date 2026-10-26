import sys, json, time
from run_eosc import run
Tdec = float(sys.argv[1]); T0s = [float(x) for x in sys.argv[2].split(',')]
lo0, hi0, tol = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
for T0 in T0s:
    lo, hi = lo0, hi0; log = []; TH = None; t0 = time.time()
    while hi - lo > tol:
        mid = 0.5*(lo+hi); t1 = time.time(); out = run(mid, T0, Tdec=Tdec); TH = out['TH'] or TH
        log.append((mid, out['status'] + ('*' if out.get('fallback') else ''), round(float(out['eps']), 2)))
        print(f"   Tdec={Tdec} T0={T0} delta={mid:.5f}: {out['status']}{'*' if out.get('fallback') else ''} eps={out['eps']:.2f} ({time.time()-t1:.0f}s)", flush=True)
        if out['status'] == 'BH': hi = mid
        elif out['status'] == 'dispersed': lo = mid
        else: break
    print(f"RESULT Tdec={Tdec} T0={T0}: TH={TH:.4f} delta_c={0.5*(lo+hi):.5f} +- {0.5*(hi-lo):.5f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    json.dump(dict(Tdec=Tdec, T0=T0, TH=TH, dc=0.5*(lo+hi), w=hi-lo, log=log), open(f'res_eoscflag_{Tdec}_{T0}.json', 'w'))
