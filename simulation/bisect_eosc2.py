import sys, json, time
from run_eosc2 import run
Te, Tm = float(sys.argv[1]), float(sys.argv[2]); T0s = [float(x) for x in sys.argv[3].split(',')]
lo0, hi0, tol = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
for T0 in T0s:
    lo, hi = lo0, hi0; log = []; TH = None; t0 = time.time()
    while hi - lo > tol:
        mid = 0.5*(lo+hi); t1 = time.time(); out = run(mid, T0, Tdec_e=Te, Tdec_mt=Tm); TH = out['TH'] or TH
        st = out['status'] + ('*' if out.get('fallback') else '')
        log.append((mid, st, round(float(out['eps']), 2)))
        print(f"   Te={Te} Tmt={Tm} T0={T0} delta={mid:.5f}: {st} eps={out['eps']:.2f} ({time.time()-t1:.0f}s)", flush=True)
        if out['status'] == 'BH': hi = mid
        elif out['status'] == 'dispersed': lo = mid
        else: break
    print(f"RESULT Te={Te} Tmt={Tm} T0={T0}: TH={TH:.4f} delta_c={0.5*(lo+hi):.5f} +- {0.5*(hi-lo):.5f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    json.dump(dict(Tdec_e=Te, Tdec_mt=Tm, T0=T0, TH=TH, dc=0.5*(lo+hi), w=hi-lo, log=log), open(f'res_eosc2_{Te}_{Tm}_{T0}.json', 'w'))
