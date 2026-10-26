"""Bracket tests for reviewer 1's code audit: for each variant, run the multigroup model at the archived threshold
+- 0.001. If the outcomes (disperse below, black hole above) are unchanged, the variant moves the threshold by less
than 0.001 (0.2%). Results: review_tests/res_test_<label>_<T0>_<delta>.json (archived files untouched).
(Second pass with the refined diagnostics: python3 bracket_mg.py diag2)"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
from multiprocessing import Pool
BASE = dict(mode='phys', G=6, x_lo=0.4, x_hi=20.0, diag_every=20)
DC = {10.0: 0.519406, 5.0: 0.524938, 3.0: 0.523063}          # archived res_mg_phys6_<T0>.json
VARIANTS = [('baseline', 10.0, {}),
            ('chilocal', 10.0, dict(chi_local=True)), ('chilocal', 5.0, dict(chi_local=True)), ('chilocal', 3.0, dict(chi_local=True)),
            ('x80', 10.0, dict(G=8, x_hi=80.0)),
            ('boosted', 10.0, dict(nnmode=1)),
            ('dthalf', 10.0, dict(cH=0.0025, cfl_r=0.2, cfl_mu=0.2, cfl_E=0.2)),
            ('boosted', 5.0, dict(nnmode=1)),
            ('x80', 5.0, dict(G=8, x_hi=80.0)),
            ('dtall', 10.0, dict(cH=0.0025, cfl_r=0.2, cfl_mu=0.2, cfl_E=0.2, ckap=0.25)),   # all limits halved, incl. kappa dt
            # combined changes (second round of review): wide grid with the boosted target, then also the local e+- correction
            ('x80boost', 10.0, dict(G=8, x_hi=80.0, nnmode=1)), ('x80boost', 5.0, dict(G=8, x_hi=80.0, nnmode=1)),
            ('x80boostchi', 10.0, dict(G=8, x_hi=80.0, nnmode=1, chi_local=True)),
            ('x80boostchi', 5.0, dict(G=8, x_hi=80.0, nnmode=1, chi_local=True))]
def job(arg):
    label, T0, kw, dm = arg
    fn = os.path.join(HERE, f'res_test_{label}_{T0}_{dm:.5f}.json')
    if os.path.exists(fn):                      # resume after an interruption
        return json.load(open(fn))
    from run_mg import run
    kk = dict(BASE); kk.update(kw)
    out = run(dm, T0, **kk)
    res = dict(label=label, T0=T0, delta=dm, kw=kw, status=out['status'], fallback=out['fallback'], eps=float(out['eps']),
               steps=out['steps'], TH=out['TH'], seconds=out['seconds'], diag=out['diag'])
    json.dump(res, open(os.path.join(HERE, f'res_test_{label}_{T0}_{dm:.5f}.json'), 'w'))
    return res
DIAG2 = [('baseline', 10.0, {}), ('x80', 10.0, dict(G=8, x_hi=80.0))]
if __name__ == '__main__':
    second = len(sys.argv) > 1 and sys.argv[1] == 'diag2'
    jobs = [(lab + ('_diag2' if second else ''), T0, kw, round(DC[T0] + s*0.001, 6))
            for lab, T0, kw in (DIAG2 if second else VARIANTS) for s in (-1, 1)]
    t0 = time.time()
    with Pool(2) as pool:
        for res in pool.imap(job, jobs):
            d = res['diag']
            print(f"{res['label']:9s} T0={res['T0']:4.1f} delta={res['delta']:.5f}: {res['status']}{'*' if res['fallback'] else ''} "
                  f"eps={res['eps']:.2f} ({res['seconds']:.0f}s)  max|q|/eps (kappa_n>H) {d['qmax']:.3f} at {d['qmax_at']}; "
                  f"T_b>1 MeV {d['qmax_hot']:.3f}; top-group fraction {d['top']:.3f} at {d['top_at']}; "
                  f"min I/(eps_s/4pi) {d.get('Imin_rel', float('nan')):.2e}  [{(time.time()-t0)/60:.0f} min]", flush=True)
