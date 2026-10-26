"""Parallel bisection (two trial amplitudes per round, run concurrently) for the multigroup code.
usage: trisect_mg.py <tag> <T0> <lo> <hi> <tol> [key=value ...]"""
import sys, json, time
from multiprocessing import Pool
from run_mg import run

def one(args):
    dm, T0, kw = args
    out = run(dm, T0, **kw)
    return dm, out['status'] + ('*' if out.get('fallback') else ''), float(out['eps']), out['TH'], out['seconds']

if __name__ == '__main__':
    tag, T0, lo, hi, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    kw = {}
    for a in sys.argv[6:]:
        k, v = a.split('=')
        kw[k] = (v if k == 'mode' else (int(v) if k in ('G', 'N', 'K') else float(v)))
    log = []; TH = None; t0 = time.time()
    with Pool(2) as pool:
        while hi - lo > tol:
            pts = [lo + (hi-lo)/3, lo + 2*(hi-lo)/3]
            res = pool.map(one, [(p, T0, kw) for p in pts])
            for dm, st, eps, th, sec in res:
                TH = th or TH; log.append((dm, st, round(eps, 2)))
                print(f"   {tag} T0={T0} delta={dm:.5f}: {st} eps={eps:.2f} ({sec:.0f}s)", flush=True)
            new_lo, new_hi = lo, hi
            for dm, st, eps, th, sec in res:
                if st.startswith('BH'): new_hi = min(new_hi, dm)
                elif st == 'dispersed': new_lo = max(new_lo, dm)
            if (new_lo, new_hi) == (lo, hi): break
            lo, hi = new_lo, new_hi
    print(f"RESULT {tag} T0={T0}: TH={TH:.4f} delta_c={0.5*(lo+hi):.5f} +- {0.5*(hi-lo):.5f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    json.dump(dict(tag=tag, T0=T0, TH=TH, dc=0.5*(lo+hi), w=hi-lo, log=log, kw=kw), open(f'res_mg_{tag}_{T0}.json', 'w'))
