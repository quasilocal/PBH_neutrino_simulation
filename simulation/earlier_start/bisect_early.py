"""Earlier start (Sec. IV E, Table IV, and the eps0 = 0.025 row of Table II). Threshold bisections for

- the heaviest mass (T_H = 0.0625 MeV) and the bottom of the e+- dip (T_H = 0.117 MeV), perfect fluid (N = 400) and
  single-rate neutrinos with Tdec = 1 MeV (N = 300), started at eps0 = 0.05 and 0.025 instead of 0.1, with the start
  temperature T0 tuned to keep T_H (T0_calibration.json, from calibrate_T0.py),
- pure radiation started at eps0 = 0.025, the baseline dependence on the starting time (eps0 = 0.05 is in Table II).

Runs started at eps0 = 0.025 use H dt <= 0.001 (background drift, Sec. III B), those at 0.05 the default 0.005.
Usage: python bisect_early.py [workers]. Results: res_<label>.json (a finished task is skipped on restart), logs in
logs/. Run summarize.py afterwards."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, '..')
sys.path.insert(0, SIM)
os.environ.setdefault('NUMBA_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')

EOS_DC = {('pf', 0.5): 'res_eos_pf_0.5.json', ('pf', 1.0): 'res_eos_pf_1.0.json',     # archived, eps0 = 0.1
          ('nu', 0.5): 'res_eosc_1.0_0.5.json', ('nu', 1.0): 'res_eosc_1.0_1.0.json'}


def tasks():
    cal = json.load(open(os.path.join(HERE, 'T0_calibration.json')))
    T = [('rad', 'rad_pf_e0.025', dict(fnu=0.0, K=2, eps0=0.025, cH=0.001), 0.491, 0.503, 2e-4),
         ('rad', 'rad_nu_e0.025', dict(fnu=0.409, K=32, eps0=0.025, cH=0.001), 0.5218, 0.5338, 2.5e-4)]
    for eps0, cH in ((0.025, 0.001), (0.05, 0.005)):
        for kind in ('pf', 'nu'):
            for T0a in (0.5, 1.0):
                c = cal[f'{kind}_{T0a}_{eps0}']
                dc = round(json.load(open(os.path.join(SIM, EOS_DC[(kind, T0a)])))['dc'], 6)
                T.append(('eos' + kind, f'eos_{kind}_T{T0a:g}_e{eps0:g}', dict(T0=c['T0'], eps0=eps0, cH=cH),
                          dc - 0.006, dc + 0.006, 2e-4 if kind == 'pf' else 2.5e-4))
    return T


def one_run(kind, dm, p):
    if kind == 'rad':
        from run_nu import run
        return run(dm, fnu=p['fnu'], K=p['K'], eps0=p['eps0'], N=400, qshape=1.0, cH=p['cH'], eps_max=15.0, rmax=9.0)
    if kind == 'eospf':
        from run_eos import run
        return run(dm, kind='pf', T0=p['T0'], K=2, N=400, eps0=p['eps0'], cH=p['cH'])
    if kind == 'eosnu':
        from run_eosc import run
        return run(dm, p['T0'], Tdec=1.0, N=300, K=32, eps0=p['eps0'], cH=p['cH'])
    raise ValueError(kind)


def job(task):
    kind, label, p, lo, hi, tol = task
    fn = os.path.join(HERE, f'res_{label}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    logf = open(os.path.join(HERE, 'logs', f'{label}.txt'), 'a', encoding='utf-8')
    log, TH, t0 = [], None, time.time()
    lo0, hi0 = lo, hi
    while hi - lo > tol:
        mid = 0.5*(lo + hi); t1 = time.time()
        out = one_run(kind, mid, p)
        TH = out.get('TH') or TH
        st = out['status'] + ('*' if out.get('fallback') else '')
        log.append((round(mid, 6), st, round(float(out['eps']), 2), round(time.time() - t1)))
        logf.write(f'{label} delta={mid:.6f}: {st} eps={float(out["eps"]):.2f} ({time.time()-t1:.0f}s)\n'); logf.flush()
        if out['status'] == 'BH': hi = mid
        elif out['status'] == 'dispersed': lo = mid
        else: break
    outcomes = {s.rstrip('*') for _, s, _, _ in log}
    res = dict(label=label, kind=kind, params=p, bracket0=[lo0, hi0], dc=0.5*(lo + hi), w=hi - lo, TH=TH, log=log,
               at_edge=(len(outcomes) == 1), minutes=round((time.time() - t0)/60, 1))
    json.dump(res, open(fn, 'w'), indent=1)
    logf.write(f'RESULT {label}: delta_c = {res["dc"]:.5f} +- {res["w"]/2:.5f} ({res["minutes"]} min)\n'); logf.close()
    return res


def init():
    if os.name == 'nt':                         # below-normal priority keeps the machine usable
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


if __name__ == '__main__':
    from multiprocessing import Pool
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    T = tasks()
    print(f'{len(T)} bisections on {workers} workers', flush=True)
    with Pool(workers, initializer=init) as pool:
        for r in pool.imap_unordered(job, T):
            print(f"{time.strftime('%H:%M')} {r['label']}: delta_c = {r['dc']:.5f} +- {r['w']/2:.5f}"
                  f"{'  AT BRACKET EDGE' if r['at_edge'] else ''} ({r['minutes']} min)", flush=True)
    print('all done', flush=True)
