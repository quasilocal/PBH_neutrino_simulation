"""Energy grid of the multigroup calculation (App. C). Multigroup runs at T_H = 1 and 0.51 MeV (T0 = 10 and 5 MeV)
with the common rest-frame neutrino-neutrino target (nnmode = 1), on grids from E/T_nu,b = 0.4 to 20, 80, 320 and
1280 with 6, 8, 10 and 12 groups. The Monte Carlo rate table ends at E/T = 20, and the rate coefficients are held at
their values there above it.

Thresholds: each grid is run at 8 amplitudes spaced STEP apart and placed symmetrically about the archived threshold
(res_mg_phys6_<T0>.json: first-order target, 6 groups to 20). merge.py takes the threshold between the largest
dispersed and the smallest collapsing amplitude.
Top group: on the widest grid at T_H = 1 MeV, the pair of runs at the archived threshold +- 0.001 (as for the
top-group fractions of the bracket tests, validation/bracket_mg.py) records the largest share of the energy of a
flavour in the top group (diag_every = 20).

Usage: python scan.py list            chunk numbers (4 runs each, for the GitHub workflow energy-grid.yml)
       python scan.py run <chunk>     one chunk
       python scan.py run all         all 66 runs, on all cores (about 10-40 min per run)
Results: points/<label>_d<amplitude>.json and points/top_d<amplitude>.json. Then run merge.py."""
import os, sys, json, time, tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, '..')
OUT = os.path.join(HERE, 'points')
GRIDS = [(6, 20.0), (8, 80.0), (10, 320.0), (12, 1280.0)]
STEP, NPTS, PER_RUNNER = 0.00025, 8, 4


def archived(T0):
    return round(json.load(open(os.path.join(SIM, f'res_mg_phys6_{T0}.json')))['dc'], 6)


def label(G, xh, T0):
    return f'egrid_G{G}_x{xh:g}_T{T0:g}'


def amplitudes(T0):
    return [round(archived(T0) + STEP*(k + 0.5), 7) for k in range(-NPTS//2, NPTS//2)]


def chunks():
    C = []
    for T0 in (10.0, 5.0):
        for G, xh in GRIDS:
            a = [('scan', G, xh, T0, d) for d in amplitudes(T0)]
            C += [a[i:i + PER_RUNNER] for i in range(0, len(a), PER_RUNNER)]
    C.append([('top', 12, 1280.0, 10.0, round(archived(10.0) + s, 6)) for s in (-0.001, 0.001)])
    return C


def init():
    os.environ['NUMBA_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
    tmp = os.environ.get('RUNNER_TEMP', tempfile.gettempdir())
    os.environ['NUMBA_CACHE_DIR'] = os.path.join(tmp, f'numba_{os.getpid()}')
    sys.path.insert(0, SIM)


def one(task):
    what, G, xh, T0, dm = task
    from run_mg import run
    t1 = time.time()
    out = run(dm, T0, mode='phys', G=G, x_lo=0.4, x_hi=xh, nnmode=1, diag_every=20 if what == 'top' else 0)
    res = dict(label=label(G, xh, T0), G=G, x_hi=xh, T0=T0, delta=dm, status=out['status'],
               fallback=bool(out['fallback']), eps=round(float(out['eps']), 3), steps=int(out['steps']),
               TH=float(out['TH']) if out['TH'] is not None else None, seconds=round(time.time() - t1))
    if what == 'top':
        at = out['diag']['top_at']
        res['top'] = float(out['diag']['top']); res['top_at'] = None if at is None else [float(v) for v in at]
        name = f'top_d{dm:.6f}.json'
    else:
        name = f'{res["label"]}_d{dm:.7f}.json'
    json.dump(res, open(os.path.join(OUT, name), 'w'), indent=1)
    return res


if __name__ == '__main__':
    if sys.argv[1] == 'list':
        print(json.dumps(list(range(len(chunks())))))
        sys.exit()
    from multiprocessing import Pool
    tasks = sum(chunks(), []) if sys.argv[2] == 'all' else chunks()[int(sys.argv[2])]
    os.makedirs(OUT, exist_ok=True)
    print(time.strftime('%H:%M:%S'), f'{len(tasks)} runs', flush=True)
    with Pool(min(len(tasks), os.cpu_count()), initializer=init) as pool:
        for r in pool.imap_unordered(one, tasks):
            print(time.strftime('%H:%M:%S'), f"{r['label']} delta={r['delta']:.7f}: {r['status']}"
                  f"{'*' if r['fallback'] else ''} eps={r['eps']:.2f} ({r['seconds']}s)", flush=True)
