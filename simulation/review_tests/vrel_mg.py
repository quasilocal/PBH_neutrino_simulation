"""Relative motion of neutrinos and plasma in the multigroup model (reviewer 2, item C7): one run just below the
archived threshold at T_H = 2 and 1 MeV, recording the largest ratio |q|/eps of the total neutrino flux to the
neutrino energy density in the plasma frame (relative velocity ~ 3/4 of it) over all cells, while 1 < T_b < 2 MeV
and while T_b > 1 MeV. Results: review_tests/res_vrel_<T0>.json."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
from multiprocessing import Pool
BASE = dict(mode='phys', G=6, x_lo=0.4, x_hi=20.0, diag_every=10)
DC = {20.0: 0.507875, 10.0: 0.519406}                       # archived res_mg_phys6_<T0>.json
def job(T0):
    from run_mg import run
    dm = round(DC[T0] - 0.001, 6)
    out = run(dm, T0, **BASE)
    res = dict(T0=T0, delta=dm, status=out['status'], eps=float(out['eps']), TH=out['TH'], diag=out['diag'])
    json.dump(res, open(os.path.join(HERE, f'res_vrel_{T0}.json'), 'w'))
    return res
if __name__ == '__main__':
    with Pool(2) as pool:
        for res in pool.imap(job, [20.0, 10.0]):
            d = res['diag']
            print(f"T0={res['T0']} delta={res['delta']}: {res['status']} eps={res['eps']:.2f}; total |q|/eps for 1<T_b<2 MeV "
                  f"{d.get('qtot_12', float('nan')):.4f} at {d.get('qtot_12_at')}; for T_b>1 MeV {d.get('qtot_hot', float('nan')):.4f} "
                  f"at {d.get('qtot_hot_at')}; per-group (kappa_n>H, T_b>1 MeV) {d['qmax_hot']:.4f}", flush=True)
