import json, numpy as np
from scipy.optimize import brentq
from seed_window import beta_max, S_GN, S_ST, S_PL, S_LN, Delta
target = 1e-15
out = {}
def solve(fun, lo, hi):
    g = lambda x: np.log10(fun(x)) - np.log10(target)
    return brentq(g, lo, hi, xtol=1e-5)
for M in (1e6, 1e7):
    row = {}
    for sh in (False, True):
        tag = 'nu' if sh else 'pf'
        row[f'GN_p_{tag}'] = solve(lambda p: beta_max(M, S_GN, p, shifted=sh), 0.3, 1.0)
        row[f'ST_p_{tag}'] = solve(lambda p: beta_max(M, S_ST, p, 1.0, shifted=sh), 0.3, 1.0)
        row[f'PL_q_{tag}'] = solve(lambda q: beta_max(M, S_PL, q, 4.5, shifted=sh), 2.2, 12.0)
        row[f'LN_s_{tag}'] = solve(lambda s: beta_max(M, S_LN, s, shifted=sh), 0.2, 1.5)
    out[M] = row
    print(f'M = {M:.0e}: Delta = {Delta(M):.4f}')
    for fam, par in (('GN', 'p'), ('ST', 'p'), ('PL', 'q'), ('LN', 's')):
        a, b = row[f'{fam}_{par}_pf'], row[f'{fam}_{par}_nu']
        print(f'   {fam}: {par} for beta_max = 1e-15: perfect fluid {a:.4f}  ->  neutrinos {b:.4f}  (change {b-a:+.4f})')
json.dump({str(k): v for k, v in out.items()}, open('seed_bounds.json', 'w'), indent=1)
