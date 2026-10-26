"""Abundance factors for the lightest tails that still reach beta_max = 1e-15 with the perfect-fluid threshold
(window edge), for M_PBH = 0.2 M_H (Dave et al.) and M_PBH = M_H. Used in Sec. V B."""
import numpy as np
from scipy.optimize import brentq
import seed_window as sw
from seed_window import beta_max, S_GN, S_ST, S_PL, S_LN
def solve(fun, lo, hi, target=1e-15):
    return brentq(lambda x: np.log10(fun(x)) - np.log10(target), lo, hi, xtol=1e-6)
fams = [('generalized normal', lambda M, p, sh: beta_max(M, S_GN, p, shifted=sh), (0.3, 1.0)),
        ('stretched exp. x=1', lambda M, p, sh: beta_max(M, S_ST, p, 1.0, shifted=sh), (0.3, 1.0)),
        ('power law x=4.5', lambda M, q, sh: beta_max(M, S_PL, q, 4.5, shifted=sh), (2.2, 12.0)),
        ('log-normal', lambda M, s, sh: beta_max(M, S_LN, s, shifted=sh), (0.2, 1.5))]
orig = sw.Delta
for gam in (0.2, 1.0):
    sw.Delta = lambda M, gamma=gam: orig(M, gamma)
    print(f'M_PBH = {gam} M_H; Delta at 1e5, 1e6, 1e7 =', [round(sw.Delta(M), 4) for M in (1e5, 1e6, 1e7)])
    for name, f, br in fams:
        row = []
        for M in (1e5, 1e6, 1e7):
            x = solve(lambda x: f(M, x, False), *br)
            row.append(f'{x:.3f}: factor {1e-15/f(M, x, True):.2f}')
        print(f'  {name:20s}', ' | '.join(row))
sw.Delta = orig
for q in (4.0,):
    print('power law n=4, x=4.5: factors', [round(beta_max(M, S_PL, q, 4.5)/beta_max(M, S_PL, q, 4.5, shifted=True), 2) for M in (1e6, 1e7)])
# sensitivity to the transfer of the shift (Sec. V B): one-point threshold raised like the central zeta (14.7%)
sw.Delta = lambda M, gamma=0.2: 0.147
row = []
for name, f, br in fams:
    x = solve(lambda x: f(1e6, x, False), *br)
    row.append(f'{name}: {1e-15/f(1e6, x, True):.1f}')
print('window-edge factors at 1e6 Msun with a 14.7% shift:', '; '.join(row))
sw.Delta = orig
