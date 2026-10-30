import os
"""Abundance ratios beta_nu/beta_pf for different statistics of the primordial perturbations.
Thresholds dc are on the compaction delta_m (radiation-era relation). With Gaussian zeta the Gaussian variable is the
linear compaction C_l = -(4/3) r zeta', related to delta_m by delta_m = C_l - (3/8) C_l^2 (type-I branch, C_l < 4/3)."""
import json, glob, numpy as np
from scipy.special import erfc, erfcinv
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'simulation') + '/'
def Cl(d): return (1 - np.sqrt(1 - 1.5*d))/0.75
def ratio_gauss(xpf, xnu, beta):        # Gaussian variable x, threshold xpf -> xnu, fixed sigma set by beta_pf
    s = xpf/(np.sqrt(2)*erfcinv(beta))
    return erfc(xnu/(np.sqrt(2)*s))/beta   # the type-I cap at C_l = 4/3 changes these by < 1e-6
def ratio_exp(xpf, xnu, beta): return beta**(xnu/xpf - 1)
def ratio_pow(xpf, xnu, n): return (xnu/xpf)**(-n)

rad = {'pf': 0.49773, 0.409: 0.52849, 0.488: 0.53379}       # N_mu = 64 value for 0.409; N_mu = 32 for 0.488
print('radiation model:')
for R in (0.409, 0.488):
    d0, d1 = rad['pf'], rad[R]; c0, c1 = Cl(d0), Cl(d1)
    print(f"  R_nu={R}: delta_m {d0:.5f}->{d1:.5f} ({100*(d1/d0-1):.2f}%), C_l {c0:.5f}->{c1:.5f} ({100*(c1/c0-1):.2f}%),"
          f" power needed +{100*((c1/c0)**2-1):.1f}% (delta_m-Gaussian: +{100*((d1/d0)**2-1):.1f}%)")
    for b in (1e-9, 1e-12, 1e-15):
        print(f"    beta_pf={b:.0e}: dm-Gauss {ratio_gauss(d0,d1,b):.3g}  Cl-Gauss {ratio_gauss(c0,c1,b):.3g}  "
              f"exp-tail {ratio_exp(c0,c1,b):.3g}  pow n=3 {ratio_pow(c0,c1,3):.3g}  n=5 {ratio_pow(c0,c1,5):.3g}")

def load(pattern):
    out = {}
    for f in glob.glob(P + pattern):
        d = json.load(open(f)); out[round(d['T0'], 3)] = (d['TH'], d['dc'])
    return out
pf = load('res_eos_pf_*.json'); nu = load('res_phys_*.json')     # physical curve (pbhsim/make_phys_curve.py)
print('\nfull model, physical curve (C_l-Gaussian / dm-Gaussian), power needed:')
for T0 in sorted(nu, reverse=True):
    if T0 not in pf: continue
    d0, d1 = pf[T0][1], nu[T0][1]; c0, c1 = Cl(d0), Cl(d1)
    rs = [ratio_gauss(c0, c1, b) for b in (1e-9, 1e-12, 1e-15)]; rd = [ratio_gauss(d0, d1, b) for b in (1e-9, 1e-12, 1e-15)]
    print(f"  T_H={nu[T0][0]:7.3f}: C_l shift {100*(c1/c0-1):5.2f}%  ratios " + ' '.join(f'{r:.3g}' for r in rs)
          + '  | dm: ' + ' '.join(f'{r:.3g}' for r in rd) + f"  power +{100*((c1/c0)**2-1):.1f}%")
