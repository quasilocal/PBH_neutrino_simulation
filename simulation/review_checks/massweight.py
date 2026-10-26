"""Probability ratio vs mass-weighted ratio for Gaussian C_l in pure radiation (R_nu = 0.409, extrapolated
delta_c values), with M/M_H = K (delta_m - delta_c)^0.36, K = 4 (pf) and 4.4 (nu) or 4 (nu, no K change).
Also the high-peak peak-theory factor (1+Delta)-type correction. Quadrature only."""
import numpy as np
from scipy.special import erfc, erfcinv
from scipy.integrate import quad
Cl = lambda d: (1 - np.sqrt(1 - 1.5*d))/0.75
dm = lambda c: c - 0.375*c*c
dpf, dnu = 0.49773, 0.49773*1.0619
cpf, cnu = Cl(dpf), Cl(dnu)
for b0 in (1e-9, 1e-12, 1e-15):
    s = cpf/(np.sqrt(2)*erfcinv(b0))
    P = lambda c: np.exp(-c*c/(2*s*s))/(np.sqrt(2*np.pi)*s)
    prob = erfc(cnu/(np.sqrt(2)*s))/erfc(cpf/(np.sqrt(2)*s))
    def bm(cc, dc, K):
        return 2*quad(lambda c: K*(dm(c)-dc)**0.36*P(c), cc, 4/3, points=[cc+1e-6, cc+5*s*s/cc], limit=400, epsabs=0, epsrel=1e-10)[0]
    mpf = bm(cpf, dpf, 4.0); mnu = bm(cnu, dnu, 4.4); mnu0 = bm(cnu, dnu, 4.0)
    nu, nup = cpf/s, cnu/s
    peak = ((nup**3 - 3*nup)/(nu**3 - 3*nu))*(nup/nu)
    print(f'beta_pf={b0:.0e}: nu={nu:.2f}  probability ratio {prob:.4f}  mass-weighted ratio (K x1.1) {mnu/mpf:.4f}  (K unchanged) {mnu0/mpf:.4f}'
          f'  mean M/M_H pf {mpf/(2*erfc(cpf/(np.sqrt(2)*s))/2):.3f}  high-peak factor {peak:.3f}')
# typical distance from threshold of forming PBHs and the fraction below 1e-4 (Sec. IV E)
for b0 in (1e-9, 1e-12, 1e-15):
    s = cpf/(np.sqrt(2)*erfcinv(b0)); nu = cpf/s
    typ = (1 - 0.75*cpf)*cpf/nu**2
    print(f'beta={b0:.0e}: typical delta_m - delta_c = {typ:.4f}; fraction below 1e-4 = {1-np.exp(-1e-4/typ):.3f}')
