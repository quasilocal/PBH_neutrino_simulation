"""Probability ratio vs mass-weighted ratio for Gaussian C_l in pure radiation (R_nu = 0.409, extrapolated
delta_c values), with M/M_H = K (delta_m - delta_c)^0.36, K = 5.9 (pf) and 6.49 (nu) or 5.9 (nu, no K change).
The ratios do not depend on the value of K. Also the peak-theory factor: the number density of high peaks above
y = C_l,c/sigma carries the prefactor y^2 - 1, the Gaussian tail 1/y. Quadrature only."""
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
    mpf = bm(cpf, dpf, 5.9); mnu = bm(cnu, dnu, 1.1*5.9); mnu0 = bm(cnu, dnu, 5.9)
    nu, nup = cpf/s, cnu/s
    peak = ((nup**2 - 1)/(nu**2 - 1))*(nup/nu)
    print(f'beta_pf={b0:.0e}: nu={nu:.2f}  probability ratio {prob:.4f}  mass-weighted ratio (K x1.1) {mnu/mpf:.4f}  (K unchanged) {mnu0/mpf:.4f}'
          f'  mean M/M_H pf {mpf/(2*erfc(cpf/(np.sqrt(2)*s))/2):.3f}  high-peak factor {peak:.3f}')
# typical distance from threshold of forming PBHs and the fraction below 1e-4 (Sec. IV E)
for b0 in (1e-9, 1e-12, 1e-15):
    s = cpf/(np.sqrt(2)*erfcinv(b0)); nu = cpf/s
    typ = (1 - 0.75*cpf)*cpf/nu**2
    print(f'beta={b0:.0e}: typical delta_m - delta_c = {typ:.4f}; fraction below 1e-4 = {1-np.exp(-1e-4/typ):.3f}')
