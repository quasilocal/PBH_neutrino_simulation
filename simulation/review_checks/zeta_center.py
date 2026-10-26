"""zeta(0) and zeta(r_m) for K(r) = A exp(-(r/r_m)^(2 alpha)/alpha), delta_m = (2/3) K(r_m) r_m^2, r_m = 1.
d zeta / d ln r = 1 - 1/sqrt(1 - K r^2)  (r: areal-type coordinate of K; zeta -> 0 at infinity)."""
import numpy as np
from scipy.integrate import quad
Cl = lambda d: (1 - np.sqrt(1 - 1.5*d))/0.75
def K(r, d, a): return 1.5*d*np.exp(1/a)*np.exp(-r**(2*a)/a)
def integrand(lnr, d, a):
    r = np.exp(lnr); x = K(r, d, a)*r*r
    return 1/np.sqrt(1 - x) - 1
def zeta(r0, d, a):
    return quad(integrand, np.log(r0), np.log(50.0), args=(d, a), limit=400, epsabs=1e-13, epsrel=1e-11)[0]
def zeta0(d, a):
    return quad(integrand, -60.0, np.log(50.0), args=(d, a), limit=800, epsabs=1e-13, epsrel=1e-11)[0]
tab = {0.3: (0.44805, 0.47747), 0.5: (0.46687, 0.49682), 1.0: (0.49773, 0.52832), 2.0: (0.53274, 0.56342), 4.0: (0.56819, 0.59734)}
print(' alpha   dc shift   Cl shift   zeta(0) pf -> nu   shift    zeta(r_m) pf -> nu  shift   max K r^2 (nu)')
for a, (dp, dn) in tab.items():
    z0p, z0n = zeta0(dp, a), zeta0(dn, a)
    zmp, zmn = zeta(1.0, dp, a), zeta(1.0, dn, a)
    rr = np.linspace(1e-4, 5, 20001); mx = np.max(K(rr, dn, a)*rr**2)
    print(f'{a:5.1f}  {100*(dn/dp-1):6.2f}%   {100*(Cl(dn)/Cl(dp)-1):6.2f}%   {z0p:.4f} -> {z0n:.4f}  {100*(z0n/z0p-1):6.2f}%   '
          f'{zmp:.4f} -> {zmn:.4f}  {100*(zmn/zmp-1):6.2f}%   {mx:.4f}')
