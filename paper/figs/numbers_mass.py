import os
"""Numbers quoted in Secs. IV D and V A, computed from the scan results (abundances with C_l statistics, as in the paper)."""
import json, glob, numpy as np
from scipy.special import erfc, erfcinv
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'simulation') + '/'
TB = np.load(P + 'thermo_tables.npz'); Tt = TB['T'][::-1]; rtot = (TB['rpl'] + TB['rnu'])[::-1]
G, c, Msun, MeV4 = 6.674e-11, 2.998e8, 1.989e30, 2.3201e8
MH = lambda TH: c**3/(2*G*np.sqrt(8*np.pi*G*np.interp(TH, Tt, rtot)*MeV4/3))/Msun
def load(pattern):
    out = {}
    for f in glob.glob(P + pattern):
        d = json.load(open(f)); out[round(d['T0'], 3)] = (d['TH'], d['dc'], d['w']/2)
    return out
pf = load('res_eos_pf_*.json'); nu = {Td: load(f'res_eosc_{Td}_*.json') for Td in (0.75, 1.0, 1.5, 2.0)}
nu['phys'] = load('res_phys_*.json')          # physical curve: multigroup weak rates (make_phys_curve.py)
rad = 0.49773
print(" T0   T_H[MeV]  M_H[Msun]    pf      phys     g0.75    g1.0     g1.5     g2.0   | phys/pf-1  g2/pf-1  pf/rad-1  phys/rad-1")
for T0 in sorted(set(pf) | set(nu['phys']), reverse=True):
    TH = (nu['phys'].get(T0) or pf.get(T0))[0]
    p0 = pf.get(T0, (0, np.nan))[1]
    row = [nu[t].get(T0, (0, np.nan))[1] for t in ('phys', 0.75, 1.0, 1.5, 2.0)]
    print(f"{T0:5.1f} {TH:8.4f}  {MH(TH):9.2e}  {p0:.5f}  " + "  ".join(f"{x:.5f}" for x in row)
          + f"  | {100*(row[0]/p0-1):+6.2f}%  {100*(row[4]/p0-1):+6.2f}%  {100*(p0/rad-1):+6.2f}%  {100*(row[0]/rad-1):+6.2f}%")
Cl = lambda d: (1 - np.sqrt(1 - 1.5*d))/0.75     # Gaussian zeta: the Gaussian variable is C_l, delta_m = C_l - (3/8) C_l^2
print("\nabundance ratio beta_nu/beta_pf for Gaussian zeta (C_l statistics, physical curve), beta_pf = 1e-9, 1e-12, 1e-15:")
for T0 in sorted(nu['phys'], reverse=True):
    if T0 not in pf: continue
    f = Cl(nu['phys'][T0][1])/Cl(pf[T0][1])
    rs = [erfc(f*np.sqrt(2)*erfcinv(b)/np.sqrt(2))/b for b in (1e-9, 1e-12, 1e-15)]
    expo = [np.log(r*b)/np.log(b) for r, b in zip(rs, (1e-9, 1e-12, 1e-15))]
    print(f"  M_H={MH(nu['phys'][T0][0]):9.2e}: " + "  ".join(f"{1/r:7.1f}x" for r in rs) + "   exponents " + " ".join(f"{x:.3f}" for x in expo) + f"   power needed x{f**2:.3f}  C_l shift {100*(f-1):.2f}%")
