import numpy as np, json, glob
from scipy.special import erfc, erfcinv
TB = np.load('thermo_tables.npz')
T = TB['T'][::-1]; rtot = (TB['rpl']+TB['rnu'])[::-1]
G = 6.674e-11; c = 2.998e8; Msun = 1.989e30; MeV4 = 2.32e8      # kg/m^3 per MeV^4
def MH(TH):
    rho = np.interp(TH, T, rtot)*MeV4
    H = np.sqrt(8*np.pi*G*rho/3)
    return c**3/(2*G*H)/Msun
rad_fl, rad_nu = 0.49773, 0.52832
res = {}
for f in glob.glob('res_eos_*.json'):
    d = json.load(open(f)); res.setdefault(d['T0'], {})[d['kind']] = d
rows = []
for T0 in sorted(res, reverse=True):
    r = res[T0]
    if 'pf' in r and 'fs' in r:
        TH = r['fs']['TH']; dpf = r['pf']['dc']; dfs = r['fs']['dc']
        rows.append((TH, MH(TH), dpf, dfs))
print(" T_H[MeV]   M_H[Msun]    pf       fs     fs/pf-1   pf/rad-1  fs/rad-1")
for TH, M, dpf, dfs in rows:
    print(f" {TH:7.4f}  {M:9.2e}  {dpf:.5f}  {dfs:.5f}  {dfs/dpf-1:+.3%}  {dpf/rad_fl-1:+.3%}  {dfs/rad_fl-1:+.3%}")
print("\nabundance ratio beta_fs/beta_pf at fixed P_zeta (beta ~ erfc(delta_c/sqrt2 sigma)), sigma set by beta_pf:")
for TH, M, dpf, dfs in rows:
    out = []
    for beta in [1e-9, 1e-12, 1e-15]:
        nu = np.sqrt(2)*erfcinv(beta); f = dfs/dpf
        out.append(f"beta={beta:.0e}: x{erfc(f*nu/np.sqrt(2))/beta:.2e}")
    print(f" T_H={TH:.3f} (M_H={M:.1e}): " + '; '.join(out))
json.dump(rows, open('eos_rows.json', 'w'))
