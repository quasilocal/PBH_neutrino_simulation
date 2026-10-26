import os
"""End-to-end PBH mass functions with and without free-streaming neutrinos.
Gaussian zeta; Gaussian variable C_l (linear compaction), delta_m = C_l - (3/8) C_l^2; type-I collapse C_l < 4/3.
Thresholds delta_c(M_H) from the simulated scans; critical scaling M = K M_H (delta_m - delta_c)^gamma.
psi(M) = f_PBH per ln M today (arbitrary normalization, same for both cases)."""
import json, glob, numpy as np
from scipy.special import erfcinv
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + '/'
TB = np.load(P + 'thermo_tables.npz'); o = np.argsort(TB['T']); Tt = TB['T'][o]; rt = (TB['rpl'] + TB['rnu'])[o]
G, c, Msun, MeV4 = 6.674e-11, 2.998e8, 1.989e30, 2.3201e8
MHof = lambda TH: c**3/(2*G*np.sqrt(8*np.pi*G*np.interp(TH, Tt, rt)*MeV4/3))/Msun
RAD = {'pf': 0.49773, 'nu': 0.52849}
def table(pat, kind):
    out = {}
    for f in glob.glob(P + pat):
        d = json.load(open(f)); out[MHof(d['TH'])] = d['dc']
    M = np.array(sorted(out)); d = np.array([out[m] for m in M])
    # below the scan: neutrinos still coupled -> perfect-fluid radiation value; above: relax to the radiation value by 1e9
    lo_val = RAD['pf']
    M = np.concatenate([[1e1], M, [1e9]]); d = np.concatenate([[lo_val], d, [RAD['pf' if kind == 'pf' else 'nu']]])
    return np.log(M), d
# 'nu': the physical curve (multigroup weak rates with nu-nu scattering; make_phys_curve.py);
# band: relaxation model with T_dec = 0.75 and 1.5 MeV
TAB = {'pf': table('res_eos_pf_*.json', 'pf'), 'nu': table('res_phys_*.json', 'nu'),
       'nulo': table('res_eosc_0.75_*.json', 'nu'), 'nuhi': table('res_eosc_1.5_*.json', 'nu')}
def dc_of(kind, MH): lnM, d = TAB[kind]; return np.interp(np.log(MH), lnM, d)
Cl_of = lambda C: (1 - np.sqrt(np.clip(1 - 1.5*C, 0, None)))/0.75

GAMMA, K_PF = 0.36, 4.0
SHIFT_RAD = RAD['nu']/RAD['pf'] - 1      # full free-streaming shift in pure radiation
def K_of(kind, MH):
    """Prefactor of M = K M_H (delta_m - delta_c)^gamma. With free-streaming neutrinos the apparent horizon forms ~10%
    heavier (Sec. IV E); where the neutrinos are still (partly) coupled the increase is scaled by the fraction of the
    full threshold shift reached at that horizon mass, so that K -> K_PF where the neutrinos are coupled."""
    if kind == 'pf': return K_PF*np.ones_like(MH)
    s = np.clip((dc_of(kind, MH)/dc_of('pf', MH) - 1)/SHIFT_RAD, 0.0, 1.0)
    return K_PF*(1 + 0.1*s)

def sigma_flat(MH, s0): return s0*np.ones_like(MH)
def sigma_lognormal(MH, Mp, width, amp):
    """rms of C_l for P_zeta(k) = amp exp(-ln^2(k/k_p)/(2 width^2)); Gaussian window; (k r_m)^2 = M_H/M_k."""
    y = np.linspace(-12, 12, 2001); dy = y[1] - y[0]          # y = ln(k/k_p)
    out = []
    for m in np.atleast_1d(MH):
        kr2 = (m/Mp)*np.exp(2*y)                              # M_k = Mp e^{-2y}
        integrand = (16/81)*kr2**2*np.exp(-kr2/2)*amp*np.exp(-y**2/(2*width**2))
        out.append(np.sqrt(np.sum(integrand)*dy))
    return np.array(out)

def psi(M, kind, sig_fn, lnMH=np.linspace(np.log(3e1), np.log(3e9), 1500)):
    MH = np.exp(lnMH); dl = lnMH[1] - lnMH[0]
    sig = sig_fn(MH); dc = dc_of(kind, MH); w = MH**-0.5; Kk = K_of(kind, MH)
    res = []
    for m in np.atleast_1d(M):
        y = (m/(Kk*MH))**(1/GAMMA)
        C = dc + y
        ok = C < 2/3 - 1e-9
        Cl = Cl_of(np.where(ok, C, 0.0))
        dCl = (y/GAMMA)/(1 - 0.75*Cl)
        PG = np.exp(-Cl**2/(2*sig**2))/(np.sqrt(2*np.pi)*sig)
        res.append(np.sum(np.where(ok, w*PG*(m/MH)*dCl, 0.0))*dl)
    return np.array(res)

if __name__ == '__main__':
    M = np.logspace(2, 8.3, 160)
    # flat spectrum: sigma fixed so that the radiation-era perfect-fluid beta = 1e-12 (erfc estimate)
    beta0 = 1e-12; s0 = Cl_of(RAD['pf'])/(np.sqrt(2)*erfcinv(beta0))
    out = {'M': M.tolist(), 's0': s0}
    for kind in ('pf', 'nu', 'nulo', 'nuhi'):
        out['flat_' + kind] = psi(M, kind, lambda mh: sigma_flat(mh, s0)).tolist()
    # log-normal peak at M_H = 1e6 Msun, width 1; amplitude giving the same peak sigma as the flat case
    Mp, width = 1e6, 1.0
    a1 = sigma_lognormal(np.array([Mp]), Mp, width, 1.0)[0]
    amp = (s0/a1)**2
    for kind in ('pf', 'nu', 'nulo', 'nuhi'):
        out['logn_' + kind] = psi(M, kind, lambda mh: sigma_lognormal(mh, Mp, width, amp)).tolist()
    json.dump(out, open('mass_functions.json', 'w'))
    f = {k: np.array(v) for k, v in out.items() if k != 's0'}
    for lab in ('flat', 'logn'):
        tot_pf = np.trapezoid(f[lab+'_pf'], np.log(M)); tot_nu = np.trapezoid(f[lab+'_nu'], np.log(M))
        ipf, inu = np.argmax(f[lab+'_pf']), np.argmax(f[lab+'_nu'])
        print(f"{lab}: total f ratio nu/pf = {tot_nu/tot_pf:.3g}; peak M pf {M[ipf]:.3g} nu {M[inu]:.3g}; "
              f"ratio at 1e3,1e4,1e5,1e6,1e7: " + ' '.join(f"{np.interp(np.log(x), np.log(M), f[lab+'_nu']/f[lab+'_pf']):.3g}" for x in (1e3,1e4,1e5,1e6,1e7)))
