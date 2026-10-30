import os
import json, glob, numpy as np
import style; style.setup()
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator
from scipy.special import erfc, erfcinv

# ---------- (a) mass function for Gaussian zeta (from pbhsim/massfn/mass_function.py)
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'simulation', 'massfn', 'mass_functions.json')))
M = np.array(d['M'])
f = {k: np.array(v) for k, v in d.items() if k not in ('M', 's0')}
band = (3.33e5, 5.09e7)       # e+e- annihilation: plasma p/e < 0.32 at horizon crossing (as in Fig. 4)

# ---------- (b) abundance ratio for different statistics, from the physical threshold curve (res_phys_*.json:
#            multigroup thresholds and the single-rate model with T_dec = 1 MeV, make_phys_curve.py)
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'simulation') + '/'
TB = np.load(P + 'thermo_tables.npz'); o = np.argsort(TB['T']); Tt = TB['T'][o]; rt = (TB['rpl'] + TB['rnu'])[o]
G, c, Msun, MeV4 = 6.674e-11, 2.998e8, 1.989e30, 2.3201e8
MHof = lambda TH: c**3/(2*G*np.sqrt(8*np.pi*G*np.interp(TH, Tt, rt)*MeV4/3))/Msun
def load(pat):
    out = {}
    for fn in glob.glob(P + pat):
        q = json.load(open(fn)); out[round(q['T0'], 3)] = (q['TH'], q['dc'])
    return out
pf, nu = load('res_eos_pf_*.json'), load('res_phys_*.json')      # physical curve (pbhsim/make_phys_curve.py)
keys = sorted((k for k in nu if k in pf), reverse=True)
MHs = np.array([MHof(nu[k][0]) for k in keys])
Cl = lambda x: (1 - np.sqrt(1 - 1.5*x))/0.75
Dl = np.array([Cl(nu[k][1])/Cl(pf[k][1]) - 1 for k in keys])     # shift of the threshold amplitude of zeta
beta = 1e-12
xg = np.sqrt(2)*erfcinv(beta)
stats = [('Gaussian', erfc((1 + Dl)*xg/np.sqrt(2))/beta),
         (r'stretched exp., $\eta=0.6$', beta**((1 + Dl)**0.6 - 1)),
         (r'power law, $n=5$', (1 + Dl)**-5.0)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(style.TXT_W, 2.6))
a2.axvspan(*band, color='#efeee9', lw=0, zorder=0)
MU_EDGE = 2e4     # Gaussian zeta at this amplitude violates the FIRAS mu bound above M_H ~ 3e4 Msun, i.e. M ~ 2e4 Msun (Sec. V A)
a1.axvspan(MU_EDGE, 2e8, facecolor='none', edgecolor=style.GRID, hatch='////', lw=0, zorder=0)
for ax in (a1, a2):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

pfa, nua, lo, hi = f['flat_pf'], f['flat_nu'], f['flat_nulo'], f['flat_nuhi']
n = np.interp(np.log(1e3), np.log(M), pfa)
a1.fill_between(M, np.minimum(lo, hi)/n, np.maximum(lo, hi)/n, color=style.BLUE, alpha=0.17, lw=0,
                label=style.tex(r'$T_{\rm dec}=0.75$--$1.5$\,MeV'))
a1.plot(M, pfa/n, color=style.INK2, lw=1.2, label='perfect fluid')
a1.plot(M, nua/n, color=style.BLUE, lw=1.3, label=r'decoupling neutrinos')
a1.set_xlim(1e2, 2e8)
a1.set_ylim(3e-6, 3); a1.yaxis.set_major_locator(FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]))
a1.set_yticklabels([r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$'])
a1.set_ylabel(r'mass function $\psi(M)$')
a1.set_xlabel(r'black-hole mass $M$ [$M_\odot$]')
a1.text(5e6, 2.5e-3, r'$e^\pm$ annihilation', fontsize=7.0, color=style.INK, ha='center', va='top')
a1.text(1.3e6, 1.6, r'excluded by $\mu$ distortions', fontsize=7.0, color=style.INK, ha='center', va='top',
        bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
h, l = a1.get_legend_handles_labels()
a1.legend([h[2], h[0], h[1]], [l[2], l[0], l[1]], loc='lower left', borderaxespad=0.2, handlelength=1.4)
style.label(a1, r'(a) Gaussian $\zeta$, scale-invariant spectrum')

a2.axhline(1.0, color=style.MUTED, lw=0.6)
for (lab, ratio), col in zip(stats, style.RAMP[::-1]):
    a2.plot(MHs, ratio, color=col, lw=1.3)
    a2.plot(MHs, ratio, **style.marker_kw(col, 'o', 3.8))
    y = np.exp(np.interp(np.log(1.2e6), np.log(MHs), np.log(ratio)))
    below = True
    a2.text(1.2e6, y/1.25 if below else y*1.25, lab, fontsize=7.5, color=style.INK, ha='center',
            va='top' if below else 'bottom')
a2.set_xlim(1e2, 2e8)
a2.set_ylim(1e-3, 2.0); a2.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1]))
a2.set_yticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$'])
a2.set_ylabel(r'probability ratio $\beta_\nu/\beta_{\rm pf}$')
a2.set_xlabel(r'horizon mass at horizon crossing $M_H$ [$M_\odot$]')
style.label(a2, r'(b) three statistics, $\beta_{\rm pf}=10^{-12}$')
fig.savefig(style.out('fig_massfn.pdf')); print('saved', fig.get_size_inches())
for lab, ratio in stats:
    print(lab, ' '.join(f'{m:.2g}:{r:.3g}' for m, r in zip(MHs, ratio)))
