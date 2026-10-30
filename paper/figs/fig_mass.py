import os
import json, glob, numpy as np
import style; style.setup()
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, NullLocator
from scipy.special import erfc, erfcinv

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'simulation') + '/'
TB = np.load(P + 'thermo_tables.npz')
order = np.argsort(TB['T']); Tt = TB['T'][order]
rtot = (TB['rpl'] + TB['rnu'])[order]; wpl = (TB['ppl']/TB['rpl'])[order]
G, c, Msun, MeV4 = 6.674e-11, 2.998e8, 1.989e30, 2.3201e8      # MeV^4 -> kg m^-3 (energy density / c^2)
def MH(TH):
    rho = np.interp(TH, Tt, rtot)*MeV4
    return c**3/(2*G*np.sqrt(8*np.pi*G*rho/3))/Msun

def load(pattern):
    out = {}
    for f in glob.glob(P + pattern):
        d = json.load(open(f)); out[round(d['T0'], 3)] = (d['TH'], d['dc'], d['w']/2)
    return out
pf = load('res_eos_pf_*.json')
nu = {Td: load(f'res_eosc_{Td}_*.json') for Td in (0.75, 1.5, 2.0)}       # relaxation model
phys = load('res_phys_*.json')                                           # physical curve (make_phys_curve.py)
mgp = load('res_mg_phys6_*.json')                                        # multigroup points
rad_fluid = 0.49773

def series(dct):
    T0 = sorted(dct, reverse=True)
    return np.array([MH(dct[t][0]) for t in T0]), np.array([dct[t][1] for t in T0]), T0

XL = (3.0e2, 1.5e8)
Tg = np.exp(np.linspace(np.log(80.0), np.log(0.01), 600)); Mg = np.array([MH(t) for t in Tg])   # M_H(T) with the true g_*(T)
m2t = lambda m: np.exp(np.interp(np.log(np.maximum(m, 1e-30)), np.log(Mg), np.log(Tg)))
t2m = lambda t: np.exp(np.interp(-np.log(np.maximum(t, 1e-30)), -np.log(Tg), np.log(Mg)))
soft = Tg[wpl[np.searchsorted(Tt, Tg).clip(0, len(Tt) - 1)] < 0.32]      # e+e- annihilation: plasma p/e < 0.32
band = (MH(soft.max()), MH(soft.min()))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(style.TXT_W, 2.55))
for ax in (a1, a2):
    ax.axvspan(*band, color='#efeee9', lw=0, zorder=0)
    ax.set_xscale('log'); ax.set_xlim(*XL)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12)); ax.xaxis.set_minor_formatter(NullFormatter())
a2.set_xlabel(r'horizon mass at horizon crossing $M_H$ [$M_\odot$]')
a1.set_xlabel(r'horizon mass at horizon crossing $M_H$ [$M_\odot$]')
for ax in (a1, a2):
    top = ax.secondary_xaxis('top', functions=(m2t, t2m))
    top.xaxis.set_major_locator(FixedLocator([10, 3, 1, 0.3, 0.1])); top.xaxis.set_minor_locator(NullLocator())
    top.set_xticklabels(['10', '3', '1', '0.3', '0.1'])
    top.set_xlabel(r'temperature at horizon crossing $T_H$ [MeV]', fontsize=7.5, labelpad=3)
    top.spines['top'].set_visible(True); top.spines['top'].set_linewidth(0.5); top.spines['top'].set_color(style.AXIS)

# (a) threshold against horizon mass
mp, dp, _ = series(pf)
a1.axhline(rad_fluid, color=style.MUTED, lw=0.6)
a1.text(1.25e8, rad_fluid + 0.0008, r'perfect fluid, $p=e/3$', fontsize=7.0, color=style.INK, va='bottom', ha='right')
a1.text(np.sqrt(band[0]*band[1]), 0.5505, r'$e^\pm$ annihilation', fontsize=7.0, color=style.INK, ha='center', va='top')
m15, d15, _ = series(nu[0.75]); m3, d3, _ = series(nu[1.5])
mlo = np.logspace(np.log10(max(m15.min(), m3.min())), np.log10(min(m15.max(), m3.max())), 400)
lo = np.interp(np.log(mlo), np.log(m15), d15); hi = np.interp(np.log(mlo), np.log(m3), d3)
a1.fill_between(mlo, np.minimum(lo, hi), np.maximum(lo, hi), color=style.BLUE, alpha=0.17, lw=0, label=style.tex(r'$T_{\rm dec}=0.75$--$1.5$\,MeV'))
a1.plot(mp, dp, color=style.INK2, lw=1.2, label='perfect fluid')
a1.plot(mp, dp, **style.marker_kw(style.INK2, 'o', 3.8))
m2, d2, _ = series(nu[2.0])
a1.plot(m2, d2, color=style.BLUE, lw=0.8, ls=(0, (3.5, 1.8)), label=style.tex(r'$T_{\rm dec}=2$\,MeV'))
mph, dph, _ = series(phys)
a1.plot(mph, dph, color=style.BLUE, lw=1.3, label='weak rates, energy groups')
mm, dm, _ = series(mgp)
a1.plot(mm, dm, **style.marker_kw(style.BLUE, 'o', 3.8))
a1.set_ylim(0.481, 0.553); a1.set_yticks([0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55])
a1.set_ylabel(r'threshold $\delta_{\rm c}$')
h, l = a1.get_legend_handles_labels()
order = [3, 0, 2, 1]
a1.legend([h[i] for i in order], [l[i] for i in order], loc='upper left', borderaxespad=0.2, handlelength=1.8)
style.label(a1, '(a)')

# (b) abundance ratio at fixed primordial spectrum
m2, d2, T02 = series(phys)
keep = [i for i, t in enumerate(T02) if t in pf]
m2c, d2c = m2[keep], d2[keep]; dpf = np.array([pf[T02[i]][1] for i in keep])
a2.axhline(1.0, color=style.MUTED, lw=0.6)
lab = []
Cl = lambda d: (1 - np.sqrt(1 - 1.5*d))/0.75      # linear compaction: delta_m = C_l - (3/8) C_l^2, C_l Gaussian for Gaussian zeta
for beta, col in zip([1e-9, 1e-12, 1e-15], style.RAMP):
    x = np.sqrt(2)*erfcinv(beta)
    ratio = erfc(Cl(d2c)/Cl(dpf)*x/np.sqrt(2))/beta
    a2.plot(m2c, ratio, color=col, lw=1.3)
    a2.plot(m2c, ratio, **style.marker_kw(col, 'o', 3.8))
    lab.append((ratio, beta))
a2.set_xlim(*XL)
for ratio, beta in lab:                     # direct labels on the plateau, just above each curve
    y = np.exp(np.interp(np.log(2.2e6), np.log(m2c), np.log(ratio)))
    a2.text(2.2e6, y*1.35, r'$\beta_{\rm pf}=10^{%d}$' % int(np.log10(beta)), fontsize=7.5, color=style.INK, ha='center', va='bottom')
a2.set_yscale('log'); a2.set_ylim(3e-4, 2.0)
a2.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1])); a2.set_yticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$'])
a2.yaxis.set_minor_formatter(NullFormatter())
a2.set_ylabel(r'probability ratio $\beta_\nu/\beta_{\rm pf}$')
style.label(a2, '(b)')
fig.savefig(style.out('fig_mass.pdf')); print('saved', fig.get_size_inches(), 'band', band)
