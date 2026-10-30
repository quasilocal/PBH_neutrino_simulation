import os
import json, numpy as np
import style; style.setup()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'simulation') + '/'
def ld(name): d = json.load(open(P + f'res_{name}.json')); return d['dc'], d['w']/2
fl = ld('fluid')[0]
gauss = [(0.10, ld('nu10_K32')), (0.20, ld('nu20_K32')), (0.30, ld('nu30_K32')), (0.409, ld('nu32')), (0.488, ld('nu488_K32'))]
shapes = [('q03', 0.3), ('q05', 0.5), (None, 1.0), ('q2', 2.0), ('q4', 4.0)]
Cl = lambda d: (1 - np.sqrt(1 - 1.5*d))/0.75     # linear compaction; for a fixed profile shape its threshold is the threshold amplitude of zeta
lin = np.array(json.load(open('linproxy_R.json')))      # linear compression estimate, decoupling at k tau_0 = 0.1
edec = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
sdec = np.array([100*(ld(f'dec_{e}')[0]/fl - 1) for e in edec])
s409 = 100*(ld('nu32')[0]/fl - 1)

fig, (a1, a3, a2) = plt.subplots(1, 3, figsize=(style.TXT_W, 2.3))

# (a) shift against the free-streaming fraction: compaction threshold (circles) and threshold amplitude of zeta (squares)
R = np.linspace(0, 0.52, 50)
a1.plot(lin[:, 0], lin[:, 1], color=style.MUTED, lw=0.9, ls=(0, (4, 2)))
a1.text(0.155, 6.6, 'linear theory,\nsingle mode', fontsize=7.5, color=style.INK, ha='right', va='bottom')
a1.annotate('', xy=(0.235, 5.75), xytext=(0.16, 6.75), arrowprops=dict(arrowstyle='-', color=style.MUTED, lw=0.5))
x = np.array([g[0] for g in gauss]); y = np.array([100*(g[1][0]/fl - 1) for g in gauss])
yz = np.array([100*(Cl(g[1][0])/Cl(fl) - 1) for g in gauss])
print('R_nu scan: dc shifts', np.round(y, 2), ' zeta shifts', np.round(yz, 2), ' zeta/R', np.round(yz/x, 2))
a1.plot(R, 15*R, color=style.BLUE, lw=1.0)
a1.plot(x, y, **style.marker_kw(style.BLUE, 'o', 4.8))
a1.plot(R, 23.5*R, color=style.ORANGE, lw=1.0)
a1.plot(x, yz, **style.marker_kw(style.ORANGE, 's', 4.4))
a1.text(0.36, 10.3, r'linear compaction, $C_l$', fontsize=7.5, color=style.INK, ha='right', va='bottom')
a1.text(0.38, 4.2, r'compaction, $\delta_{\rm c}$', fontsize=7.5, color=style.INK, ha='left', va='top')
a1.set_xlabel(r'free-streaming fraction $R_\nu$')
a1.set_ylabel(r'threshold shift [\%]')
a1.set_xlim(0, 0.52); a1.set_ylim(0, 12.5)
a1.set_yticks([0, 2, 4, 6, 8, 10, 12])
style.label(a1, r'(a) Gaussian profile')

# (b) shape dependence at R_nu = 0.409: threshold of the compaction and threshold amplitude of zeta
av, sd, sz = [], [], []
for tag, qv in shapes:
    f = ld('fl_' + tag)[0] if tag else fl
    n = ld('nu409_' + tag)[0] if tag else ld('nu32')[0]
    av.append(qv); sd.append(100*(n/f - 1)); sz.append(100*(Cl(n)/Cl(f) - 1))
print('shape shifts:', [(a, round(d, 2), round(z, 2)) for a, d, z in zip(av, sd, sz)])
a3.plot(av, sz, color=style.ORANGE, lw=1.0); a3.plot(av, sz, **style.marker_kw(style.ORANGE, 's', 4.4))
a3.plot(av, sd, color=style.BLUE, lw=1.0); a3.plot(av, sd, **style.marker_kw(style.BLUE, 'o', 4.8))
a3.text(1.0, sz[2] + 0.55, r'linear compaction, $C_l$', fontsize=7.5, color=style.INK, ha='center', va='bottom')
a3.text(1.0, sd[2] - 0.65, r'compaction, $\delta_{\rm c}$', fontsize=7.5, color=style.INK, ha='center', va='top')
a3.set_xscale('log'); a3.set_xlim(0.25, 4.8); a3.set_ylim(3.0, 12.5)
a3.xaxis.set_major_locator(FixedLocator(av)); a3.xaxis.set_minor_locator(NullLocator())
a3.xaxis.set_major_formatter(FixedFormatter(['0.3', '0.5', '1', '2', '4']))
a3.set_yticks([4, 6, 8, 10, 12])
a3.set_xlabel(r'profile parameter $\alpha$')
a3.set_ylabel(r'threshold shift [\%]')
style.label(a3, r'(b) profile shape, $R_\nu=0.409$')

# (b) shift against the decoupling time (gradual decoupling, R_nu = 0.409)
a2.axhline(s409, color=style.MUTED, lw=0.9, ls=(0, (4, 2)))
a2.text(10.5, s409 + 0.25, r'instantaneous, $\epsilon_0=0.1$', fontsize=7.5, color=style.INK, ha='right', va='bottom')
a2.axvline(1.0, color=style.MUTED, lw=0.5)
a2.text(1.07, 0.35, 'horizon\ncrossing', fontsize=7.5, color=style.INK, ha='left', va='bottom', linespacing=1.1)
a2.plot(edec, sdec, color=style.BLUE, lw=1.0)
a2.plot(edec, sdec, **style.marker_kw(style.BLUE, 'o', 4.8))
a2.set_xscale('log')
a2.set_xlim(0.18, 11.5); a2.set_ylim(0, 7.6)
a2.xaxis.set_major_locator(FixedLocator(edec)); a2.xaxis.set_minor_locator(NullLocator())
a2.xaxis.set_major_formatter(FixedFormatter(['0.25', '0.5', '1', '2', '4', '8']))
a2.set_yticks([0, 2, 4, 6])
a2.set_xlabel(r'decoupling time $\epsilon_{\rm dec}$')
a2.set_ylabel(r'$\Delta\delta_{\rm c}/\delta_{\rm c}$ [\%]')
style.label(a2, r'(c) gradual decoupling')
fig.savefig(style.out('fig_scaling.pdf')); print('saved', fig.get_size_inches())
