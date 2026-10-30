import numpy as np
import style; style.setup()
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

d = np.load('dyn_data.npz'); fl, nu = d['fluid'], d['nu']
nu2 = np.load('dyn_nu0535.npy')
# columns: eps, C_max, plasma centre contrast, neutrino centre contrast, Pr/e_nu (core), max 2M/R (collapsing shells)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(style.COL_W, 3.9), sharex=True, height_ratios=[1, 1.05])

# (a) peak of the compaction function
a1.plot(fl[:, 0], fl[:, 1], color=style.INK2, label=r'perfect fluid')
a1.plot(nu[:, 0], nu[:, 1], color=style.BLUE, label=r'free-streaming $\nu$')
a1.plot(nu2[:, 0], nu2[:, 1], color=style.BLUE, lw=1.0, ls=(0, (4, 1.6)), label=r'free-streaming $\nu$, $\delta_m=0.535$')
for h, c in [(fl, style.INK2), (nu2, style.BLUE)]:
    a1.plot(h[-1, 0], h[-1, 1], **style.marker_kw(c, 'o', 5))
a1.text(5.25, 0.93, 'black holes', fontsize=7.5, color=style.INK, ha='left', va='center')
for h in (fl, nu2):
    a1.annotate('', xy=(h[-1, 0], h[-1, 1]), xytext=(5.2, 0.93),
                arrowprops=dict(arrowstyle='-', lw=0.5, color=style.MUTED, shrinkA=0, shrinkB=3.5))
a1.set_ylabel(r'peak compaction $C_{\max}$')
a1.set_ylim(0, 1.05); a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
a1.legend(loc='lower left', borderaxespad=0.15)
style.label(a1, '(a)')

# (b) central densities at the same amplitude, each relative to its own background value
a2.semilogy(fl[:, 0], 1 + fl[:, 2], color=style.INK2)
a2.semilogy(nu[:, 0], 1 + nu[:, 2], color=style.BLUE)
a2.semilogy(nu[:, 0], 1 + nu[:, 3], color=style.ORANGE)
a2.text(3.62, 2.2e3, 'perfect fluid', fontsize=7.5, color=style.INK, ha='right', va='center')
a2.text(4.40, 2.6e2, 'plasma', fontsize=7.5, color=style.INK, ha='left', va='center')
a2.text(6.0, 15.0, 'neutrinos', fontsize=7.5, color=style.INK, ha='left', va='center')
a2.set_ylabel(r'central density $\rho/\rho_b$')
a2.set_xlabel(r'$\epsilon = 1/(aHr_m)$')
a2.set_ylim(0.3, 2e4); a2.set_xlim(0, 8)
a2.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
a2.yaxis.set_minor_formatter(NullFormatter())
style.label(a2, '(b)')
fig.savefig(style.out('fig_dynamics.pdf')); print('saved', fig.get_size_inches())
