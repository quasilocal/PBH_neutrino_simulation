import numpy as np
import style; style.setup()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = np.load('lin_offcentre.npz')
taus, L = d['taus'], d['L']            # L rows: plasma, neutrinos, total (per unit delta_m)

def extrap(K, comp):
    """Zero-amplitude extrapolation of the simulated contrast per unit delta_m, interpolated onto taus."""
    a, b = d[f'K{K}_0.005'], d[f'K{K}_0.0025']
    fa = np.interp(taus, a[:, 0], a[:, 2 + comp])/0.005
    fb = np.interp(taus, b[:, 0], b[:, 2 + comp])/0.0025
    return 2*fb - fa

fig, (a1, a2) = plt.subplots(2, 1, figsize=(style.COL_W, 3.35), sharex=True, height_ratios=[2.0, 1.0])
cols = [style.BLUE, style.ORANGE]
for comp in (0, 1):
    a1.plot(taus, L[comp], color=cols[comp], lw=1.2)
    s = extrap(64, comp)
    sel = slice(1, None, 2)
    a1.plot(taus[sel], s[sel], **style.marker_kw(cols[comp], 'o', 4.2))
    norm = np.max(np.abs(L[comp]))
    a2.plot(taus, 100*(extrap(64, comp) - L[comp])/norm, color=cols[comp], lw=1.2)
    a2.plot(taus, 100*(extrap(32, comp) - L[comp])/norm, color=cols[comp], lw=0.8, ls=(0, (3, 1.5)))
a1.set_ylabel(r'$\delta(r=0.4\,r_m)/\delta_m$'); a1.set_ylim(0, 3.1); a1.set_yticks([0, 1, 2, 3])
hl = [Line2D([], [], color=style.INK2, lw=1.2), Line2D([], [], **style.marker_kw(style.INK2, 'o', 4.2))]
a1.legend(hl, ['linear theory', r'simulation ($N_\mu=64$, $\delta_m\to0$)'], loc='upper left', ncol=2, columnspacing=1.2,
          borderaxespad=0.1)
a1.text(3.22, 0.72, 'plasma', fontsize=7.5, color=style.INK, ha='right', va='center')
a1.text(3.28, 1.80, 'neutrinos', fontsize=7.5, color=style.INK, ha='left', va='center')
style.label(a1, '(a)')
a2.axhline(0, color=style.MUTED, lw=0.5)
a2.set_ylim(-0.6, 0.6); a2.set_yticks([-0.5, 0, 0.5])
a2.set_ylabel(r'residual [\%]')
a2.set_xlabel(r'$\epsilon = 1/(aHr_m)$')
a2.set_xlim(0, 4.1)
a2.text(0.985, 0.06, style.tex(r'solid: $N_\mu=64$,\ \ dashed: $N_\mu=32$'), transform=a2.transAxes, ha='right', va='bottom', fontsize=7.5,
        color=style.INK)
style.label(a2, '(b)')
fig.savefig(style.out('fig_linear.pdf')); print('saved', fig.get_size_inches())
