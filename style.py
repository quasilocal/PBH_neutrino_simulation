"""Shared figure style for the paper (print, REVTeX 4.2 two-column PRD).
Figures are exported at their final printed width (no rescaling in LaTeX), so all text prints at the sizes set here.
Colour roles follow the dataviz reference palette (validated in light mode: blue/orange/aqua categorical; the
ordinal blue ramp #5598e7/#256abf/#0d366b passes the ordinal checks)."""
import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# LaTeX text rendering needs a TeX installation with the type1cm and cm-super packages; set PBH_USETEX=0 to use
# Matplotlib's built-in mathtext instead (same figures, slightly different fonts).
USETEX = os.environ.get('PBH_USETEX', '1') != '0'

def tex(s):
    """Text-mode LaTeX that Matplotlib's mathtext does not understand (en dash, thin and explicit spaces) is
    converted to plain characters when LaTeX is not used."""
    if USETEX:
        return s
    return s.replace('--', '\u2013').replace('\\,', '\u2009').replace('\\ ', ' ')

def out(name):
    """Figures are written next to paper.tex, one level above this folder, or into the folder named by the
    environment variable PBH_FIGDIR (used by reproduce.py, so that the shipped figures are not overwritten)."""
    d = os.environ.get('PBH_FIGDIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)

COL_W = 246.0/72.27     # REVTeX PRD \columnwidth [in]
TXT_W = 510.0/72.27     # REVTeX PRD \textwidth [in]

INK = '#0b0b0b'        # primary text
INK2 = '#52514e'       # secondary ink; the perfect-fluid series
MUTED = '#898781'      # reference lines and secondary annotations
GRID = '#e1e0d9'       # hairline grid
AXIS = '#52514e'       # axes and ticks
BLUE = '#2a78d6'       # this work: free-streaming / decoupling neutrinos
ORANGE = '#eb6834'     # neutrino component; second category
AQUA = '#1baf7a'       # third category (labelled use only)
RAMP = ['#3a82d8', '#1f5aa6', '#0e2c55']   # ordinal blue ramp, light -> dark (validated: monotone L, light end 3.9:1 on white)

def setup():
    plt.rcParams.update({
        'text.usetex': USETEX,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}' if USETEX else '',
        'font.family': 'serif',
        'font.size': 8.5,
        'axes.labelsize': 8.5,
        'axes.titlesize': 8.5,
        'legend.fontsize': 7.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK,
        'axes.linewidth': 0.5,
        'axes.titlelocation': 'left',
        'axes.titlepad': 3.0,
        'xtick.color': AXIS, 'ytick.color': AXIS,
        'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4, 'grid.linestyle': '-',
        'axes.axisbelow': True,
        'lines.linewidth': 1.3, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
        'lines.markersize': 4.4, 'lines.markeredgewidth': 0.7,
        'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.5,
        'legend.borderaxespad': 0.3, 'legend.labelspacing': 0.35,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02,
        'figure.constrained_layout.hspace': 0.02, 'figure.constrained_layout.wspace': 0.02,
        'savefig.dpi': 600, 'savefig.bbox': None, 'savefig.pad_inches': 0.0,
        'pdf.fonttype': 42,
    })

def marker_kw(color, marker='o', size=4.4):
    """Filled marker with a white ring (surface gap) so it stays legible where it crosses lines."""
    return dict(marker=marker, markersize=size, markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=0.7, linestyle='none', color=color)

def label(ax, s):
    """Panel label, always top-left above the axes."""
    ax.set_title(s, loc='left', fontsize=8.5, color=INK)
