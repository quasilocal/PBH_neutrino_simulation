import os
import numpy as np
from run_nu import run
from dyn_runs_diag import diag
o = run(0.535, fnu=0.409, K=32, N=400, eps0=0.1, eps_max=9.0, hist_every=10, diag=diag, disperse_rule=False)
print('nu 0.535', o['status'], round(o['eps'], 2), flush=True)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figs', 'dyn_nu0535.npy'), np.array(o['hist2']))
