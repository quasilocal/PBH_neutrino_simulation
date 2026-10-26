import os
import numpy as np, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linear_nu import rhs
from scipy.integrate import solve_ivp
def first_peak(Rnu, x0=0.1, lmax=40):
    psi = 2/3; phi = psi; d = -2*psi - (2/3)*x0**2*phi; th = 0.5*x0*psi
    y0 = np.zeros(5+lmax-1); y0[:5] = [phi, d, th, d, th]
    xs = np.linspace(x0, 8, 8001)
    sol = solve_ivp(rhs, (x0, 8), y0, t_eval=xs, args=(Rnu, lmax), method='DOP853', rtol=1e-10, atol=1e-13)
    return np.max(np.abs(-(2/3)*xs**2*sol.y[0]))
f0 = first_peak(0.0)
Rs = np.linspace(0, 0.52, 14)
out = [(R, 100*(f0/first_peak(R) - 1)) for R in Rs]
for R, s in out: print(f"R={R:.3f}: linear compression-peak proxy shift {s:.2f}%")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figs', 'linproxy_R.json'), 'w'))
