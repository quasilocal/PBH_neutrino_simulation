import os
import numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linear_nu import rhs
from scipy.integrate import solve_ivp
def first_peak(Rnu, x0, lmax=40):
    psi = 2/3; phi = psi; d = -2*psi - (2/3)*x0**2*phi; th = 0.5*x0*psi
    y0 = np.zeros(5+lmax-1); y0[:5] = [phi, d, th, d, th]
    xs = np.linspace(x0, 8, 8001)
    sol = solve_ivp(rhs, (x0, 8), y0, t_eval=xs, args=(Rnu, lmax), method='DOP853', rtol=1e-10, atol=1e-13)
    D = -(2/3)*xs**2*sol.y[0]
    return np.max(np.abs(D))
for x0 in [0.05, 0.1, 0.25, 0.5]:
    f = first_peak(0.0, x0)
    print(f"k tau_dec = {x0}: first-peak ratio nu/fluid: R=0.409 {first_peak(0.409, x0)/f:.4f}, R=0.488 {first_peak(0.488, x0)/f:.4f}")
