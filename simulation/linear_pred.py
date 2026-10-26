import os
"""Linear-theory prediction for the centre density contrasts (fluid-comoving slices) of the Gaussian-K
initial data, with neutrinos decoupling (isotropic) at tau0 = eps0.  Uses the Boltzmann hierarchy RHS of
linear_nu.py (Ma-Bertschinger, k = 1 units, x = k tau)."""
import numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linear_nu import rhs
from scipy.integrate import solve_ivp

def mode(k, tau0, taus, Rnu, lmax=40):
    x0 = k*tau0
    psi = 2.0/3.0; phi = psi                    # single fluid before decoupling, zeta = 1
    d = -2*psi - (2/3)*x0**2*phi                # Poisson constraint at x0
    th = 0.5*x0*psi
    y0 = np.zeros(5+lmax-1); y0[:5] = [phi, d, th, d, th]
    xs = k*np.asarray(taus)
    sol = solve_ivp(rhs, (x0, xs[-1]), y0, t_eval=xs, args=(Rnu, lmax), method='DOP853', rtol=1e-9, atol=1e-12)
    phi, dg, tg, dn, tn = sol.y[:5]
    Dg = dg + 4*tg/xs; Dn = dn + 4*tg/xs        # fluid-comoving slices (photon-fluid velocity)
    return Dg, Dn, (1-Rnu)*Dg + Rnu*Dn

def centre(Rnu, tau0, taus, delta_m, kmax=10.0, nk=200):
    A = 1.5*np.e*delta_m
    ks = np.linspace(kmax/nk, kmax, nk)
    out = np.zeros((3, len(taus)))
    for k in ks:
        zt = 0.25*A*np.pi**1.5*np.exp(-k*k/4)    # zeta(k) for K(r) = A exp(-r^2), r_m = 1
        res = mode(k, tau0, taus, Rnu)
        for j in range(3):
            out[j] += k*k*zt*res[j]
    return -out*(ks[1]-ks[0])/(2*np.pi**2)     # rows: fluid, neutrino, total (sign: MB zeta>0 is an underdensity)

if __name__ == '__main__':
    taus = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    for Rnu in [0.0, 0.409]:
        c = centre(Rnu, 0.02, taus, 0.01)
        print(f"Rnu={Rnu}:")
        for j, nm in enumerate(['fluid', 'nu', 'total']):
            print(f"   {nm:5s}", ' '.join(f"{v: .5f}" for v in c[j]))
    # superhorizon check: total at small tau should be tau^2 * (2/3) * 3/2 e dm * ... = delta_rho(r=0)*tau^2
    print("gradient-expansion centre value tau^2*(2/3)K(0) at tau=0.5:", 0.25*(2/3)*1.5*np.e*0.01)
