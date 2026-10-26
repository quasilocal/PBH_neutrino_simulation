"""
Linear adiabatic mode in a radiation-dominated universe: tightly coupled photon fluid (w=1/3)
plus massless free-streaming neutrinos (fraction R_nu), conformal Newtonian gauge
(Ma & Bertschinger 1995 conventions, k=1, tau = k*tau). Normalized to zeta = 1.
"""
import numpy as np
from scipy.integrate import solve_ivp

def rhs(tau, y, Rnu, lmax):
    Rg = 1.0 - Rnu
    phi, dg, tg, dn, tn = y[:5]
    F = np.zeros(lmax+2)              # F[2..lmax]
    F[2:lmax+1] = y[5:5+lmax-1]
    sig = F[2]/2
    psi = phi - 6.0*Rnu*sig/tau**2
    dphi = -psi/tau + 2.0*(Rg*tg + Rnu*tn)/tau**2
    ddg = -4/3*tg + 4*dphi
    dtg = dg/4 + psi
    ddn = -4/3*tn + 4*dphi
    dtn = dn/4 - sig + psi
    dF = np.zeros(lmax+1)
    F[lmax+1] = (2*lmax+1)*F[lmax]/tau - F[lmax-1]
    dF[2] = 8/15*tn - 3/5*F[3]
    for l in range(3, lmax+1):
        dF[l] = (l*F[l-1] - (l+1)*F[l+1])/(2*l+1)
    return np.concatenate([[dphi, ddg, dtg, ddn, dtn], dF[2:lmax+1]])

def initial(Rnu, tau0, lmax):
    C = 0.5                                  # zeta = 2C = 1
    psi = 20*C/(15+4*Rnu); phi = (1+0.4*Rnu)*psi
    th = 0.5*tau0*psi; F2 = 2*(tau0**2*psi/15)
    # fix the O(tau0^2) part of delta so the 00 (Poisson) constraint holds exactly at tau0;
    # otherwise a constraint-violating mode Delta - Delta_Poisson ∝ tau is excited (0.5% error at tau=40)
    d = -2*psi - (2/3)*tau0**2*phi
    y = np.zeros(5+lmax-1); y[:5] = [phi, d, th, d, th]; y[5] = F2
    return y

def evolve(Rnu, tau_max=40.0, lmax=40, tau0=1e-3, n=20001):
    t = np.linspace(tau0, tau_max, n)
    sol = solve_ivp(rhs, (tau0, tau_max), initial(Rnu, tau0, lmax), t_eval=t, args=(Rnu, lmax),
                    method='DOP853', rtol=1e-10, atol=1e-13)
    phi = sol.y[0]; Rg = 1-Rnu
    dg, tg, dn, tn = sol.y[1], sol.y[2], sol.y[3], sol.y[4]
    sig = sol.y[5]/2; psi = phi - 6*Rnu*sig/t**2
    th_tot = Rg*tg + Rnu*tn
    Delta = -(2/3)*t**2*phi                          # total comoving density contrast (Poisson)
    Dg = dg + 4*th_tot/t; Dn = dn + 4*th_tot/t       # component contrasts on total-comoving slices
    return dict(t=t, phi=phi, psi=psi, Delta=Delta, Dg=Dg, Dn=Dn, Rnu=Rnu)

if __name__ == '__main__':
    res = {R: evolve(R) for R in [0.0, 0.409, 0.488]}      # R_nu after and before e+- annihilation, as in the paper
    r0 = res[0.0]
    # check against the analytic perfect-fluid solution Delta = -6 phi0 (sin x/x - cos x), x = tau/sqrt3
    x = r0['t']/np.sqrt(3); phi0 = 2/3
    an = -6*phi0*(np.sin(x)/x - np.cos(x))
    print('perfect-fluid check, max |numeric - analytic| =', np.max(np.abs(r0['Delta']-an)))
    for R, r in res.items():
        D = r['Delta']; t = r['t']
        i = np.argmax(np.abs(D[t < 8]))          # first extremum of the total comoving contrast
        sup = r['phi'][0]
        print(f"R_nu={R:5.3f}: superhorizon phi/zeta={sup:.4f} (ratio to fluid {sup/r0['phi'][0]:.4f}); "
              f"first max |Delta|={abs(D[i]):.4f} at tau={t[i]:.2f} (ratio {abs(D[i])/np.max(np.abs(r0['Delta'][r0['t']<8])):.4f}); "
              f"|Delta| at tau=1,2,3: {abs(np.interp(1,t,D)):.4f},{abs(np.interp(2,t,D)):.4f},{abs(np.interp(3,t,D)):.4f}; "
              f"late amplitude ratio(t 25-40) {np.max(np.abs(D[t>25]))/np.max(np.abs(r0['Delta'][r0['t']>25])):.3f}")
    np.savez('linear_nu.npz', **{f"{k}_{R}": v for R, r in res.items() for k, v in r.items() if k != 'Rnu'})
