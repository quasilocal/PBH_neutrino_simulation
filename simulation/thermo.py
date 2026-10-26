"""Thermodynamics of the photon + e+e- plasma (mu = 0) and decoupled neutrinos (N_eff = 3, instantaneous
decoupling while e+- are relativistic). Units: MeV, hbar = c = k_B = 1."""
import numpy as np
from scipy.integrate import quad
ME = 0.51099895

def epm(T):
    """e+e- energy density and pressure (g = 4)."""
    if T < ME/60:
        return 0.0, 0.0
    f = lambda p, w: p*p*np.sqrt(p*p+ME*ME)/(np.exp(np.sqrt(p*p+ME*ME)/T)+1.0) if w == 0 else p**4/np.sqrt(p*p+ME*ME)/(np.exp(np.sqrt(p*p+ME*ME)/T)+1.0)
    pmax = 60*T + 10*ME
    rho = (2/np.pi**2)*quad(f, 0, pmax, args=(0,), limit=200, epsabs=0, epsrel=1e-11)[0]
    p = (2/(3*np.pi**2))*quad(f, 0, pmax, args=(1,), limit=200, epsabs=0, epsrel=1e-11)[0]
    return rho, p

def tables(Tmax=200.0, Tmin=0.004, n=2400):
    T = np.exp(np.linspace(np.log(Tmax), np.log(Tmin), n))
    rg = (np.pi**2/15)*T**4
    re, pe = np.array([epm(x) for x in T]).T
    rpl = rg + re; ppl = rg/3 + pe
    spl = (rpl + ppl)/T
    s_rel = (2*np.pi**2/45)*5.5*T**3                  # plasma entropy if e+- were massless
    Tnu = T*(spl/s_rel)**(1/3)                         # neutrino temperature (entropy conservation)
    rnu = 3*(7/8)*2*(np.pi**2/30)*Tnu**4
    return dict(T=T, rpl=rpl, ppl=ppl, spl=spl, rnu=rnu, Tnu=Tnu)

if __name__ == '__main__':
    tb = tables(n=400)
    T = tb['T']
    for Tq in [10, 2, 1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]:
        i = np.argmin(abs(T-Tq))
        Rnu = tb['rnu'][i]/(tb['rnu'][i]+tb['rpl'][i])
        wpl = tb['ppl'][i]/tb['rpl'][i]
        wtot = (tb['ppl'][i]+tb['rnu'][i]/3)/(tb['rpl'][i]+tb['rnu'][i])
        print(f"T={T[i]:.3f} MeV: R_nu={Rnu:.4f} w_plasma={wpl:.4f} w_total={wtot:.4f} Tnu/T={tb['Tnu'][i]/T[i]:.4f}")
    np.savez('thermo_tables.npz', **tables())
