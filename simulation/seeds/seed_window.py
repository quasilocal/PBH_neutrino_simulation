import os
"""Distortion-capped PBH abundance for non-Gaussian tails (setup of Dave et al. 2026, arXiv:2607.03138):
beta_max(M) = 2 S(u_c), u_c = zeta_c/sigma_max(M), sigma_max^2 = mu_lim/(2.2 W_mu(M)), zeta_c = 0.67 (perfect fluid).
With free-streaming neutrinos the threshold amplitude of zeta rises by Delta(M_H) (from our mass scan, C_l shift)."""
import json, glob, numpy as np, mpmath as mp
from scipy.integrate import quad
from scipy.optimize import brentq
mp.mp.dps = 30
MU_LIM, ZC = 9e-5, 0.67

def W_mu(M): return np.exp(-np.sqrt(1.5e5/M)) - np.exp(-4.5e9/M)
def sigma_max(M): return np.sqrt(MU_LIM/(2.2*W_mu(M)))

# --- threshold shift Delta(M_H) of the linear compaction (= zeta amplitude), from the physical curve (multigroup + T_dec = 1 MeV, make_phys_curve.py)
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + '/'
TB = np.load(P + 'thermo_tables.npz'); o = np.argsort(TB['T']); Tt = TB['T'][o]; rt = (TB['rpl'] + TB['rnu'])[o]
G, c, Msun, MeV4 = 6.674e-11, 2.998e8, 1.989e30, 2.3201e8
MHof = lambda TH: c**3/(2*G*np.sqrt(8*np.pi*G*np.interp(TH, Tt, rt)*MeV4/3))/Msun
def load(pat):
    out = {}
    for f in glob.glob(P + pat):
        d = json.load(open(f)); out[round(d['T0'], 3)] = (d['TH'], d['dc'])
    return out
pf, nu = load('res_eos_pf_*.json'), load('res_phys_*.json')     # physical curve (make_phys_curve.py)
Cl = lambda d: (1 - np.sqrt(1 - 1.5*d))/0.75
keys = sorted(k for k in nu if k in pf)
lnMH = np.array([np.log(MHof(nu[k][0])) for k in keys]); shift = np.array([Cl(nu[k][1])/Cl(pf[k][1]) - 1 for k in keys])
o = np.argsort(lnMH); lnMH, shift = lnMH[o], shift[o]
def Delta(M_pbh, gamma=0.2): return float(np.interp(np.log(M_pbh/gamma), lnMH, shift))

# --- standardized collapse functions S(u) for the four families
def S_GN(u, p):                               # Family A, symmetric generalized normal
    lam = mp.sqrt(mp.gamma(1/mp.mpf(p))/mp.gamma(3/mp.mpf(p)))
    return 0.5*mp.gammainc(1/mp.mpf(p), (mp.mpf(u)/lam)**p, mp.inf, regularized=True)

def _std(core_tail):                           # mean and std of a raw PDF given as (pdf on grid pieces)
    pass

def raw_moments(pdf, xs, tail_moment):
    Z = m1 = m2 = 0.0
    a = -np.inf
    for b in list(xs) + [np.inf]:
        if b == np.inf:
            z, mm1, mm2 = tail_moment(a)
        else:
            z = quad(pdf, a, b, limit=200)[0]; mm1 = quad(lambda x: x*pdf(x), a, b, limit=200)[0]
            mm2 = quad(lambda x: x*x*pdf(x), a, b, limit=200)[0]
        Z += z; m1 += mm1; m2 += mm2; a = b
    mu = m1/Z; return Z, mu, np.sqrt(m2/Z - mu**2)

def S_ST(u, p, xx):                            # Family B, Gaussian core + stretched-exponential tail
    a, b = xx**(2-p)/p, xx**2*(0.5 - 1/p)
    pdf = lambda x: np.exp(-x*x/2) if x <= xx else np.exp(-(a*x**p + b))
    def tail(x0):                              # integrals of 1, x, x^2 over (x0, inf) of exp(-(a x^p + b)), x0 = xx
        out = []
        for n in (0, 1, 2):
            k = (n + 1)/p
            out.append(float(mp.exp(-b)/p*mp.mpf(a)**(-k)*mp.gammainc(k, a*mp.mpf(x0)**p, mp.inf)))
        return out
    Z, mu, sd = raw_moments(pdf, [xx], tail)
    xc = mu + sd*u
    k = 1/p
    return mp.exp(-b)/p*mp.mpf(a)**(-k)*mp.gammainc(k, a*mp.mpf(xc)**p, mp.inf)/Z

def S_PL(u, q, xx):                            # Family C, Gaussian core + power-law tail
    x0 = xx - (q + 1)/xx; A = np.exp(-xx**2/2)*(xx - x0)**(q + 1)
    pdf = lambda x: np.exp(-x*x/2) if x <= xx else A*(x - x0)**(-(q + 1))
    def tail(xa):                              # closed forms over (xa, inf) with y = x - x0
        ya = xa - x0
        I0 = A*ya**(-q)/q; I1 = A*(ya**(1-q)/(q-1) + x0*ya**(-q)/q)
        I2 = A*(ya**(2-q)/(q-2) + 2*x0*ya**(1-q)/(q-1) + x0**2*ya**(-q)/q)
        return I0, I1, I2
    Z, mu, sd = raw_moments(pdf, [xx], tail)
    xc = mu + sd*u
    return A*(xc - x0)**(-q)/q/Z

def S_LN(u, s):                                # Family D, log-normal (standardized; one parameter s)
    Xc = (mp.log(1 + u*mp.sqrt(mp.e**(s*s) - 1)) + s*s/2)/s
    return 0.5*mp.erfc(Xc/mp.sqrt(2))

def beta_max(M, S, *args, shifted=False):
    zc = ZC*(1 + Delta(M)) if shifted else ZC
    return float(2*S(zc/sigma_max(M), *args))

if __name__ == '__main__':
    print('Delta(M) [C_l shift] at M_PBH = 1e5, 1e6, 1e7:', [round(Delta(M), 4) for M in (1e5, 1e6, 1e7)])
    print('u_c (pf) at 1e5, 1e6, 1e7:', [round(ZC/sigma_max(M), 1) for M in (1e5, 1e6, 1e7)])
    cases = [('GN p=0.6', S_GN, (0.6,)), ('GN p=1', S_GN, (1.0,)),
             ('ST p=0.6 x=1', S_ST, (0.6, 1.0)), ('ST p=0.6 x=0.4', S_ST, (0.6, 0.4)), ('ST p=1 x=1', S_ST, (1.0, 1.0)),
             ('PL q=4 x=4.5', S_PL, (4.0, 4.5)), ('PL q=7.5 x=4.5', S_PL, (7.5, 4.5)), ('PL q=4 x=7.5', S_PL, (4.0, 7.5)),
             ('LN s=0.5', S_LN, (0.5,)), ('LN s=0.7', S_LN, (0.7,)), ('LN s=1', S_LN, (1.0,))]
    for name, S, args in cases:
        row = []
        for M in (1e5, 1e6, 1e7):
            b0, b1 = beta_max(M, S, *args), beta_max(M, S, *args, shifted=True)
            row.append(f'{b0:.2e}->{b1:.2e} (x{b1/b0:.2f})')
        print(f'{name:16s}', ' | '.join(row))
