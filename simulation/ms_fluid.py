"""
Spherically symmetric PBH formation in a radiation FRW universe: Misner-Sharp equations in the
comoving (Lagrangian) gauge of a perfect fluid p = e/3, with hooks for a second, free-streaming
component (neutrino moments eps, q, Pr per cell; zero in the fluid-only runs).
Units G = c = 1, comoving r with r_m = 1, a(t0) = 1, eps0 = 1/(a H r_m) at t0.
"""
import numpy as np
from numba import njit

PI = np.pi

@njit(cache=True)
def rho_b(t):
    return 3.0/(32.0*PI*t*t)

@njit(cache=True)
def derived(t, R, U, e, eps, q, Pr, fnu, cQ):
    """Return M (edges), Gam (edges), p (cells, incl. AV), Ac (cells), Ae (edges)."""
    N = e.size
    M = np.zeros(N+1); Gam = np.ones(N+1)
    Ac = np.empty(N); Ae = np.empty(N+1); p = np.empty(N)
    eb = (1.0-fnu)*rho_b(t)
    for i in range(N):
        Ac[i] = (eb/e[i])**0.25
        dU = U[i+1]-U[i]
        Q = 0.0
        if dU < 0.0:
            Q = cQ*cQ*(e[i]+e[i]/3.0)*dU*dU
        p[i] = e[i]/3.0 + Q
    for i in range(N):
        dV = (4.0*PI/3.0)*(R[i+1]**3 - R[i]**3)
        # flux correction qU/Gamma uses the inner-edge Gamma (q = 0 in fluid-only runs)
        Ubar = 0.5*(U[i]+U[i+1])
        M[i+1] = M[i] + dV*(e[i] + eps[i] + q[i]*Ubar/Gam[i])
        g2 = 1.0 + U[i+1]**2 - 2.0*M[i+1]/R[i+1]
        Gam[i+1] = np.sqrt(g2) if g2 > 1e-12 else 1e-6
    Ae[0] = Ac[0]; Ae[N] = Ac[N-1]
    for i in range(1, N):
        Ae[i] = np.sqrt(Ac[i-1]*Ac[i])
    return M, Gam, p, Ac, Ae

@njit(cache=True)
def rhs(t, R, U, e, eps, q, Pr, fnu, cQ, dR, dUt, de):
    N = e.size
    M, Gam, p, Ac, Ae = derived(t, R, U, e, eps, q, Pr, fnu, cQ)
    dR[0] = 0.0; dUt[0] = 0.0
    for i in range(1, N+1):
        if i < N:
            Rc0 = 0.5*(R[i-1]+R[i]); Rc1 = 0.5*(R[i]+R[i+1])
            dpdR = (p[i]-p[i-1])/(Rc1-Rc0)
            hp = 0.5*(e[i-1]+p[i-1]+e[i]+p[i])
            pe = 0.5*(p[i-1]+p[i]); Pre = 0.5*(Pr[i-1]+Pr[i])
        else:
            dpdR = 0.0                      # D_r p = 0 at the outer edge (as in Escriva 2020)
            hp = e[N-1]+p[N-1]; pe = p[N-1]; Pre = Pr[N-1]
        dR[i] = Ae[i]*U[i]
        dUt[i] = Ae[i]*(-M[i]/R[i]**2 - 4.0*PI*R[i]*(pe+Pre) - Gam[i]**2*dpdR/hp)
    for i in range(N):
        theta = 3.0*(R[i+1]**2*U[i+1] - R[i]**2*U[i])/(R[i+1]**3 - R[i]**3)
        Rc = 0.5*(R[i]+R[i+1]); Gc = 0.5*(Gam[i]+Gam[i+1])
        theta -= 4.0*PI*Rc*q[i]/Gc
        de[i] = -Ac[i]*(e[i]+p[i])*theta
    return M, Gam, Ac

def initial_data(r, delta_m, eps0, fnu=0.0, qshape=1.0):
    """Gradient-expansion data for K(r) = A exp(-(1/q) r^{2q}) (r_m = 1). Returns t0, R, U, rho_cells."""
    H0 = 1.0/eps0; t0 = 1.0/(2.0*H0)
    A = 1.5*np.exp(1.0/qshape)*delta_m
    K = A*np.exp(-(r**(2*qshape))/qshape)
    rK = -2.0*r**(2*qshape)*K                       # r K'(r)
    eps2 = eps0**2
    dR = -(2*K + rK/3.0)/12.0
    dU = -K/6.0
    dM = (2.0/3.0)*K
    R = r*(1.0 + eps2*dR)
    U = H0*R*(1.0 + eps2*dU)
    rb = rho_b(t0)
    Mi = (4*np.pi/3)*rb*R**3*(1.0 + eps2*dM)
    rho = np.diff(Mi)/((4*np.pi/3)*np.diff(R**3))   # cell averages consistent with M
    return t0, R, U, rho

def make_grid(N, rmax, beta=3.0):
    x = np.linspace(0.0, 1.0, N+1)
    return rmax*np.sinh(beta*x)/np.sinh(beta)
