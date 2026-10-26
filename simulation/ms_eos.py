"""
Misner-Sharp perfect fluid (p = e/3) + massless free-streaming neutrinos (energy-integrated intensity
I(t, r, mu) in the fluid frame, finite volume in (r, mu), MUSCL/MC slopes, upwind).  Fluid-comoving gauge.
Transport (non-conservative form for I, using dV/dt = A theta V with V = int R^2 B dr):
  dI/dt = -A th I - [F_r(i+1)-F_r(i)]/V - [F_mu(k+1/2)-F_mu(k-1/2)]/(dmu V) - A (U/R + mu a + mu^2 S) I
  F_r = A R^2 mu I (edges),  F_mu = A (1-mu^2) [G1 - a V - mu S V] I,  G1 = (R_{i+1}^2 - R_i^2)/2
"""
import numpy as np
from numba import njit

PI = np.pi

@njit(cache=True)
def rho_b(t):
    return 3.0/(32.0*PI*t*t)

@njit(cache=True)
def mc(dm, dp):
    if dm*dp <= 0.0:
        return 0.0
    s = 1.0 if dm > 0 else -1.0
    return s*min(2*abs(dm), 2*abs(dp), 0.5*abs(dm+dp))

@njit(cache=True)
def state(t, R, U, Gam, e, I, mue, le, lp, lg):
    """Derived quantities. I has shape (N, K)."""
    N, K = I.shape
    dmu = mue[1]-mue[0]
    eps = np.zeros(N); q = np.zeros(N); Pr = np.zeros(N)
    for i in range(N):
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for k in range(K):
            m0 = mue[k]; m1 = mue[k+1]
            s0 += I[i, k]
            s1 += I[i, k]*0.5*(m0+m1)
            s2 += I[i, k]*(m1**3-m0**3)/(3*dmu)
        eps[i] = 2*PI*s0*dmu; q[i] = 2*PI*s1*dmu; Pr[i] = 2*PI*s2*dmu
    M = np.zeros(N+1)
    for i in range(N):
        dV = (4.0*PI/3.0)*(R[i+1]**3 - R[i]**3)
        Ubar = 0.5*(U[i]+U[i+1]); Gbar = 0.5*(Gam[i]+Gam[i+1])
        M[i+1] = M[i] + dV*(e[i] + eps[i] + q[i]*Ubar/Gbar)
    Ac = np.empty(N); Ae = np.empty(N+1); p = np.empty(N)
    gref = np.interp(np.log(e[N-1]), le, lg)
    for i in range(N):
        x = np.log(e[i])
        p[i] = np.exp(np.interp(x, le, lp))
        Ac[i] = np.exp(gref - np.interp(x, le, lg))
    Ae[0] = Ac[0]; Ae[N] = Ac[N-1]
    for i in range(1, N):
        Ae[i] = np.sqrt(Ac[i-1]*Ac[i])
    # fluid acceleration a = -Gam dp/dR/(e+p) at edges (0 at centre and outer edge), then cell centres
    ae = np.zeros(N+1)
    for i in range(1, N):
        Rc0 = 0.5*(R[i-1]+R[i]); Rc1 = 0.5*(R[i]+R[i+1])
        dpdR = (p[i]-p[i-1])/(Rc1-Rc0)
        ae[i] = -Gam[i]*dpdR/(0.5*(e[i-1]+p[i-1]+e[i]+p[i]))
    return eps, q, Pr, M, p, Ac, Ae, ae

@njit(cache=True)
def rhs(t, R, U, Gam, e, I, mue, le, lp, lg, Ib_out, dR, dUt, dG, de, dI):
    N, K = I.shape
    dmu = mue[1]-mue[0]
    eps, q, Pr, M, p, Ac, Ae, ae = state(t, R, U, Gam, e, I, mue, le, lp, lg)
    # ---- fluid + metric ----
    dR[0] = 0.0; dUt[0] = 0.0; dG[0] = 0.0
    for i in range(1, N+1):
        if i < N:
            pe = 0.5*(p[i-1]+p[i]); Pre = 0.5*(Pr[i-1]+Pr[i]); qe = 0.5*(q[i-1]+q[i])
        else:
            pe = p[N-1]; Pre = Pr[N-1]; qe = q[N-1]
        dR[i] = Ae[i]*U[i]
        dUt[i] = Ae[i]*(-M[i]/R[i]**2 - 4.0*PI*R[i]*(pe+Pre) + Gam[i]*ae[i])
        dG[i] = Ae[i]*(4.0*PI*R[i]*qe + ae[i]*U[i])      # D_t Gamma = 4 pi R q + a U (evolved: avoids U^2 - 2M/R cancellation)
    th = np.empty(N); Gc = np.empty(N); V = np.empty(N); S = np.empty(N); ac = np.empty(N); UR = np.empty(N)
    for i in range(N):
        dR3 = R[i+1]**3 - R[i]**3
        Rc = 0.5*(R[i]+R[i+1]); Gc[i] = 0.5*(Gam[i]+Gam[i+1])
        th[i] = 3.0*(R[i+1]**2*U[i+1] - R[i]**2*U[i])/dR3 - 4.0*PI*Rc*q[i]/Gc[i]
        de[i] = -Ac[i]*(e[i]+p[i])*th[i]
        V[i] = dR3/(3.0*Gc[i])
        UR[i] = (U[i+1]+U[i])/(R[i+1]+R[i])
        S[i] = (U[i+1]-U[i])/(R[i+1]-R[i]) - 4.0*PI*Rc*q[i]/Gc[i] - UR[i]
        ac[i] = 0.5*(ae[i]+ae[i+1])
    # ---- neutrinos ----
    # radial fluxes at edges
    Fr = np.zeros((N+1, K))
    for k in range(K):
        mu = 0.5*(mue[k]+mue[k+1])
        for i in range(1, N+1):
            if mu > 0:
                j = i-1
                if j >= 1 and j <= N-2:
                    sl = mc(I[j, k]-I[j-1, k], I[j+1, k]-I[j, k])
                else:
                    sl = 0.0
                Iface = I[j, k] + 0.5*sl
            else:
                if i == N:
                    Iface = Ib_out
                else:
                    j = i
                    if j >= 1 and j <= N-2:
                        sl = mc(I[j, k]-I[j-1, k], I[j+1, k]-I[j, k])
                    else:
                        sl = 0.0
                    Iface = I[j, k] - 0.5*sl
            Fr[i, k] = Ae[i]*R[i]**2*mu*Iface
    for i in range(N):
        G1 = 0.5*(R[i+1]**2 - R[i]**2)
        # angular fluxes at interior bin edges
        Fm = np.zeros(K+1)
        for k in range(1, K):
            m = mue[k]
            c = Ac[i]*(1.0-m*m)*(G1 - ac[i]*V[i] - m*S[i]*V[i])
            if c > 0:
                sl = mc(I[i, k-1]-I[i, k-2], I[i, k]-I[i, k-1]) if k-1 >= 1 else 0.0
                Iface = I[i, k-1] + 0.5*sl
            else:
                sl = mc(I[i, k]-I[i, k-1], I[i, k+1]-I[i, k]) if k+1 <= K-1 else 0.0
                Iface = I[i, k] - 0.5*sl
            Fm[k] = c*Iface
        for k in range(K):
            mu = 0.5*(mue[k]+mue[k+1]); mu2 = (mue[k+1]**3-mue[k]**3)/(3*dmu)
            dI[i, k] = (-Ac[i]*th[i]*I[i, k] - (Fr[i+1, k]-Fr[i, k])/V[i] - (Fm[k+1]-Fm[k])/(dmu*V[i])
                        - Ac[i]*(UR[i] + mu*ac[i] + mu2*S[i])*I[i, k])
    return M, Ac, ac, S, V

@njit(cache=True)
def timestep(R, U, e, I, mue, Gam, Ac, ac, S, cfl_r, cfl_mu):
    N, K = I.shape
    dmu = mue[1]-mue[0]
    dt = 1e30
    for i in range(N):
        dRc = R[i+1]-R[i]; Gc = 0.5*(Gam[i]+Gam[i+1])
        dt = min(dt, cfl_r*dRc/(abs(Gc)*Ac[i]))
        dR3 = R[i+1]**3 - R[i]**3
        V = dR3/(3.0*Gc); G1 = 0.5*(R[i+1]**2 - R[i]**2)
        cmax = Ac[i]*(abs(G1/V) + abs(ac[i]) + abs(S[i]))
        dt = min(dt, cfl_mu*dmu/cmax)
    return dt
