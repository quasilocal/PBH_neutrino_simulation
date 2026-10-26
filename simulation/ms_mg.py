"""Multigroup version of ms_eosc2.py.

The neutrinos are split into NF flavours x G energy groups, I[s, i, k] with s = f*G + g. Group g covers a fixed range of
comoving energy x = E a(t)/a_0 (ell = ln x). Each group obeys the energy-integrated transport equation of ms_eosc2.py,
plus the flux between groups caused by the local redshift relative to the background,
    d ell/dt = H_b - w,   w = A (U/R + mu a + mu^2 S),
which sums to zero over the groups, so that the sum over groups is exactly the energy-integrated equation.
Collisions, per group:
  nu - e+-:  A kap_e (eps_s/4pi - I)            relaxation to isotropy in the plasma frame (momentum to the plasma)
  nu - nu :  A kap_n (eps_s/4pi + c_s mu - I)   relaxation to isotropy in the neutrino frame to first order
                                                (conserves the energy and flux of each group; no momentum to the plasma)
kap_e = kbe[s] (T/T_b)^pe   (pe = 4 for energy-dependent rates, 5 for the gray rate of ms_eosc2)
kap_n = kbn[s] (eps_nu/eps_nu,b)   (rate proportional to T_nu^4)."""
import numpy as np
from numba import njit

PI = np.pi

@njit(cache=True)
def mc(dm, dp):
    if dm*dp <= 0.0:
        return 0.0
    s = 1.0 if dm > 0 else -1.0
    return s*min(2*abs(dm), 2*abs(dp), 0.5*abs(dm+dp))

@njit(cache=True)
def state(t, R, U, Gam, e, I, mue, le, lp, lg, lT, kbe, kbn, Tb, enub, pe, lchi, chib):
    S_, N, K = I.shape
    dmu = mue[1]-mue[0]
    eps_s = np.zeros((S_, N)); q_s = np.zeros((S_, N)); Pr_s = np.zeros((S_, N))
    for s in range(S_):
        for i in range(N):
            s0 = 0.0; s1 = 0.0; s2 = 0.0
            for k in range(K):
                m0 = mue[k]; m1 = mue[k+1]
                s0 += I[s, i, k]
                s1 += I[s, i, k]*0.5*(m0+m1)
                s2 += I[s, i, k]*(m1**3-m0**3)/(3*dmu)
            eps_s[s, i] = 2*PI*s0*dmu; q_s[s, i] = 2*PI*s1*dmu; Pr_s[s, i] = 2*PI*s2*dmu
    eps = np.zeros(N); q = np.zeros(N); Pr = np.zeros(N)
    for s in range(S_):
        for i in range(N):
            eps[i] += eps_s[s, i]; q[i] += q_s[s, i]; Pr[i] += Pr_s[s, i]
    M = np.zeros(N+1)
    for i in range(N):
        dV = (4.0*PI/3.0)*(R[i+1]**3 - R[i]**3)
        Ubar = 0.5*(U[i]+U[i+1]); Gbar = 0.5*(Gam[i]+Gam[i+1])
        M[i+1] = M[i] + dV*(e[i] + eps[i] + q[i]*Ubar/Gbar)
    Ac = np.empty(N); Ae = np.empty(N+1); p = np.empty(N)
    kap_e = np.zeros((S_, N)); kap_n = np.zeros((S_, N)); kq = np.zeros(N)
    for i in range(N):
        x = np.log(e[i])
        p[i] = np.exp(np.interp(x, le, lp))
        fe = (np.exp(np.interp(x, le, lT))/Tb)**pe
        if chib > 0.0:                                  # optional: e+- mass correction at the local temperature
            fe *= np.exp(np.interp(x, le, lchi))/chib
        fn = eps[i]/enub if enub > 0.0 else 0.0
        for s in range(S_):
            kap_e[s, i] = kbe[s]*fe
            kap_n[s, i] = kbn[s]*fn
            kq[i] += kap_e[s, i]*q_s[s, i]              # momentum given to the plasma (nu - e+- only)
    ae = np.zeros(N+1)
    for i in range(1, N):
        Rc0 = 0.5*(R[i-1]+R[i]); Rc1 = 0.5*(R[i]+R[i+1])
        dpdR = (p[i]-p[i-1])/(Rc1-Rc0)
        ae[i] = (-Gam[i]*dpdR + 0.5*(kq[i-1]+kq[i]))/(0.5*(e[i-1]+p[i-1]+e[i]+p[i]))
    ae[N] = kq[N-1]/(e[N-1]+p[N-1])
    Ac[N-1] = 1.0
    for i in range(N-2, -1, -1):
        Rc0 = 0.5*(R[i]+R[i+1]); Rc1 = 0.5*(R[i+1]+R[i+2])
        Ac[i] = Ac[i+1]*np.exp(-ae[i+1]*(Rc1-Rc0)/Gam[i+1])
    Ae[0] = Ac[0]; Ae[N] = Ac[N-1]
    for i in range(1, N):
        Ae[i] = np.sqrt(Ac[i-1]*Ac[i])
    return eps_s, q_s, q, Pr, M, p, Ac, Ae, ae, kap_e, kap_n

@njit(cache=True)
def rhs(t, R, U, Gam, e, I, mue, le, lp, lg, lT, kbe, kbn, Tb, enub, pe, G, dl, Hb, Ib_out, dR, dUt, dG, de, dI,
        lchi, chib, nnmode):
    S_, N, K = I.shape
    NF = S_//G
    dmu = mue[1]-mue[0]
    eps_s, q_s, q, Pr, M, p, Ac, Ae, ae, kap_e, kap_n = state(t, R, U, Gam, e, I, mue, le, lp, lg, lT, kbe, kbn,
                                                              Tb, enub, pe, lchi, chib)
    sum_mu2 = 0.0
    for k in range(K):
        mu = 0.5*(mue[k]+mue[k+1]); sum_mu2 += mu*mu*dmu
    dR[0] = 0.0; dUt[0] = 0.0; dG[0] = 0.0
    for i in range(1, N+1):
        if i < N:
            pe_ = 0.5*(p[i-1]+p[i]); Pre = 0.5*(Pr[i-1]+Pr[i]); qe = 0.5*(q[i-1]+q[i])
        else:
            pe_ = p[N-1]; Pre = Pr[N-1]; qe = q[N-1]
        dR[i] = Ae[i]*U[i]
        dUt[i] = Ae[i]*(-M[i]/R[i]**2 - 4.0*PI*R[i]*(pe_+Pre) + Gam[i]*ae[i])
        dG[i] = Ae[i]*(4.0*PI*R[i]*qe + ae[i]*U[i])
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
    # optional nu-nu target (nnmode = 1): isotropic distribution in a common neutrino rest frame, I ~ (1 - v mu)^-4.
    # Each group keeps its energy; v is fixed by the rate-weighted flux, sum_s kap_n q_s / sum_s kap_n eps_s, so that
    # the term conserves the total neutrino momentum exactly (the rates differ between groups).
    gn = np.zeros((N, K))
    if nnmode == 1:
        for i in range(N):
            et = 0.0; qt = 0.0
            for s in range(S_):
                et += kap_n[s, i]*eps_s[s, i]; qt += kap_n[s, i]*q_s[s, i]
            f = qt/et if et > 0.0 else 0.0
            vlo = -0.999; vhi = 0.999
            for it in range(60):
                vm = 0.5*(vlo + vhi); sg = 0.0; sm = 0.0
                for k in range(K):
                    mu = 0.5*(mue[k]+mue[k+1]); g = (1.0 - vm*mu)**(-4)
                    sg += g; sm += g*mu
                if sm/sg < f:
                    vlo = vm
                else:
                    vhi = vm
            vm = 0.5*(vlo + vhi); sg = 0.0
            for k in range(K):
                mu = 0.5*(mue[k]+mue[k+1]); gn[i, k] = (1.0 - vm*mu)**(-4); sg += gn[i, k]
            for k in range(K):
                gn[i, k] = gn[i, k]/(2*PI*dmu*sg)
    Fr = np.zeros((N+1, K)); Fm = np.zeros(K+1)
    for s in range(S_):
        for k in range(K):
            mu = 0.5*(mue[k]+mue[k+1])
            for i in range(1, N+1):
                if mu > 0:
                    j = i-1
                    if j >= 1 and j <= N-2:
                        sl = mc(I[s, j, k]-I[s, j-1, k], I[s, j+1, k]-I[s, j, k])
                    else:
                        sl = 0.0
                    Iface = I[s, j, k] + 0.5*sl
                else:
                    if i == N:
                        Iface = Ib_out[s]
                    else:
                        j = i
                        if j >= 1 and j <= N-2:
                            sl = mc(I[s, j, k]-I[s, j-1, k], I[s, j+1, k]-I[s, j, k])
                        else:
                            sl = 0.0
                        Iface = I[s, j, k] - 0.5*sl
                Fr[i, k] = Ae[i]*R[i]**2*mu*Iface
        for i in range(N):
            G1 = 0.5*(R[i+1]**2 - R[i]**2)
            Fm[0] = 0.0; Fm[K] = 0.0
            for k in range(1, K):
                m = mue[k]
                c = Ac[i]*(1.0-m*m)*(G1 - ac[i]*V[i] - m*S[i]*V[i])
                if c > 0:
                    sl = mc(I[s, i, k-1]-I[s, i, k-2], I[s, i, k]-I[s, i, k-1]) if k-1 >= 1 else 0.0
                    Iface = I[s, i, k-1] + 0.5*sl
                else:
                    sl = mc(I[s, i, k]-I[s, i, k-1], I[s, i, k+1]-I[s, i, k]) if k+1 <= K-1 else 0.0
                    Iface = I[s, i, k] - 0.5*sl
                Fm[k] = c*Iface
            cs = q_s[s, i]/(2*PI*sum_mu2)
            for k in range(K):
                mu = 0.5*(mue[k]+mue[k+1]); mu2 = (mue[k+1]**3-mue[k]**3)/(3*dmu)
                tgt = eps_s[s, i]*gn[i, k] if nnmode == 1 else eps_s[s, i]/(4*PI) + cs*mu
                dI[s, i, k] = (-Ac[i]*th[i]*I[s, i, k] - (Fr[i+1, k]-Fr[i, k])/V[i] - (Fm[k+1]-Fm[k])/(dmu*V[i])
                               - Ac[i]*(UR[i] + mu*ac[i] + mu2*S[i])*I[s, i, k]
                               + Ac[i]*kap_e[s, i]*(eps_s[s, i]/(4*PI) - I[s, i, k])
                               + Ac[i]*kap_n[s, i]*(tgt - I[s, i, k]))
    # energy flux between the groups of each flavour (comoving energy ell = ln x), MC-limited upwind.
    # The background expansion rate is evaluated from this (Runge-Kutta stage) state, at the outer edge where A = 1.
    Hb = U[N]/R[N]
    vmax = 0.0
    if G > 1:
        for f in range(NF):
            for i in range(N):
                for k in range(K):
                    mu = 0.5*(mue[k]+mue[k+1]); mu2 = (mue[k+1]**3-mue[k]**3)/(3*dmu)
                    v = Hb - Ac[i]*(UR[i] + mu*ac[i] + mu2*S[i])
                    if abs(v) > vmax:
                        vmax = abs(v)
                    Fprev = 0.0
                    for g in range(G):
                        if g < G-1:
                            sA = f*G + g; sB = sA + 1
                            if v > 0:
                                sl = mc(I[sA, i, k]-I[sA-1, i, k], I[sB, i, k]-I[sA, i, k]) if g >= 1 else 0.0
                                Iface = I[sA, i, k] + 0.5*sl
                            else:
                                sl = mc(I[sB, i, k]-I[sA, i, k], I[sB+1, i, k]-I[sB, i, k]) if g+2 <= G-1 else 0.0
                                Iface = I[sB, i, k] - 0.5*sl
                            Fnext = v*Iface/dl
                        else:
                            Fnext = 0.0
                        dI[f*G+g, i, k] -= (Fnext - Fprev)
                        Fprev = Fnext
    return M, Ac, ac, S, V, kap_e, kap_n, vmax

@njit(cache=True)
def timestep(R, U, e, I, mue, Gam, Ac, ac, S, cfl_r, cfl_mu):
    S_, N, K = I.shape
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
