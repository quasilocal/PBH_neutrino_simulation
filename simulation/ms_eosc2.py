"""Two-flavour version of ms_eosc.py: electron neutrinos and mu/tau neutrinos are evolved as separate
energy-integrated intensities I[s, i, k] (s = 0: nu_e, s = 1: nu_mu,tau), each coupled to the plasma by its own
relaxation rate kap_s = kb_s (T/T_b)^5 with kb_s = H_b (T_b/T_dec,s)^3. Everything else as in ms_eosc.py."""
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
def state(t, R, U, Gam, e, I, mue, le, lp, lg, lT, kb, Tb):
    """I has shape (S, N, K); kb has shape (S,)."""
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
    Ac = np.empty(N); Ae = np.empty(N+1); p = np.empty(N); kap = np.zeros((S_, N)); kq = np.zeros(N)
    for i in range(N):
        x = np.log(e[i])
        p[i] = np.exp(np.interp(x, le, lp))
        Tr5 = (np.exp(np.interp(x, le, lT))/Tb)**5
        for s in range(S_):
            if kb[s] > 0.0:
                kap[s, i] = kb[s]*Tr5
            kq[i] += kap[s, i]*q_s[s, i]                  # momentum given to the plasma by all flavours
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
    return eps_s, q, Pr, M, p, Ac, Ae, ae, kap

@njit(cache=True)
def rhs(t, R, U, Gam, e, I, mue, le, lp, lg, lT, kb, Tb, Ib_out, dR, dUt, dG, de, dI):
    S_, N, K = I.shape
    dmu = mue[1]-mue[0]
    eps_s, q, Pr, M, p, Ac, Ae, ae, kap = state(t, R, U, Gam, e, I, mue, le, lp, lg, lT, kb, Tb)
    dR[0] = 0.0; dUt[0] = 0.0; dG[0] = 0.0
    for i in range(1, N+1):
        if i < N:
            pe = 0.5*(p[i-1]+p[i]); Pre = 0.5*(Pr[i-1]+Pr[i]); qe = 0.5*(q[i-1]+q[i])
        else:
            pe = p[N-1]; Pre = Pr[N-1]; qe = q[N-1]
        dR[i] = Ae[i]*U[i]
        dUt[i] = Ae[i]*(-M[i]/R[i]**2 - 4.0*PI*R[i]*(pe+Pre) + Gam[i]*ae[i])
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
    for s in range(S_):
        Fr = np.zeros((N+1, K))
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
            Fm = np.zeros(K+1)
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
            for k in range(K):
                mu = 0.5*(mue[k]+mue[k+1]); mu2 = (mue[k+1]**3-mue[k]**3)/(3*dmu)
                dI[s, i, k] = (-Ac[i]*th[i]*I[s, i, k] - (Fr[i+1, k]-Fr[i, k])/V[i] - (Fm[k+1]-Fm[k])/(dmu*V[i])
                               - Ac[i]*(UR[i] + mu*ac[i] + mu2*S[i])*I[s, i, k]
                               + Ac[i]*kap[s, i]*(eps_s[s, i]/(4*PI) - I[s, i, k]))
    return M, Ac, ac, S, V, kap

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
