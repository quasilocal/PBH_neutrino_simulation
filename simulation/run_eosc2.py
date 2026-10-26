"""Two-flavour driver (cf. run_eosc.py): nu_e (1/3 of the neutrino energy) decouples at Tdec_e, nu_mu,tau (2/3) at Tdec_mt."""
import numpy as np, time
from ms_eosc2 import rhs, timestep
from ms_fluid import initial_data, make_grid
from run_eos import make_eos, TB
FR = np.array([1/3, 2/3])

def run(delta_m, T0, Tdec_e=1.5, Tdec_mt=2.5, N=300, K=32, rmax=9.0, beta=3.0, eps0=0.1, cfl_r=0.4, cfl_mu=0.4, cH=0.005,
        eps_max=15.0, qshape=1.0, hist_every=20, disperse_rule=True, switch_ratio=100.0):
    Td = np.array([Tdec_e, Tdec_mt]); Tdmax = Td.max()
    tight = (T0/Tdmax)**3 > switch_ratio          # one perfect fluid until the most weakly coupled flavour has kap/H = switch_ratio
    tab, frac, Tof = make_eos('pf' if tight else 'fs', T0, eps0)
    le, lp, lg, lnu = tab['le'], tab['lp'], tab['lg'], tab['lnu']
    lT = np.log(TB['T'][::-1])
    if tight:
        tab_fs, frac_fs, Tof_fs = make_eos('fs', T0, eps0)
        fn_tab = (TB['rnu']/(TB['rnu']+TB['rpl']))[::-1]
    r = make_grid(N, rmax, beta)
    t0, R, U, rho = initial_data(r, delta_m, eps0, 0.0, qshape)
    e, rnu = frac(rho)
    mue = np.linspace(-1, 1, K+1)
    I = np.stack([np.repeat((f*rnu/(4*np.pi))[:, None], K, axis=1) for f in FR])
    Mi = np.concatenate([[0.0], np.cumsum((4*np.pi/3)*np.diff(R**3)*rho)])
    Gam = np.ones(N+1); Gam[1:] = np.sqrt(1.0 + U[1:]**2 - 2*Mi[1:]/R[1:])
    t = t0
    dR = np.zeros(N+1); dU = np.zeros(N+1); dG = np.zeros(N+1); de = np.zeros(N); dI = np.zeros((2, N, K))
    Ib = lambda ee: FR*np.exp(np.interp(np.log(ee[-1]), le, lnu))/(4*np.pi)
    peakC = -1.0; step = 0; fallback = False; status = 'undecided'; TH = None
    while True:
        Hb = U[-1]/R[-1]; epsn = r[-1]/U[-1]; rhob = 3*Hb*Hb/(8*np.pi)
        Tb = float(np.exp(np.interp(np.log(e[-1]), le, lT)))
        if tight and (Tb/Tdmax)**3 <= switch_ratio:
            Tl = np.exp(np.interp(np.log(e), le, lT)); fn = np.interp(Tl, np.exp(lT), fn_tab)
            I = np.stack([np.repeat((f*fn*e/(4*np.pi))[:, None], K, axis=1) for f in FR]); e = (1-fn)*e
            tab, Tof = tab_fs, Tof_fs; le, lp, lg, lnu = tab['le'], tab['lp'], tab['lg'], tab['lnu']
            tight = False
            Tb = float(np.exp(np.interp(np.log(e[-1]), le, lT)))
        kb = Hb*(Tb/Td)**3 if not tight else np.zeros(2)
        try:
            M, Ac, ac, S, V, kap = rhs(t, R, U, Gam, e, I, mue, le, lp, lg, lT, kb, Tb, Ib(e), dR, dU, dG, de, dI)
        except (ZeroDivisionError, FloatingPointError):
            status = 'crash'; break
        if TH is None and epsn >= 1.0:
            TH = Tb
        twoMR = 2*M[1:]/R[1:]
        C = 2*(M[1:] - (4*np.pi/3)*rhob*R[1:]**3)/R[1:]
        inner = r[1:] < 4.0
        iC = np.argmax(C[inner]); peakC = max(peakC, C[iC])
        if np.any((twoMR >= 1.0) & (U[1:] < 0)):
            status = 'BH'; break
        if disperse_rule and epsn > 3.0 and C[iC] < 0.5*peakC:
            status = 'dispersed'; break
        if epsn > eps_max: break
        try:
            dt = min(timestep(R, U, e, I, mue, Gam, Ac, ac, S, cfl_r, cfl_mu), cH/Hb)
            km = np.max(Ac[None, :]*kap)
            if km > 0: dt = min(dt, 0.5/km)
            R1 = R + dt*dR; U1 = U + dt*dU; G1 = Gam + dt*dG; e1 = e + dt*de; I1 = I + dt*dI
            rhs(t+dt, R1, U1, G1, e1, I1, mue, le, lp, lg, lT, kb, Tb, Ib(e1), dR, dU, dG, de, dI)
            R2 = 0.75*R + 0.25*(R1+dt*dR); U2 = 0.75*U + 0.25*(U1+dt*dU); G2 = 0.75*Gam + 0.25*(G1+dt*dG)
            e2 = 0.75*e + 0.25*(e1+dt*de); I2 = 0.75*I + 0.25*(I1+dt*dI)
            rhs(t+0.5*dt, R2, U2, G2, e2, I2, mue, le, lp, lg, lT, kb, Tb, Ib(e2), dR, dU, dG, de, dI)
            R = R/3 + 2/3*(R2+dt*dR); U = U/3 + 2/3*(U2+dt*dU); Gam = Gam/3 + 2/3*(G2+dt*dG)
            e = e/3 + 2/3*(e2+dt*de); I = I/3 + 2/3*(I2+dt*dI)
        except (ZeroDivisionError, FloatingPointError):
            coll = U[1:] < 0
            status = 'BH' if (coll.any() and np.max(twoMR[coll]) > 0.95) else 'crash'
            fallback = (status == 'BH')
            break
        t += dt; step += 1
        if np.any(e <= 0) or not np.all(np.isfinite(U)) or not np.all(np.isfinite(I)):
            status = 'crash'; break
    return dict(status=status, fallback=fallback, eps=epsn, steps=step, peakC=peakC, TH=TH)
