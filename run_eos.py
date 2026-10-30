import os
import numpy as np, time
from ms_eos import rhs, state, timestep
from ms_fluid import initial_data, make_grid
TB = dict(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thermo_tables.npz')))   # load into memory (safe with forked workers)

def make_eos(kind, T0=None, eps0=0.1, fnu=0.409):
    """Tables in code units: le (log e_fluid), lp (log p), lg (lapse potential g), lnu (log rho_nu background
    vs log e_fluid); and functions giving fluid / neutrino fractions of the total density."""
    rho0 = 3.0/(8*np.pi*eps0**2)                       # total density at t0 in code units (H0 = 1/eps0)
    if kind == 'rad':
        le = np.linspace(-30, 60, 2000)
        tab = dict(le=le, lp=le+np.log(1/3), lg=le/4, lnu=le+np.log(max(fnu, 1e-30)/(1-fnu)))
        frac = lambda rho: ((1-fnu)*rho, fnu*rho)
        return tab, frac, None
    T = TB['T'][::-1]; rpl = TB['rpl'][::-1]; ppl = TB['ppl'][::-1]; rnu = TB['rnu'][::-1]
    if kind == 'pf':
        e_ph = rpl + rnu; p_ph = ppl + rnu/3
    elif kind == 'fs':
        e_ph = rpl; p_ph = ppl
    rtot = rpl + rnu
    kappa = rho0/np.interp(T0, T, rtot)
    le = np.log(kappa*e_ph); lp = np.log(kappa*p_ph)
    # g(e) = int dp/(e+p) (trapezoid in T)
    dp = np.diff(p_ph); hm = 0.5*((e_ph+p_ph)[1:]+(e_ph+p_ph)[:-1])
    lg = np.concatenate([[0.0], np.cumsum(dp/hm)])
    lnu = np.log(kappa*rnu) if kind == 'fs' else le - 60.0
    tab = dict(le=le, lp=lp, lg=lg, lnu=lnu)
    def frac(rho_code):
        Tl = np.interp(np.log(rho_code/kappa), np.log(rtot), T)
        fn = np.interp(Tl, T, rnu/rtot) if kind == 'fs' else 0.0*Tl
        return (1-fn)*rho_code, fn*rho_code
    Tof = lambda e_code: np.interp(np.log(e_code), le, T)  # temperature from fluid density
    return tab, frac, Tof

def run(delta_m, kind='rad', T0=None, fnu=0.409, N=400, K=32, rmax=9.0, beta=3.0, eps0=0.1, cfl_r=0.4, cfl_mu=0.4,
        cH=0.005, eps_max=15.0, qshape=1.0, hist_every=20, disperse_rule=True, eps_dec=None):
    switch = (kind == 'fs' and eps_dec is not None and eps_dec > eps0)
    tab, frac, Tof = make_eos('pf' if switch else kind, T0, eps0, fnu)
    le, lp, lg, lnu = tab['le'], tab['lp'], tab['lg'], tab['lnu']
    if switch:
        tab_fs, frac_fs, Tof_fs = make_eos('fs', T0, eps0, fnu)
        Tt = TB['T'][::-1]; fn_tab = (TB['rnu']/(TB['rnu']+TB['rpl']))[::-1]
    r = make_grid(N, rmax, beta)
    t0, R, U, rho = initial_data(r, delta_m, eps0, 0.0, qshape)
    e, rnu = frac(rho)
    mue = np.linspace(-1, 1, K+1)
    I = np.repeat((rnu/(4*np.pi))[:, None], K, axis=1)
    Mi = np.concatenate([[0.0], np.cumsum((4*np.pi/3)*np.diff(R**3)*rho)])
    Gam = np.ones(N+1); Gam[1:] = np.sqrt(1.0 + U[1:]**2 - 2*Mi[1:]/R[1:])
    t = t0
    dR = np.zeros(N+1); dU = np.zeros(N+1); dG = np.zeros(N+1); de = np.zeros(N); dI = np.zeros((N, K))
    peakC = -1.0; hist = []; step = 0; status = 'undecided'; TH = None
    Ib = lambda ee: np.exp(np.interp(np.log(ee[-1]), le, lnu))/(4*np.pi)   # late binding: follows the table switch
    while True:
        try:
            M, Ac, ac, S, V = rhs(t, R, U, Gam, e, I, mue, le, lp, lg, Ib(e), dR, dU, dG, de, dI)
        except (ZeroDivisionError, FloatingPointError):
            status = 'crash'; break
        Hb = U[-1]/R[-1]; epsn = r[-1]/U[-1]; rhob = 3*Hb*Hb/(8*np.pi)
        if switch and epsn >= eps_dec:          # instantaneous neutrino decoupling: split the fluid
            Tl = Tof(e); fn = np.interp(Tl, Tt, fn_tab)
            I = np.repeat((fn*e/(4*np.pi))[:, None], K, axis=1); e = (1-fn)*e
            tab, Tof = tab_fs, Tof_fs; le, lp, lg, lnu = tab['le'], tab['lp'], tab['lg'], tab['lnu']
            switch = False
            M, Ac, ac, S, V = rhs(t, R, U, Gam, e, I, mue, le, lp, lg, Ib(e), dR, dU, dG, de, dI)
        if TH is None and epsn >= 1.0 and Tof is not None:
            TH = float(Tof(e[-1]))
        twoMR = 2*M[1:]/R[1:]
        C = 2*(M[1:] - (4*np.pi/3)*rhob*R[1:]**3)/R[1:]
        iC = np.argmax(C[r[1:] < 4.0]); peakC = max(peakC, C[iC])
        if step % hist_every == 0:
            hist.append((t, epsn, C[iC], r[1+iC], e[0]/e[-1]))
        if np.any((twoMR >= 1.0) & (U[1:] < 0)):
            status = 'BH'; break
        if disperse_rule and epsn > 3.0 and C[iC] < 0.5*peakC:
            status = 'dispersed'; break
        if epsn > eps_max: break
        try:
            dt = min(timestep(R, U, e, I, mue, Gam, Ac, ac, S, cfl_r, cfl_mu), cH/Hb)
            R1 = R + dt*dR; U1 = U + dt*dU; G1 = Gam + dt*dG; e1 = e + dt*de; I1 = I + dt*dI
            rhs(t+dt, R1, U1, G1, e1, I1, mue, le, lp, lg, Ib(e1), dR, dU, dG, de, dI)
            R2 = 0.75*R + 0.25*(R1+dt*dR); U2 = 0.75*U + 0.25*(U1+dt*dU); G2 = 0.75*Gam + 0.25*(G1+dt*dG)
            e2 = 0.75*e + 0.25*(e1+dt*de); I2 = 0.75*I + 0.25*(I1+dt*dI)
            rhs(t+0.5*dt, R2, U2, G2, e2, I2, mue, le, lp, lg, Ib(e2), dR, dU, dG, de, dI)
            R = R/3 + 2/3*(R2+dt*dR); U = U/3 + 2/3*(U2+dt*dU); Gam = Gam/3 + 2/3*(G2+dt*dG)
            e = e/3 + 2/3*(e2+dt*de); I = I/3 + 2/3*(I2+dt*dI)
        except (ZeroDivisionError, FloatingPointError):
            coll = U[1:] < 0
            status = 'BH' if (coll.any() and np.max(twoMR[coll]) > 0.95) else 'crash'
            break
        t += dt; step += 1
        if np.any(e <= 0) or not np.all(np.isfinite(U)) or not np.all(np.isfinite(I)):
            status = 'crash'; break
    return dict(status=status, eps=epsn, steps=step, peakC=peakC, hist=np.array(hist), TH=TH)

if __name__ == '__main__':
    for kw in [dict(kind='rad', fnu=0.0, K=2), dict(kind='rad', fnu=0.409)]:
        for dm in [0.4975, 0.4980] if kw['fnu'] == 0 else [0.5280, 0.5287]:
            t1 = time.time(); out = run(dm, **kw)
            print(kw, dm, out['status'], f"eps={out['eps']:.2f} ({time.time()-t1:.0f}s)")
