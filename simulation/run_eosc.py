"""Plasma fluid (photons + e+-, full EOS) + neutrinos with an energy-integrated relaxation-time collision term
Gamma = H_b (T_b/T_dec)^3 (T/T_b)^5 (momentum given to the plasma).  Lapse integrated from the acceleration."""
import numpy as np, time
from ms_eosc import rhs, timestep
from ms_fluid import initial_data, make_grid
from run_eos import make_eos, TB

def run(delta_m, T0, Tdec=2.0, N=300, K=32, rmax=9.0, beta=3.0, eps0=0.1, cfl_r=0.4, cfl_mu=0.4, cH=0.005,
        eps_max=15.0, qshape=1.0, hist_every=20, disperse_rule=True, snap_eps=(), collide=True, switch_ratio=100.0):
    # While Gamma/H = (T/Tdec)^3 > switch_ratio the neutrinos are tightly coupled: evolve one perfect fluid,
    # then split it into plasma + isotropic neutrinos and switch on the transport with collisions.
    tight = collide and (T0/Tdec)**3 > switch_ratio
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
    I = np.repeat((rnu/(4*np.pi))[:, None], K, axis=1)
    Mi = np.concatenate([[0.0], np.cumsum((4*np.pi/3)*np.diff(R**3)*rho)])
    Gam = np.ones(N+1); Gam[1:] = np.sqrt(1.0 + U[1:]**2 - 2*Mi[1:]/R[1:])
    t = t0
    dR = np.zeros(N+1); dU = np.zeros(N+1); dG = np.zeros(N+1); de = np.zeros(N); dI = np.zeros((N, K))
    Ib = lambda ee: np.exp(np.interp(np.log(ee[-1]), le, lnu))/(4*np.pi)
    peakC = -1.0; hist = []; step = 0; fallback = False; status = 'undecided'; TH = None; snaps = {}; snap_eps = list(snap_eps)
    while True:
        Hb = U[-1]/R[-1]; epsn = r[-1]/U[-1]; rhob = 3*Hb*Hb/(8*np.pi)
        Tb = float(np.exp(np.interp(np.log(e[-1]), le, lT)))
        if tight and (Tb/Tdec)**3 <= switch_ratio:
            Tl = np.exp(np.interp(np.log(e), le, lT)); fn = np.interp(Tl, np.exp(lT), fn_tab)
            I = np.repeat((fn*e/(4*np.pi))[:, None], K, axis=1); e = (1-fn)*e
            tab, Tof = tab_fs, Tof_fs; le, lp, lg, lnu = tab['le'], tab['lp'], tab['lg'], tab['lnu']
            tight = False
            Tb = float(np.exp(np.interp(np.log(e[-1]), le, lT)))
        kb = Hb*(Tb/Tdec)**3 if (collide and not tight) else 0.0
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
        if step % hist_every == 0:
            dmu = mue[1]-mue[0]
            Iout = I[-1].sum()
            hist.append((t, epsn, C[iC], r[1+iC], e[0]/e[-1], (I[0].sum()/Iout if Iout > 0 else np.nan), Tb, kb/Hb))
        if snap_eps and epsn >= snap_eps[0]:
            snaps[snap_eps.pop(0)] = dict(eps=epsn, R=R.copy(), U=U.copy(), e=e.copy(), I=I.copy(), M=M.copy(), Gam=Gam.copy(), Hb=Hb)
        if np.any((twoMR >= 1.0) & (U[1:] < 0)):
            status = 'BH'; break
        if disperse_rule and epsn > 3.0 and C[iC] < 0.5*peakC:
            status = 'dispersed'; break
        if epsn > eps_max: break
        try:
            dt = min(timestep(R, U, e, I, mue, Gam, Ac, ac, S, cfl_r, cfl_mu), cH/Hb)
            km = np.max(Ac*kap)
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
    return dict(status=status, fallback=fallback, eps=epsn, steps=step, peakC=peakC, hist=np.array(hist), TH=TH, snaps=snaps,
                r=r, R=R, U=U, e=e, I=I, mue=mue)

if __name__ == '__main__':
    # quick checks: (i) strongly coupled limit at high T reproduces the perfect fluid; (ii) timing
    for T0, Td, dm in [(40.0, 1.5, 0.4975), (40.0, 1.5, 0.4985), (5.0, 2.0, 0.52), (5.0, 2.0, 0.54)]:
        t1 = time.time(); out = run(dm, T0, Tdec=Td)
        print(f"T0={T0} Tdec={Td} dm={dm}: {out['status']} eps={out['eps']:.2f} TH={out['TH']:.3f} steps={out['steps']} ({time.time()-t1:.0f}s)", flush=True)
