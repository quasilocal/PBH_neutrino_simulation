"""Multigroup driver (cf. run_eosc2.py). Two flavours (nu_e: 1/3 of the neutrino energy, nu_mu,tau: 2/3), each split into
G groups of comoving energy x = E/T_nu,b between x_lo and x_hi (log-spaced).

mode='phys': energy-dependent weak rates from multigroup/weak_rates.json (Monte Carlo over the processes of Hannestad &
             Madsen 1995 with Fermi-Dirac statistics and Pauli blocking):
               nu - e+- : Gamma = c_e,f(x) G_F^2 E T^4 chi_e(T)   (relaxation to isotropy in the plasma frame)
               nu - nu  : Gamma = c_n(x)  G_F^2 E T_nu^4          (relaxation to isotropy in the neutrino frame)
             chi_e = rho_e+-(T)/rho_e+-(T, m_e = 0) corrects the e+- density for the electron mass.
             Rates above CAP*H are capped at CAP*H (the groups are then tightly coupled either way).
             The neutrinos and plasma are evolved as one perfect fluid while the total rate of nu_mu,tau with
             E = T_nu exceeds SWITCH*H.
mode='gray': the energy-independent rate of run_eosc2.py, kap = H_b (T_b/T_dec)^3 (T/T_b)^5 for every group and no
             nu - nu term. The sum over groups then obeys exactly the equations of the gray code (validation)."""
import numpy as np, json, os, time
from ms_mg import rhs, timestep, state
from ms_fluid import initial_data, make_grid
from run_eos import make_eos, TB

HERE = os.path.dirname(os.path.abspath(__file__))
FR = np.array([1/3, 2/3])
GF2 = (1.1663787e-11)**2      # MeV^-4
MPL = 1.220890e22             # MeV
_T = TB['T'][::-1]; _rpl = TB['rpl'][::-1]; _rnu = TB['rnu'][::-1]
_W = json.load(open(os.path.join(HERE, 'multigroup', 'weak_rates.json')))

def background(T):
    """Physical background at plasma temperature T [MeV]: H [MeV], T_nu [MeV], chi_e."""
    rpl = np.interp(T, _T, _rpl); rnu = np.interp(T, _T, _rnu)
    H = np.sqrt(8*np.pi*(rpl + rnu)/3)/MPL
    Tnu = (rnu/(5.25*np.pi**2/30))**0.25
    chi = (rpl - (np.pi**2/15)*T**4)/((7*np.pi**2/60)*T**4)
    return H, Tnu, chi

def coef(table, x):
    xs = np.array(_W[table]['x'])
    return lambda key: np.interp(np.log(x), np.log(xs), np.array(_W[table][key]))

def groups(G, x_lo, x_hi):
    edges = np.exp(np.linspace(np.log(x_lo), np.log(x_hi), G+1))
    return edges, np.sqrt(edges[:-1]*edges[1:]), np.log(x_hi/x_lo)/G

def spectral_weights(edges, tau):
    """Fraction of the energy of a Fermi-Dirac spectrum with temperature tau (in units of T_nu,b) in each group,
    renormalized to the groups' range. tau may be an array (one value per cell): returns (G, len(tau))."""
    tau = np.atleast_1d(tau)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        x = np.exp(np.linspace(np.log(a), np.log(b), 41))[:, None]
        y = (x/tau)**4/(np.exp(np.minimum(x/tau, 700)) + 1)          # energy per ln x
        out.append(np.trapezoid(y, np.log(x[:, 0]), axis=0))
    out = np.array(out)
    return out/out.sum(axis=0)

def run(delta_m, T0, mode='phys', G=8, x_lo=0.25, x_hi=25.0, Tdec=2.0, scale_e=1.0, scale_n=1.0, nunu=True,
        CAP=100.0, SWITCH=100.0, N=300, K=32, rmax=9.0, beta=3.0, eps0=0.1, cfl_r=0.4, cfl_mu=0.4, cfl_E=0.4,
        cH=0.005, eps_max=15.0, qshape=1.0, disperse_rule=True, verbose=False, chi_local=False, nnmode=0,
        diag_every=0, ckap=0.5):
    """chi_local: evaluate the e+- mass correction chi_e at the local plasma temperature (default: background,
    as in the archived runs). nnmode = 1: nu-nu target isotropic in a common neutrino rest frame, with the frame
    velocity fixed by the rate-weighted flux so that the term conserves the total neutrino momentum and the energy
    of each group (default 0: monopole + dipole of each group in the plasma frame). diag_every > 0: record flux
    ratios, the top-group energy fraction and the minimum intensity every diag_every steps. ckap: collisional
    time-step limit, kappa dt <= ckap."""
    edges, xg, dl = groups(G, x_lo, x_hi)
    NF = 2; S_ = NF*G
    if mode == 'phys':
        ce_e = coef('e_FD', xg)('nu_e'); ce_m = coef('mu_FD', xg)('nu_e'); cn = coef('e_FD', xg)('nu_nu')
        c1 = coef('mu_FD', np.array([1.0]))
        c_sw = (c1('nu_e')[0], c1('nu_nu')[0])
        pe = 4.0
        def rates(Tb):
            H, Tnu, chi = background(Tb)
            re = np.concatenate([ce_e, ce_m])*GF2*np.tile(xg, NF)*Tnu*Tb**4*chi/H*scale_e
            rn = np.tile(cn, NF)*GF2*np.tile(xg, NF)*Tnu**5/H*scale_n*(1.0 if nunu else 0.0)
            return np.minimum(re, CAP), np.minimum(rn, CAP)
        def tight_at(Tb):
            H, Tnu, chi = background(Tb)
            tot = (c_sw[0]*Tb**4*chi*scale_e + c_sw[1]*Tnu**4*scale_n*(1.0 if nunu else 0.0))*GF2*Tnu/H
            return tot > SWITCH
    else:
        pe = 5.0
        def rates(Tb):
            return np.full(S_, (Tb/Tdec)**3), np.zeros(S_)
        def tight_at(Tb):
            return (Tb/Tdec)**3 > SWITCH
    tight = tight_at(T0)
    tab, frac, Tof = make_eos('pf' if tight else 'fs', T0, eps0)
    le, lp, lg, lnu = tab['le'], tab['lp'], tab['lg'], tab['lnu']
    lT = np.log(_T)
    chi_T = (_rpl - (np.pi**2/15)*_T**4)/((7*np.pi**2/60)*_T**4)
    lchi = np.log(np.maximum(chi_T, 1e-300))
    def chi_at(T): return float(np.exp(np.interp(np.log(T), lT, lchi)))
    D = dict(qmax=0.0, qmax_at=None, qmax_hot=0.0, qmax_hot_at=None, top=0.0, top_at=None)
    tab_fs, frac_fs, Tof_fs = make_eos('fs', T0, eps0)
    fn_tab = (_rnu/(_rnu + _rpl))
    r = make_grid(N, rmax, beta)
    t0, R, U, rho = initial_data(r, delta_m, eps0, 0.0, qshape)
    e, rnu = frac(rho)
    mue = np.linspace(-1, 1, K+1)
    fl_of_s = np.repeat(np.arange(NF), G)
    def split(e_tot_or_nu, enu_b):
        """Neutrino intensities isotropic in the plasma frame, Fermi-Dirac spectrum at the local temperature."""
        tau = (np.maximum(e_tot_or_nu, 1e-300)/enu_b)**0.25
        w = spectral_weights(edges, tau)                               # (G, N)
        I = np.empty((S_, N, K))
        for s in range(S_):
            I[s] = np.repeat((FR[fl_of_s[s]]*w[s % G]*e_tot_or_nu/(4*np.pi))[:, None], K, axis=1)
        return I
    if tight:
        I = np.zeros((1, N, K))                      # dummy while neutrinos and plasma are one perfect fluid
    else:
        enu_b0 = np.exp(np.interp(np.log(e[-1]), le, lnu))
        I = split(rnu, enu_b0)
    Mi = np.concatenate([[0.0], np.cumsum((4*np.pi/3)*np.diff(R**3)*rho)])
    Gam = np.ones(N+1); Gam[1:] = np.sqrt(1.0 + U[1:]**2 - 2*Mi[1:]/R[1:])
    t = t0
    dR = np.zeros(N+1); dU = np.zeros(N+1); dG = np.zeros(N+1); de = np.zeros(N); dI = np.zeros(I.shape)
    wb = spectral_weights(edges, 1.0)[:, 0]
    w_s = np.array([FR[fl_of_s[s]]*wb[s % G] for s in range(S_)])
    def Ib_and_enub(ee):
        enub = np.exp(np.interp(np.log(ee[-1]), le, lnu))
        return w_s*enub/(4*np.pi), enub
    peakC = -1.0; step = 0; fallback = False; status = 'undecided'; TH = None; t_start = time.time()
    while True:
        Hb = U[-1]/R[-1]; epsn = r[-1]/U[-1]; rhob = 3*Hb*Hb/(8*np.pi)
        Tb = float(np.exp(np.interp(np.log(e[-1]), le, lT)))
        if tight and not tight_at(Tb):
            Tl = np.exp(np.interp(np.log(e), le, lT)); fn = np.interp(Tl, _T, fn_tab)
            enu = fn*e
            tab, Tof = tab_fs, Tof_fs; le, lp, lg, lnu = tab['le'], tab['lp'], tab['lg'], tab['lnu']
            e = (1-fn)*e
            enu_b = np.exp(np.interp(np.log(e[-1]), le, lnu))
            I = split(enu, enu_b); dI = np.zeros(I.shape)
            tight = False
            Tb = float(np.exp(np.interp(np.log(e[-1]), le, lT)))
            if verbose: print(f'   switch to transport at T_b = {Tb:.3f} MeV, eps = {epsn:.3f}', flush=True)
        Gu = 1 if tight else G
        if tight:
            kbe = np.zeros(1); kbn = np.zeros(1); Ib = np.zeros(1); enub = 1.0
        else:
            re, rn = rates(Tb)
            kbe = Hb*re; kbn = Hb*rn
            Ib, enub = Ib_and_enub(e)
        chib = chi_at(Tb) if (chi_local and not tight) else 0.0
        try:
            M, Ac, ac, S, V, kap_e, kap_n, vmax = rhs(t, R, U, Gam, e, I, mue, le, lp, lg, lT, kbe, kbn, Tb, enub, pe,
                                                      Gu, dl, Hb, Ib, dR, dU, dG, de, dI, lchi, chib, nnmode)
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
        if diag_every > 0 and not tight and step % diag_every == 0:
            st_out = state(t, R, U, Gam, e, I, mue, le, lp, lg, lT, kbe, kbn, Tb, enub, pe, lchi, chib)
            eps_s, q_s, kn = st_out[0], st_out[1], st_out[10]
            ecell = eps_s.sum(axis=0)
            ok = eps_s > 1e-3*ecell[None, :]                      # groups holding at least 0.1% of the cell's neutrino energy
            ratio = np.where(ok, np.abs(q_s)/np.where(ok, eps_s, 1.0), 0.0)
            coll = (kn > Hb) & ok
            Irel = np.where(ok[:, :, None], I/np.where(ok, eps_s/(4*np.pi), 1.0)[:, :, None], 1.0)
            if Irel.min() < D.get('Imin_rel', 1.0):
                D['Imin_rel'] = float(Irel.min())
            if I.min() < D.get('Imin_abs', np.inf):
                D['Imin_abs'] = float(I.min()); D['Imin_abs_frac'] = float(I.min()/(eps_s.sum(axis=0).max()/(4*np.pi)))
            if coll.any():
                j = np.unravel_index(np.argmax(np.where(coll, ratio, 0.0)), ratio.shape)
                if ratio[j] > D['qmax']:
                    D['qmax'] = float(ratio[j]); D['qmax_at'] = (float(r[j[1]]), float(epsn), Tb)
                if Tb > 1.0 and ratio[j] > D['qmax_hot']:
                    D['qmax_hot'] = float(ratio[j]); D['qmax_hot_at'] = (float(r[j[1]]), float(epsn), Tb)
            qt = q_s.sum(axis=0); et = eps_s.sum(axis=0)          # total neutrino flux / energy of each cell (plasma frame)
            rt = np.where(et > 0, np.abs(qt)/np.where(et > 0, et, 1.0), 0.0); jr = int(np.argmax(rt))
            for key, lo, hi in (('qtot_hot', 1.0, np.inf), ('qtot_12', 1.0, 2.0)):
                if lo < Tb < hi and rt[jr] > D.get(key, 0.0):
                    D[key] = float(rt[jr]); D[key + '_at'] = (float(r[jr]), float(epsn), Tb)
            ef = eps_s.reshape(NF, G, -1); tot = ef.sum(axis=1)
            okf = tot > 1e-12*tot.max()
            frac_top = np.where(okf, ef[:, -1, :]/np.where(okf, tot, 1.0), 0.0)
            jt = np.unravel_index(np.argmax(frac_top), frac_top.shape)
            if frac_top[jt] > D['top']:
                D['top'] = float(frac_top[jt]); D['top_at'] = (float(r[jt[1]]), float(epsn), Tb)
        try:
            dt = min(timestep(R, U, e, I, mue, Gam, Ac, ac, S, cfl_r, cfl_mu), cH/Hb)
            km = np.max(Ac[None, :]*(kap_e + kap_n))
            if km > 0: dt = min(dt, ckap/km)              # collisional limit: kappa dt <= ckap (0.5 in the archived runs)
            if vmax > 0: dt = min(dt, cfl_E*dl/vmax)
            R1 = R + dt*dR; U1 = U + dt*dU; G1 = Gam + dt*dG; e1 = e + dt*de; I1 = I + dt*dI
            Ib1, enub1 = (Ib_and_enub(e1) if not tight else (Ib, enub))
            rhs(t+dt, R1, U1, G1, e1, I1, mue, le, lp, lg, lT, kbe, kbn, Tb, enub1, pe, Gu, dl, Hb, Ib1, dR, dU, dG, de, dI,
                lchi, chib, nnmode)
            R2 = 0.75*R + 0.25*(R1+dt*dR); U2 = 0.75*U + 0.25*(U1+dt*dU); G2 = 0.75*Gam + 0.25*(G1+dt*dG)
            e2 = 0.75*e + 0.25*(e1+dt*de); I2 = 0.75*I + 0.25*(I1+dt*dI)
            Ib2, enub2 = (Ib_and_enub(e2) if not tight else (Ib, enub))
            rhs(t+0.5*dt, R2, U2, G2, e2, I2, mue, le, lp, lg, lT, kbe, kbn, Tb, enub2, pe, Gu, dl, Hb, Ib2, dR, dU, dG, de, dI,
                lchi, chib, nnmode)
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
    return dict(status=status, fallback=fallback, eps=epsn, steps=step, peakC=peakC, TH=TH,
                seconds=time.time()-t_start, I=I, e=e, r=r, diag=D)

if __name__ == '__main__':
    # where do the groups decouple? total rate / H at the group energies, for a few temperatures
    edges, xg, dl = groups(8, 0.25, 25.0)
    print('group centres x:', np.round(xg, 2), ' energy fractions:', np.round(spectral_weights(edges, 1.0)[:, 0], 4))
    for Tb in (8.0, 4.0, 2.0, 1.5, 1.0, 0.7):
        H, Tnu, chi = background(Tb)
        for fl, key in (('e', 'e_FD'), ('mu', 'mu_FD')):
            c = coef(key, xg)
            re = c('nu_e')*GF2*xg*Tnu*Tb**4*chi/H; rn = c('nu_nu')*GF2*xg*Tnu**5/H
            print(f'T={Tb:4.1f} MeV {fl:2s} chi={chi:.3f}  nu-e/H:', ' '.join(f'{v:8.2f}' for v in re),
                  '| nu-nu/H:', ' '.join(f'{v:8.2f}' for v in rn))
