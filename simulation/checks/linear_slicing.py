"""Linear theory only (no collapse runs): which density the single-mode factors describe, and how the
compaction on plasma-orthogonal slices (the slicing of the simulations, including the flux term of the mass
constraint) responds to free streaming.  Appendix B and Sec. IV C.  MB conformal Newtonian gauge, k tau = x.
Neutrinos isotropic (one fluid) at tau0, as in the simulations."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from linear_nu import rhs, initial
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn as jn

def mode_iso(k, tau0, taus, R, lmax=40):
    x0 = k*tau0; psi = 2/3; phi = psi; d = -2*psi - (2/3)*x0**2*phi
    y0 = np.zeros(5+lmax-1); y0[:5] = [phi, d, 0.5*x0*psi, d, 0.5*x0*psi]
    xs = k*np.asarray(taus)
    s = solve_ivp(rhs, (x0, xs[-1]), y0, t_eval=xs, args=(R, lmax), method='DOP853', rtol=1e-10, atol=1e-13)
    return xs, s.y

def fields(y, xs, R):
    phi, dg, tg, dn, tn = y[:5]
    D_tot = -(2/3)*xs**2*phi                       # zero total momentum slices (Poisson)
    dpl = (1-R)*(dg + 4*tg/xs) + R*(dn + 4*tg/xs)  # plasma-orthogonal slices
    return D_tot, dpl, tg, tn                      # theta/k

if __name__ == '__main__':
    xs_grid = np.linspace(0.1, 8, 8001); m = (xs_grid > 1.0) & (xs_grid < 6.5); ref = None
    print('single mode, isotropic at x0 = 0.1: first-maximum ratios relative to R = 0')
    for R in (0.0, 0.409, 0.488):
        xs, y = mode_iso(1.0, 0.1, xs_grid, R); Dt, dp, _, _ = fields(y, xs, R)
        v = (np.abs(Dt)[m].max(), np.abs(dp)[m].max()); ref = ref or v
        print(f'  R={R:5.3f}: zero-momentum slices {v[0]/ref[0]:.4f}   plasma slices {v[1]/ref[1]:.4f}')
    print('superhorizon (x = 0.05) coefficients relative to R = 0')
    for label in ('isotropic at x0=0.005', 'MB regular mode'):
        base = None
        for R in (0.0, 0.409, 0.488):
            if label.startswith('iso'):
                xs, y = mode_iso(1.0, 0.005, np.array([0.05]), R)
            else:
                s = solve_ivp(rhs, (1e-4, 0.05), initial(R, 1e-4, 40), t_eval=[0.05], args=(R, 40), method='DOP853', rtol=1e-11, atol=1e-14)
                xs, y = np.array([0.05]), s.y
            Dt, dp, _, _ = fields(y, xs, R); c = (Dt[0], dp[0]); base = base or c
            print(f'  {label:22s} R={R:5.3f}: zero-momentum {c[0]/base[0]:.4f}   plasma slices {c[1]/base[1]:.4f}')
    # Gaussian K profile, r_m = 1, decoupling at eps0 = 0.1
    ks = np.linspace(0.02, 12, 600); dk = ks[1]-ks[0]; zk = np.exp(-ks**2/4)
    taus = np.linspace(0.12, 7.0, 700); rr = np.linspace(1e-4, 1.0, 400)
    J0 = jn(0, np.outer(ks, rr)); J1 = jn(1, np.outer(ks, rr)); w = (ks**2*zk*dk/(2*np.pi**2))
    out = {}
    for R in (0.0, 0.409):
        A = np.zeros((4, len(ks), len(taus)))
        for i, k in enumerate(ks):
            xs, y = mode_iso(k, 0.1, taus, R); Dt, dp, tg, tn = fields(y, xs, R)
            A[:, i] = Dt, dp, tn - tg, dp
        Dt_r = J0.T @ (w[:, None]*A[0]); dp_r = J0.T @ (w[:, None]*A[1]); dv_r = J1.T @ (w[:, None]*A[2])
        vol = lambda f: 3*np.trapezoid(rr[:, None]**2*f, rr, axis=0)
        flux = (rr[:, None]/taus[None, :])*(4/3)*R*dv_r          # H R q / rho with v_r = int k^2 theta/k j1
        out[R] = dict(C_tot=vol(Dt_r)/taus**2, C_sim=vol(dp_r + flux)/taus**2,
                      cen_tot=w @ A[0], cen_pl=w @ A[1])
    for e in (0.2, 0.3, 0.5, 0.7, 1.0):
        print(f'eps={e}: compaction at r_m, simulation slicing ratio {np.interp(e,taus,out[0.409]["C_sim"])/np.interp(e,taus,out[0.0]["C_sim"]):.4f};'
              f' zero-momentum slicing {np.interp(e,taus,out[0.409]["C_tot"])/np.interp(e,taus,out[0.0]["C_tot"]):.4f}')
    mm = taus > 0.8
    for key in ('cen_tot', 'cen_pl'):
        a0, a1 = np.abs(out[0.0][key]), np.abs(out[0.409][key])
        print(f'centre density, {key}: first-maximum ratio {a1[mm].max()/a0[mm].max():.4f} at eps {taus[mm][np.argmax(a1[mm])]:.2f}')
