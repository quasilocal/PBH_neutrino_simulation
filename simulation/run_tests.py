"""Quick checks of the code against numbers quoted in the paper (Sec. III B, App. B). About five minutes on one core.
Each run also reports four sanity checks: positivity of the neutrino intensity, monotonic areal radius (no shell
crossing), the residual of the constraint Gamma^2 = 1 + U^2 - 2M/R, and whether the emergency fallback rule
(breakdown with 2M/R > 0.95, counted as a black hole) decided the outcome."""
import time, numpy as np
from run_nu import run
from run_nuc import run as run_nuc
from ms_nu import rho_b
import linear_nu

def checks(out):
    """Positivity, mesh, constraint and fallback status for the final state of a run."""
    R, U, G, e, I = out['R'], out['U'], out['Gam'], out['e'], out['I']
    mue = out['mue']; dmu = mue[1] - mue[0]; mu = 0.5*(mue[:-1] + mue[1:])
    eps = 2*np.pi*I.sum(axis=1)*dmu; q = 2*np.pi*(I*mu).sum(axis=1)*dmu
    M = np.zeros(len(R))
    for i in range(len(R) - 1):
        dV = (4*np.pi/3)*(R[i+1]**3 - R[i]**3)
        M[i+1] = M[i] + dV*(e[i] + eps[i] + q[i]*0.5*(U[i] + U[i+1])/(0.5*(G[i] + G[i+1])))
    res = np.abs(G[1:]**2 - 1 - U[1:]**2 + 2*M[1:]/R[1:])
    return (f"min I = {I.min():.1e}, min dR = {np.diff(R).min():.1e}, max constraint residual = {res.max():.1e}, "
            f"fallback = {out.get('fallback', 'n/a')}")

def show(label, out, expect):
    print(f"{label:46s} {out['status']:9s} (expected {expect:9s}) eps = {out['eps']:5.2f}  [{checks(out)}]", flush=True)

t0 = time.time()
print('1. Homogeneous universe (Sec. III B b): anisotropy and inhomogeneity < 1e-9; mean-density drift -1.4e-6 at eps = 3')
out = run(0.0, fnu=0.409, N=400, K=32, eps0=0.1, eps_max=3.0, stop_rule=False)
Im = out['I'].mean(axis=1)
print(f"   anisotropy {np.abs(out['I']/Im[:, None] - 1).max():.1e}, inhomogeneity {np.ptp(out['e'])/out['e'].mean():.1e}, "
      f"drift e/e_b - 1 = {out['e'].mean()/(0.591*rho_b(out['t'])) - 1:.1e}", flush=True)

print('2. Linear theory (App. B), neutrinos decoupled from the start: superhorizon factors 1.049 (R_nu = 0.409) and 1.058 (0.488);\n'
      '   first maximum of Delta lower by the factors 0.911 and 0.897')
res = {R: linear_nu.evolve(R) for R in (0.0, 0.409, 0.488)}
for R in (0.409, 0.488):
    D, t = res[R]['Delta'], res[R]['t']; D0, t00 = res[0.0]['Delta'], res[0.0]['t']
    print(f"   R_nu = {R}: superhorizon {res[R]['phi'][0]/res[0.0]['phi'][0]:.3f}, "
          f"first maximum {np.abs(D[t < 8]).max()/np.abs(D0[t00 < 8]).max():.3f}", flush=True)

print('3. Thresholds bracketed by two runs each (N = 400, N_mu = 32, eps0 = 0.1)')
show('   perfect fluid, delta_m = 0.4965 (d_c = 0.49773)', run(0.4965, fnu=0.0, eps_max=15.0), 'dispersed')
show('   perfect fluid, delta_m = 0.4990', run(0.4990, fnu=0.0, eps_max=15.0), 'BH')
show('   free streaming, delta_m = 0.5270 (d_c = 0.52832)', run(0.5270, fnu=0.409, eps_max=15.0), 'dispersed')
show('   free streaming, delta_m = 0.5295', run(0.5295, fnu=0.409, eps_max=15.0), 'BH')
show('   strong coupling, delta_m = 0.4970 (d_c = 0.4980)', run_nuc(0.4970, kap=100.0, eps_max=15.0), 'dispersed')
show('   strong coupling, delta_m = 0.4995', run_nuc(0.4995, kap=100.0, eps_max=15.0), 'BH')
print(f'done in {(time.time() - t0)/60:.1f} min')
