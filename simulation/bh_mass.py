"""Apparent-horizon mass at formation for supercritical runs, perfect fluid (fnu=0) vs free-streaming neutrinos.
Same code, resolution and initial data for both; usage: python3 bh_mass.py fnu dc x1,x2,... [N] [K]"""
import sys, json, time, numpy as np
import run_nu
from ms_nu import rhs as _rhs

def run_ah(delta_m, fnu, N=400, K=32, eps0=0.1):
    """Wrap run_nu.run: stop at the first trapped shell and return the apparent-horizon data."""
    rec = {}
    def diag(t, epsn, r, R, U, e, I, M, mue, C):
        return None
    out = run_nu.run(delta_m, fnu=fnu, N=N, K=K, eps0=eps0, hist_every=10**9, stop_rule=True, disperse_rule=True,
                     eps_max=30.0)
    return out

if __name__ == '__main__':
    fnu = float(sys.argv[1]); dc = float(sys.argv[2]); xs = [float(v) for v in sys.argv[3].split(',')]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 400; K = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    res = []
    for x in xs:
        t1 = time.time(); out = run_ah(dc + x, fnu, N, K)
        R, U, r = out['R'], out['U'], out['r']
        # recompute M on the final state to locate the trapped region
        from ms_nu import rhs
        N1 = len(R); dR = np.zeros(N1); dU = np.zeros(N1); dG = np.zeros(N1); de = np.zeros(N1-1); dI = np.zeros_like(out['I'])
        e, I, Gam, mue = out['e'], out['I'], out['Gam'], out['mue']
        Ib = (fnu/(1-fnu)*e[-1]/(4*np.pi)) if fnu > 0 else 0.0
        M = rhs(out['t'], R, U, Gam, e, I, mue, fnu, Ib, dR, dU, dG, de, dI)[0]
        twoMR = 2*M[1:]/R[1:]; trapped = (twoMR >= 1.0) & (U[1:] < 0)
        tH = out['t0']/0.1**2            # horizon crossing (eps = 1) for eps0 = 0.1; M_H = t_H in code units
        if trapped.any():
            i = np.where(trapped)[0].max() + 1          # outermost trapped edge (index into R, M)
            # interpolate 2M/R = 1 between edge i and i+1 (outer boundary of the trapped region)
            f0, f1 = 2*M[i]/R[i] - 1, 2*M[i+1]/R[i+1] - 1
            w = f0/(f0 - f1) if f0 != f1 else 0.0
            Rah = R[i] + w*(R[i+1]-R[i]); Mah = 0.5*Rah
            item = dict(x=x, status=out['status'], eps=float(out['eps']), M_AH=float(Mah/tH), r_AH=float(r[i]), fallback=False)
        else:
            j = np.argmax(np.where(U[1:] < 0, twoMR, 0)) + 1
            item = dict(x=x, status=out['status'], eps=float(out['eps']), M_AH=float(M[j]/tH), r_AH=float(r[j]),
                        fallback=True, max2MR=float(twoMR[j-1]))
        res.append(item)
        print(f"  fnu={fnu} x={x:.1e}: {item['status']} eps={item['eps']:.2f} M_AH/M_H={item['M_AH']:.4f} "
              f"r_AH={item['r_AH']:.4f} fallback={item['fallback']} ({time.time()-t1:.0f}s)", flush=True)
        json.dump(res, open(f'res_bhmass_fnu{fnu}_N{N}_K{K}.json', 'w'))
