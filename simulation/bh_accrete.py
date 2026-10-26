"""Follow the run past apparent-horizon formation (no stop rule) and record the AH mass until the code breaks down."""
import sys, json, numpy as np, time
import run_nu
def diag(t, epsn, r, R, U, e, I, M, mue, C):
    twoMR = 2*M[1:]/R[1:]; trapped = (twoMR >= 1.0) & (U[1:] < 0)
    if not trapped.any(): return (float(epsn), np.nan, np.nan, float(np.max(twoMR)))
    i = np.where(trapped)[0].max() + 1
    f0, f1 = 2*M[i]/R[i] - 1, 2*M[i+1]/R[i+1] - 1; w = f0/(f0 - f1)
    Rah = R[i] + w*(R[i+1] - R[i])
    return (float(epsn), float(0.5*Rah), float(r[i]), float(np.max(twoMR)))
fnu, delta = float(sys.argv[1]), float(sys.argv[2])
t1 = time.time()
out = run_nu.run(delta, fnu=fnu, N=400, K=32, stop_rule=False, eps_max=14.0, hist_every=5, diag=diag)
tH = out['t0']/0.1**2
h = [(a, (b/tH if b == b else None), c, d) for a, b, c, d in out['hist2']]
json.dump(dict(fnu=fnu, delta=delta, status=out['status'], eps_end=float(out['eps']), hist=h), open(f'res_accrete_fnu{fnu}_d{delta}.json', 'w'))
ah = [x for x in h if x[1] is not None]
print(f"fnu={fnu} delta={delta}: {out['status']} at eps={out['eps']:.2f} ({time.time()-t1:.0f}s); AH from eps={ah[0][0]:.2f} "
      f"M_AH={ah[0][1]:.4f} to eps={ah[-1][0]:.2f} M_AH={ah[-1][1]:.4f} ({len(ah)} samples)" if ah else f"no AH; {out['status']}", flush=True)
