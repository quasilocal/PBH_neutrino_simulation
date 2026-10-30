import time
from run_eos import run
# regression: fs with switch at eps_dec=0.2 vs pf, T0=5 (TH~0.5 MeV), one amplitude each side
for kw in [dict(kind='fs', T0=5.0, eps_dec=0.25), dict(kind='fs', T0=5.0)]:
    for dm in [0.52, 0.54]:
        t1 = time.time(); out = run(dm, **kw)
        print(kw, dm, out['status'], f"eps={out['eps']:.2f} TH={out['TH']:.3f} ({time.time()-t1:.0f}s)", flush=True)
