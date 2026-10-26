import time
from run_eosc import run
for dm in [0.5084, 0.5093]:
    t1 = time.time(); o = run(dm, 40.0, Tdec=2.0)
    print(f"T0=40 Tdec=2 dm={dm}: {o['status']} eps={o['eps']:.2f} steps={o['steps']} ({time.time()-t1:.0f}s)", flush=True)
