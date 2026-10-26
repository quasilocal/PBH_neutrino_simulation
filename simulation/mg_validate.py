"""Validation of the multigroup code: with the energy-independent (gray) rate it must reproduce run_eosc.py."""
import time, sys
from run_mg import run
import run_eosc
T0 = 20.0
for dm in (0.5205, 0.5220):
    t1 = time.time(); a = run(dm, T0, mode='gray', Tdec=2.0, G=8)
    t2 = time.time(); b = run_eosc.run(dm, T0, Tdec=2.0)
    print(f'delta={dm}: multigroup(gray, G=8) {a["status"]} eps={a["eps"]:.4f} peakC={a["peakC"]:.6f} ({t2-t1:.0f}s) | '
          f'gray code {b["status"]} eps={b["eps"]:.4f} peakC={b["peakC"]:.6f} ({time.time()-t2:.0f}s)', flush=True)
