import numpy as np, time
from run_nu import run
from ms_nu import rho_b
for eps0, cH, N, K in [(0.02,0.005,100,8),(0.02,0.0025,100,8),(0.02,0.005,200,8),(0.05,0.005,100,8)]:
    out = run(0.0, fnu=0.409, N=N, K=K, eps0=eps0, eps_max=1.0, stop_rule=False, cH=cH)
    t = out['t']
    print(f"eps0={eps0} cH={cH} N={N}: steps={out['steps']}, drift at eps=1: {out['e'].mean()/(0.591*rho_b(t))-1:.3e}")
