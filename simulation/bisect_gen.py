import sys, json, time
from bisect_nu import bisect
# usage: bisect_gen.py tag fnu K eps0 N lo hi qshape [tol]
tag, fnu, K, eps0, N, lo, hi, qs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])
tol = float(sys.argv[9]) if len(sys.argv) > 9 else 5e-4
cH = float(sys.argv[10]) if len(sys.argv) > 10 else 0.005
rmax = float(sys.argv[11]) if len(sys.argv) > 11 else 9.0
t1 = time.time()
dc, w, log = bisect(lo, hi, tol=tol, tag=tag, fnu=fnu, K=K, eps0=eps0, N=N, qshape=qs, cH=cH, eps_max=15.0, rmax=rmax)
print(f"RESULT {tag}: delta_c = {dc:.5f} +- {w/2:.5f}  ({(time.time()-t1)/60:.1f} min)")
json.dump(dict(tag=tag, fnu=fnu, K=K, eps0=eps0, N=N, qshape=qs, rmax=rmax, cH=cH, dc=dc, w=w, log=log), open(f'res_{tag}.json', 'w'))
