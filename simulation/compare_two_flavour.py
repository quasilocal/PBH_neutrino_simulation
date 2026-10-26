"""Compare the two-flavour decoupling scan (nu_e at 1.5 MeV, nu_mu,tau at 2.5 MeV) with the one-flavour scans."""
import json, glob, os
P = os.path.dirname(os.path.abspath(__file__)) + '/'
def load(pat):
    out = {}
    for f in glob.glob(P + pat):
        d = json.load(open(f)); out[round(d['T0'], 3)] = (d['TH'], d['dc'], d['w']/2)
    return out
two = load('res_eosc2_1.5_2.5_*.json'); one = {t: load(f'res_eosc_{t}_*.json') for t in (1.5, 2.0, 3.0)}
pf = load('res_eos_pf_*.json')
chk = load('res_eosc2_2.0_2.0_*.json')
print(' T0   T_H    two-flavour        1.5      2.0      3.0     pf     | two/2.0-1  two/pf-1  2.0/pf-1')
for T0 in sorted(two, reverse=True):
    TH, d, e = two[T0]
    o = [one[t].get(T0, (0, float('nan')))[1] for t in (1.5, 2.0, 3.0)]
    p = pf.get(T0, (0, float('nan')))[1]
    print(f'{T0:5.0f} {TH:6.3f}  {d:.5f}+-{e:.5f}  ' + '  '.join(f'{x:.5f}' for x in o) + f'  {p:.5f} | {100*(d/o[1]-1):+6.2f}%  {100*(d/p-1):+6.2f}%  {100*(o[1]/p-1):+6.2f}%')
for T0, (TH, d, e) in chk.items():
    print(f'check, equal T_dec = 2 MeV at T0={T0}: {d:.5f}+-{e:.5f} vs one-flavour {one[2.0][T0][1]:.5f}+-{one[2.0][T0][2]:.5f}')
