"""Physical threshold curve for the full model: multigroup results (bisect_mg.py, weak rates with nu-nu scattering,
6 groups) where computed, and the gray model with T_dec = 1 MeV elsewhere, where the neutrinos are either tightly
coupled (T_H >= 8 MeV) or free-streaming throughout the collapse (T_H <= 0.2 MeV) and the two agree.
Writes res_phys_<T0>.json in the format of res_eosc_*.json (dc = threshold, w = bracket width, TH, T0, source)."""
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
def load(pat):
    out = {}
    for f in glob.glob(os.path.join(HERE, pat)):
        d = json.load(open(f)); out[round(d['T0'], 3)] = d
    return out
mg = load('res_mg_phys6_*.json'); g1 = load('res_eosc_1.0_*.json')
for T0 in sorted(set(mg) | set(g1), reverse=True):
    d = mg.get(T0) or g1[T0]
    src = 'multigroup' if T0 in mg else 'gray T_dec = 1 MeV'
    json.dump(dict(T0=T0, TH=d['TH'], dc=d['dc'], w=d['w'], source=src),
              open(os.path.join(HERE, f'res_phys_{T0}.json'), 'w'))
    print(f'T0={T0:6.1f} TH={d["TH"]:.4f} dc={d["dc"]:.5f} +- {d["w"]/2:.5f}  ({src})')
