"""Collects the runs of scan.py (points/*.json): thresholds in res_<label>.json and both results in summary.txt.
The threshold lies between the largest dispersed and the smallest collapsing amplitude, and delta_c is the midpoint
+- half the gap. Shifts are relative to the perfect fluid at the same mass (res_eos_pf_<T0>.json), in delta_c and in
the linear compaction C_l,c = (4/3)[1 - (1 - 3 delta_c/2)^(1/2)]."""
import os, json, glob
from scan import HERE, SIM, OUT, GRIDS, label, amplitudes

Cl = lambda d: (4/3)*(1 - (1 - 1.5*d)**0.5)
pts = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, 'egrid_*.json'))]
lines = ['Energy grid: multigroup thresholds with the common rest-frame nu-nu target (nnmode = 1), from 8 runs per grid',
         'spaced 0.00025 apart about the archived threshold. Outcomes: D = dispersed, B = black hole, then eps at the end.', '']
for T0, TH in ((10.0, '1.00'), (5.0, '0.51')):
    pf = json.load(open(os.path.join(SIM, f'res_eos_pf_{T0}.json')))['dc']
    a = json.load(open(os.path.join(SIM, f'res_mg_phys6_{T0}.json')))['dc']
    lines.append(f'T_H = {TH} MeV: perfect fluid {pf:.5f}, archived (first-order target, 6 groups to 20) {a:.5f}, '
                 f'shift {100*(a/pf - 1):.2f}% in delta_c and {100*(Cl(a)/Cl(pf) - 1):.2f}% in C_l,c')
    for G, xh in GRIDS:
        lab = label(G, xh, T0)
        P = {p['delta']: (p['status'], p['eps']) for p in pts if p['label'] == lab}
        missing = [d for d in amplitudes(T0) if d not in P]
        D = [d for d, v in P.items() if v[0] == 'dispersed']; B = [d for d, v in P.items() if v[0] == 'BH']
        other = {d: v[0] for d, v in P.items() if v[0] not in ('dispersed', 'BH')}
        mono = not (D and B and max(D) > min(B))
        ok = bool(D and B and mono)
        lo, hi = (max(D), min(B)) if ok else (None, None)
        res = dict(label=lab, G=G, x_hi=xh, T0=T0, points={f'{d:.7f}': v for d, v in sorted(P.items())},
                   missing=missing, other=other, bracketed=ok, monotone=mono,
                   dc=0.5*(lo + hi) if ok else None, w=(hi - lo) if ok else None)
        json.dump(res, open(os.path.join(HERE, f'res_{lab}.json'), 'w'), indent=1)
        row = f'   G={G:2d}, E/T_nu,b <= {xh:<6g}'
        if ok:
            dc = res['dc']
            row += (f' {dc:.5f}+-{res["w"]/2:.5f}   shift {100*(dc/pf - 1):.2f}% / {100*(Cl(dc)/Cl(pf) - 1):.2f}%'
                    f'   minus archived {dc - a:+.5f}')
        else:
            row += '   NOT BRACKETED'
        if not mono: row += '   NOT MONOTONE'
        if missing: row += f'   missing {missing}'
        if other: row += f'   other outcomes {other}'
        lines.append(row)
        lines.append('      ' + '  '.join(f'{d:.5f} {v[0][0].upper()}{v[1]:.2f}' for d, v in sorted(P.items())))
    lines.append('')
lines.append('Top group (640 <= E/T_nu,b <= 1280) on the widest grid, T_H = 1 MeV: largest share of the energy of a flavour')
for f in sorted(glob.glob(os.path.join(OUT, 'top_*.json'))):
    t = json.load(open(f)); at = t['top_at']
    lines.append(f"   delta = {t['delta']:.6f}: {t['status']} at eps = {t['eps']:.2f}, top-group share at most {t['top']:.1e}"
                 + ('' if at is None else f", reached at r = {at[0]:.2f}, eps = {at[1]:.3f}, T_b = {at[2]:.2f} MeV"))
text = '\n'.join(lines)
print(text)
open(os.path.join(HERE, 'summary.txt'), 'w', encoding='utf-8').write(text + '\n')
