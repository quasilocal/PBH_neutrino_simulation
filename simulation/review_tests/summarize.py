"""Table of the bracket tests (res_test_*.json): for each variant, the runs at the archived threshold - 0.001 and
+ 0.001 should disperse and form a black hole. Writes summary.txt next to this script."""
import os, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
DESC = {'baseline': 'archived model (G = 6, x = 0.4-20)', 'baseline_diag2': 'archived model, refined diagnostics',
        'chilocal': 'e+- mass correction at the local temperature', 'x80': 'G = 8, x up to 80',
        'x80_diag2': 'G = 8, x up to 80, refined diagnostics', 'boosted': 'nu-nu target: equilibrium in a common rest frame',
        'dthalf': 'H dt and CFL numbers halved', 'dtall': 'all limits halved, incl. kappa dt <= 0.25',
        'x80boost': 'G = 8, x up to 80, with the boosted nu-nu target',
        'x80boostchi': 'G = 8, x up to 80, boosted nu-nu target and local e+- correction'}
DC = {10.0: 0.519406, 5.0: 0.524938, 3.0: 0.523063}
rows = []
for f in sorted(glob.glob(os.path.join(HERE, 'res_test_*.json'))):
    r = json.load(open(f)); d = r['diag']
    expect = 'dispersed' if r['delta'] < DC[r['T0']] else 'BH'
    ok = 'same' if r['status'] == expect and not r['fallback'] else 'CHANGED'
    rows.append(f"{r['label']:15s} T_H = {r['TH']:.2f} MeV  delta_m = {r['delta']:.5f}  {r['status']:9s} (archived: {expect:9s}) {ok:7s} "
                f"eps = {r['eps']:.2f}  max|q_s|/eps_s (kappa_n > H) {d['qmax']:.3f}, T_b > 1 MeV {d['qmax_hot']:.3f}; "
                f"top-group fraction {d['top']:.2f}; min I/(eps_s/4pi) {d.get('Imin_rel', float('nan')):.3f}   [{DESC.get(r['label'], '')}]")
txt = ('Bracket tests of the multigroup model (App. C, Tests). Archived thresholds: T_H = 1.00, 0.51, 0.31 MeV -> '
       '0.519406, 0.524938, 0.523063.\nThe max|q_s|/eps_s column counts only groups holding >= 0.1% of the cell energy. The first '
       'baseline, chilocal and x80 (T_H = 1 MeV) runs used an earlier version of the diagnostic without this filter and without the '
       'minimum intensity (nan); the filter matters only on the x80 grid, whose nearly empty top groups gave meaningless ratios '
       '(compare the x80_diag2 rows).\n\n' + '\n'.join(rows) + '\n')
open(os.path.join(HERE, 'summary.txt'), 'w').write(txt); print(txt)
