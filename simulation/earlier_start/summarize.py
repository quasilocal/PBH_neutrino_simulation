"""Table of the earlier-start test (bisect_early.py) next to the archived runs started at eps0 = 0.1 (and 0.05 for
pure radiation, Table II). Writes summary.txt. Thresholds are delta_c +- half the bisection bracket. Shifts are
relative to the perfect fluid at the same mass and starting time, in delta_c and in the linear compaction
C_l,c = (4/3)[1 - (1 - 3 delta_c/2)^(1/2)]."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, '..')


def load(path):
    d = json.load(open(path))
    return d['dc'], d['w']/2, d.get('at_edge', False)


arch = lambda fn: load(os.path.join(SIM, fn))
new = lambda label: load(os.path.join(HERE, f'res_{label}.json'))
Cl = lambda d: (4/3)*(1 - (1 - 1.5*d)**0.5)
fmt = lambda v: f'{v[0]:.5f}+-{v[1]:.5f}' + (' EDGE' if v[2] else '')

ROWS = [('pure radiation', {0.1: (arch('res_fluid.json'), arch('res_nu409_fine.json')),
                            0.05: (arch('res_fl_e005.json'), arch('res_nu409_e005.json')),
                            0.025: (new('rad_pf_e0.025'), new('rad_nu_e0.025'))}),
        ('heaviest, T_H=0.0625 MeV', {0.1: (arch('res_eos_pf_0.5.json'), arch('res_eosc_1.0_0.5.json')),
                                      0.05: (new('eos_pf_T0.5_e0.05'), new('eos_nu_T0.5_e0.05')),
                                      0.025: (new('eos_pf_T0.5_e0.025'), new('eos_nu_T0.5_e0.025'))}),
        ('dip, T_H=0.117 MeV', {0.1: (arch('res_eos_pf_1.0.json'), arch('res_eosc_1.0_1.0.json')),
                                0.05: (new('eos_pf_T1_e0.05'), new('eos_nu_T1_e0.05')),
                                0.025: (new('eos_pf_T1_e0.025'), new('eos_nu_T1_e0.025'))})]
cal = json.load(open(os.path.join(HERE, 'T0_calibration.json')))
lines = ['Earlier start: thresholds against the starting time eps0 (the archived runs start at eps0 = 0.1).',
         'Start temperatures T0 (MeV) at eps0 = 0.05 and 0.025, tuned to keep T_H (T0_calibration.json): '
         + ', '.join(f'{k} {v["T0"]:.5f}' for k, v in cal.items()), '',
         '   case                     eps0   perfect fluid      neutrinos          shift in delta_c   shift in C_l,c']
for name, d in ROWS:
    for eps0 in (0.1, 0.05, 0.025):
        pf, nu = d[eps0]
        lines.append(f'   {name:24s} {eps0:<6} {fmt(pf):18s} {fmt(nu):18s} {100*(nu[0]/pf[0] - 1):6.3f}%'
                     f'            {100*(Cl(nu[0])/Cl(pf[0]) - 1):6.3f}%')
text = '\n'.join(lines)
print(text)
open(os.path.join(HERE, 'summary.txt'), 'w', encoding='utf-8').write(text + '\n')
