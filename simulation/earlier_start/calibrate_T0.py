"""Start temperatures T0 for the earlier-start test: for each treatment and starting time eps0, find the plasma
temperature at the start that gives the same temperature at horizon crossing T_H as the archived run started at
eps0 = 0.1. T_H depends only on the background, so the probes use a coarse grid and stop just after eps = 1.
First checks that the probe reproduces the archived T_H at eps0 = 0.1. Writes T0_calibration.json."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import run_eos, run_eosc

# archived runs started at eps0 = 0.1: (treatment, T0) -> T_H in MeV
ARCHIVED = {('pf', 0.5): 'res_eos_pf_0.5.json', ('pf', 1.0): 'res_eos_pf_1.0.json',
            ('nu', 0.5): 'res_eosc_1.0_0.5.json', ('nu', 1.0): 'res_eosc_1.0_1.0.json'}


def TH(kind, T0, eps0):
    if kind == 'pf':
        out = run_eos.run(0.3, kind='pf', T0=T0, K=2, N=40, eps0=eps0, cH=0.001, eps_max=1.02)
    else:
        out = run_eosc.run(0.3, T0, Tdec=1.0, N=40, K=4, eps0=eps0, cH=0.001, eps_max=1.02)
    return out['TH']


if __name__ == '__main__':
    result = {}
    for (kind, T0), fn in ARCHIVED.items():
        target = json.load(open(os.path.join(HERE, '..', fn)))['TH']
        probe = TH(kind, T0, 0.1)
        print(f'{kind} T0={T0} eps0=0.1: archived T_H={target:.5f}, probe {probe:.5f} ({100*(probe/target-1):+.2f}%)',
              flush=True)
        for eps0 in (0.05, 0.025):
            T = T0*0.1/eps0
            for it in range(6):                       # T_H is nearly proportional to T0: fixed-point iteration
                th = TH(kind, T, eps0)
                if abs(th/target - 1) < 2e-4:
                    break
                T *= target/th
            print(f'    eps0={eps0}: T0={T:.5f} MeV gives T_H={th:.5f} ({100*(th/target-1):+.3f}%)', flush=True)
            result[f'{kind}_{T0}_{eps0}'] = dict(kind=kind, T0_archived=T0, eps0=eps0, T0=T, TH=th, TH_archived=target)
    json.dump(result, open(os.path.join(HERE, 'T0_calibration.json'), 'w'), indent=1)
