"""One entry point to check and reproduce the results of the paper. Works on Linux, macOS and Windows.

    python reproduce.py check      quick checks against numbers quoted in the paper (about 5 min)
    python reproduce.py figures    redraw the five figures from the stored results (under a minute)
    python reproduce.py tables     recompute the mass functions, the Table IV ratios, the seed tables (Table V) and the
                                   review checks from the stored thresholds, and compare with the stored outputs
                                   (about 2 min)
    python reproduce.py all        all three
    python reproduce.py list       the commands for the long simulations behind each table and figure

Option --latex renders the figure text with LaTeX (needs a TeX installation with type1cm and cm-super); by default
Matplotlib's built-in mathtext is used. Everything is written to reproduced/; no shipped file is changed: the
'tables' step works on a copy of simulation/ in reproduced/work/.
"""
import os, re, sys, json, time, shutil, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SIM, FIGS = os.path.join(ROOT, 'simulation'), os.path.join(ROOT, 'paper', 'figs')
OUT = os.path.join(ROOT, 'reproduced')
LOGS = os.path.join(OUT, 'logs')


def run(script, cwd, log, env_extra=None):
    """Run a Python script with this interpreter, echo its output and save it to reproduced/logs/<log>."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', **(env_extra or {}))
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    print(f'\n--- {os.path.relpath(os.path.join(cwd, script), ROOT)}', flush=True)
    p = subprocess.Popen([sys.executable, script], cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    lines = []
    for line in p.stdout:
        print('   ' + line, end='', flush=True)
        lines.append(line)
    p.wait()
    text = ''.join(lines)
    with open(os.path.join(LOGS, log), 'w', encoding='utf-8') as f:
        f.write(text)
    status = 'ok' if p.returncode == 0 else f'FAILED (exit code {p.returncode})'
    print(f'    {status}, {time.time() - t0:.0f} s', flush=True)
    return p.returncode == 0, text


NUM = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


def compare_numbers(a, b):
    """Compare the numbers in two texts: (number of numbers, largest relative difference, or None if the counts differ)."""
    x, y = NUM.findall(a), NUM.findall(b)
    if len(x) != len(y):
        return len(x), None
    worst = 0.0
    for u, v in zip(x, y):
        u, v = float(u), float(v)
        if u != v:
            worst = max(worst, abs(u - v) / max(abs(u), abs(v)))
    return len(x), worst


def flatten(obj):
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from flatten(obj[k])
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from flatten(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)


def compare_json(f_new, f_old):
    a, b = list(flatten(json.load(open(f_new)))), list(flatten(json.load(open(f_old))))
    if len(a) != len(b):
        return len(a), None
    worst = max((abs(u - v) / max(abs(u), abs(v)) for u, v in zip(a, b) if u != v), default=0.0)
    return len(a), worst


def report(label, n, worst):
    if worst is None:
        print(f'    {label}: DIFFERENT STRUCTURE (number of values differs)')
    else:
        print(f'    {label}: {n} numbers, largest relative difference {worst:.1e}'
              + ('  (identical)' if worst == 0 else ''))


def check():
    ok, _ = run('run_tests.py', SIM, 'run_tests.txt')
    print('\nCompare each line with the value quoted in it; our own output is in '
          'simulation/review_tests/log_run_tests.txt.')
    return ok


def figures(latex):
    figdir = os.path.join(OUT, 'figures')
    env = {'PBH_FIGDIR': figdir, 'PBH_USETEX': '1' if latex else '0'}
    ok = True
    for name in ('dynamics', 'linear', 'scaling', 'mass', 'massfn'):
        ok &= run(f'fig_{name}.py', FIGS, f'fig_{name}.txt', env)[0]
    print(f'\nFigures written to {os.path.relpath(figdir, ROOT)}; compare them with paper/fig_*.pdf.')
    return ok


def tables():
    work = os.path.join(OUT, 'work')
    if os.path.isdir(work):
        shutil.rmtree(work)
    # work on copies, so that the shipped results stay untouched; the scripts find their inputs by relative paths
    shutil.copytree(SIM, os.path.join(work, 'simulation'), ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(FIGS, os.path.join(work, 'paper', 'figs'), ignore=shutil.ignore_patterns('__pycache__'))
    wsim, wfigs = os.path.join(work, 'simulation'), os.path.join(work, 'paper', 'figs')
    ok = True
    # mass functions (Fig. 5a and the log-normal numbers of Sec. V A)
    ok &= run('mass_function.py', os.path.join(wsim, 'massfn'), 'mass_function.txt')[0]
    report('massfn/mass_functions.json', *compare_json(os.path.join(wsim, 'massfn', 'mass_functions.json'),
                                                         os.path.join(SIM, 'massfn', 'mass_functions.json')))
    # numbers of Secs. IV D and V A, Table IV
    ok &= run('numbers_mass.py', wfigs, 'numbers_mass.txt')[0]
    ok &= run('numbers_stats.py', wfigs, 'numbers_stats.txt')[0]
    # Table V and the window-edge factors of Sec. V B
    for name in ('seed_window', 'seed_bounds', 'seed_edges'):
        good, text = run(f'{name}.py', os.path.join(wsim, 'seeds'), f'{name}.txt')
        ok &= good
        ref = os.path.join(SIM, 'seeds', f'{name}_out.txt')
        if good and os.path.exists(ref):
            report(f'seeds/{name}_out.txt', *compare_numbers(text, open(ref, encoding='utf-8').read()))
    report('seeds/seed_bounds.json', *compare_json(os.path.join(wsim, 'seeds', 'seed_bounds.json'),
                                                   os.path.join(SIM, 'seeds', 'seed_bounds.json')))
    # checks added in review: central zeta at threshold, the two slicings in linear theory, mass-weighted ratios
    for name in ('zeta_center', 'linear_slicing', 'massweight'):
        ok &= run(f'{name}.py', os.path.join(wsim, 'review_checks'), f'{name}.txt')[0]
    print(f'\nOutputs and logs are in {os.path.relpath(OUT, ROOT)}; the printed numbers are the ones quoted in '
          'Secs. IV D and V and in Tables IV and V.')
    return ok


LONG = """Long simulations (run from simulation/; one core unless stated; times measured on one core of a 2-core machine).
Each writes a res_*.json file; the archived ones are the results quoted in the paper. See README.md for details.

  Table I, Table II, Fig. 3   python bisect_gen.py <tag> <R_nu> <N_mu> <eps0> <N> <lo> <hi> <alpha> 2.5e-4   0.1-3.5 min each
                              e.g. python bisect_gen.py nu409_q4 0.409 32 0.1 400 0.54 0.70 4.0 2.5e-4
                              gradual decoupling: python bisect_dec.py ...
  Fig. 4, single rate         python bisect_eosc.py <T_dec> <T0 list> 0.48 0.55 3e-4               3-4 min per mass
                              perfect fluid: python bisect_eos.py pf <T0 list> ...
  Fig. 4, multigroup          cd multigroup, python weak_rates.py (Monte Carlo rates), cd .., then
                              python bisect_mg.py phys6 <T0> <lo> <hi> 4e-4 mode=phys G=6 x_lo=0.4 x_hi=20
                                                                                                  about 1 h per mass
  Table III                   python bh_mass.py 0.0 0.49767 0.1,0.03,0.01,0.003,0.001,0.0003,0.0001
                              python bh_mass.py 0.409 0.52832 0.1,0.03,0.01,0.003,0.001,0.0003,0.0001
  Fig. 2                      python lin_offcentre.py                                             4 min
  App. C tests                python review_tests/bracket_mg.py (about 30 runs of 10 min, on 2 cores)
                              python review_tests/vrel_mg.py, python review_tests/bisect_eosc_N400.py
The shell scripts q_*.sh and queue*.sh list the exact brackets we used; on Windows run their python lines by hand
(or use WSL)."""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    latex = '--latex' in sys.argv
    what = args[0] if args else 'help'
    if what == 'list':
        print(LONG)
        return
    steps = {'check': [check], 'figures': [lambda: figures(latex)], 'tables': [tables],
             'all': [check, lambda: figures(latex), tables]}.get(what)
    if steps is None:
        print(__doc__)
        return
    t0 = time.time()
    ok = all([step() for step in steps])
    print(f"\n{'All steps finished without errors' if ok else 'SOME STEPS FAILED (see above)'} "
          f"in {(time.time() - t0)/60:.1f} min.")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
