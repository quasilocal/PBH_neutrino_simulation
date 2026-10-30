cd "$(dirname "$0")"
python3 bisect_eosc.py 3.0 80,160 0.48 0.53 3e-4
python3 bisect_eos.py pf 80.0,160.0 0.48 0.52 2e-4
