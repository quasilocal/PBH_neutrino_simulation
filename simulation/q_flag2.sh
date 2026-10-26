cd "$(dirname "$0")"
python3 bisect_gen.py pf_flag 0.0 32 0.1 400 0.49 0.51 1.0 2.5e-4
python3 bisect_eosc_flag.py 2.0 1.0 0.48 0.55 3e-4
