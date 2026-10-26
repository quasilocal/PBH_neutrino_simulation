cd "$(dirname "$0")"
python3 bisect_gen.py fl_e005 0.0 2 0.05 400 0.48 0.52 1.0 2e-4 0.002
python3 bisect_gen.py nu409_e005 0.409 32 0.05 400 0.50 0.56 1.0 5e-4 0.002
python3 bisect_gen.py fl_q05 0.0 2 0.1 400 0.40 0.55 0.5 2e-4
python3 bisect_gen.py nu409_q05 0.409 32 0.1 400 0.42 0.60 0.5
