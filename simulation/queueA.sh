cd "$(dirname "$0")"
python3 bisect_gen.py fl_e02 0.0 2 0.2 400 0.48 0.52 1.0 2e-4
python3 bisect_gen.py nu409_e02 0.409 32 0.2 400 0.50 0.56 1.0
python3 bisect_gen.py fl_q2 0.0 2 0.1 400 0.50 0.65 2.0 2e-4
python3 bisect_gen.py nu409_q2 0.409 32 0.1 400 0.52 0.70 2.0
