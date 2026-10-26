cd "$(dirname "$0")"
python3 bisect_gen.py nu409_N800 0.409 32 0.1 800 0.524 0.532 1.0
python3 bisect_gen.py nu409_r15 0.409 32 0.1 667 0.524 0.532 1.0 5e-4 0.005 15.0
python3 bisect_gen.py nu20_K32 0.20 32 0.1 400 0.49 0.53 1.0
