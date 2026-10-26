cd "$(dirname "$0")"
python3 disp_check.py
python3 bisect_gen.py nu488_e025 0.488 32 0.25 400 0.52 0.56 1.0
python3 bisect_gen.py fl_e025 0.0 2 0.25 400 0.48 0.52 1.0 2e-4
python3 bisect_gen.py nu30_K32 0.30 32 0.1 400 0.49 0.54 1.0
