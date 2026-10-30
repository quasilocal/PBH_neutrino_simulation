cd "$(dirname "$0")"
python3 bisect_mg.py phys6 5 0.518 0.530 4e-4 mode=phys G=6 x_lo=0.4 x_hi=20
python3 bisect_mg.py phys6 7 0.519 0.527 4e-4 mode=phys G=6 x_lo=0.4 x_hi=20
