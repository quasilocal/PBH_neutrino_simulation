cd "$(dirname "$0")"
python3 bisect_mg.py phys6 40 0.4975 0.5015 4e-4 mode=phys G=6 x_lo=0.4 x_hi=20
python3 bisect_mg.py phys6 3 0.521 0.527 4e-4 mode=phys G=6 x_lo=0.4 x_hi=20
