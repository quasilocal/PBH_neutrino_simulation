cd "$(dirname "$0")"
nice -n 19 python3 bisect_eosc.py 1.0 3,2,1.5,1.0,0.5 0.505 0.535 2.6e-4
