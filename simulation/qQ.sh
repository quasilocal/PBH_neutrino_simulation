cd "$(dirname "$0")"
python3 bisect_eosc.py 3.0 40,20,10,7,5,3,2,1.5,1.0,0.5 0.48 0.545 3e-4
python3 bisect_eosc.py 2.0 3,2,1.5,1.0,0.5 0.48 0.545 3e-4
python3 bisect_eos.py pf 40.0,20.0,10.0,7.0 0.48 0.52 2e-4
