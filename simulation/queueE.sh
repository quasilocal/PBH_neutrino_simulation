cd "$(dirname "$0")"
1086 2>/dev/null; do sleep 20; done
python3 bisect_eos.py fs 5.0 0.50 0.56 5e-4 0.25
python3 bisect_eos.py fs 3.0 0.50 0.56 5e-4 0.155
python3 bisect_eos.py fs 2.0 0.50 0.56 5e-4 0.107
