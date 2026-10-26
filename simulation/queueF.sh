cd "$(dirname "$0")"
1086 2>/dev/null; do sleep 20; done
python3 bisect_eos.py fs 1.5,1.0,0.5 0.50 0.56 5e-4
