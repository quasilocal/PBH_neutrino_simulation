cd "$(dirname "$0")"
while pgrep -f "q_mg4.sh" >/dev/null; do sleep 15; done
python3 bisect_mg.py nonunu6 10 0.5230 0.5300 4e-4 mode=phys G=6 x_lo=0.4 x_hi=20 nunu=0
