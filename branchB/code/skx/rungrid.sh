for c in "13 23" "15 26" "19 33" "23 40" "27 47" "31 54"; do
 for b in 0.3 0.4 0.5 0.6 0.7 0.85; do
  python3 kgrid.py $c $b
 done
done
