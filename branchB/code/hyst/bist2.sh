while pgrep -f stairH.sh > /dev/null; do sleep 5; done
for u in 0.1050 0.1000; do python3 sweep.py skx bu$u 400000 $u; done > bist2.log 2>&1
