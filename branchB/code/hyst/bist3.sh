while pgrep -f bist1.sh > /dev/null; do sleep 5; done
for u in 0.1050 0.0990; do python3 sweep.py skx bd$u 200000 $u S_sdn_0.1100.npy; done > bist3.log 2>&1
