for u in 0.1000 0.0980 0.0960; do python3 sweep.py skx bd$u 300000 $u S_sdn_0.1100.npy; done > bist1.log 2>&1
