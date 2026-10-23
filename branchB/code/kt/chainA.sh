for T in 0.17 0.19 0.21 0.23; do for init in cry liq; do NUMBA_NUM_THREADS=1 python3 run_kt2.py 0.0 $T $init 25000 31; done; done
