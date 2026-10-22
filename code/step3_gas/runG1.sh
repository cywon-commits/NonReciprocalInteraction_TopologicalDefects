for l in 5 50 inf; do python pair_gas.py "{\"ell\": \"$l\", \"Gamma\": 2.2e-8, \"L\": 1500, \"tmax\": 170000, \"tburn\": 20000, \"seed\": 11}"; done
python pair_gas.py '{"ell": 15, "Gamma": 2.2e-8, "L": 1500, "tmax": 170000, "tburn": 20000, "R0": 8, "seed": 12}'
