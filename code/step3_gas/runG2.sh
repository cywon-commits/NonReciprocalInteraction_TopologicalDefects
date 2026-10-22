for l in 15 150; do python pair_gas.py "{\"ell\": \"$l\", \"Gamma\": 2.2e-8, \"L\": 1500, \"tmax\": 170000, \"tburn\": 20000, \"seed\": 21}"; done
python pair_gas.py '{"ell": 15, "Gamma": 8.8e-8, "L": 750, "tmax": 120000, "tburn": 15000, "seed": 22}'
python pair_gas.py '{"ell": 15, "Gamma": 2.2e-8, "L": 1500, "tmax": 170000, "tburn": 20000, "dt": 0.05, "seed": 23}'
