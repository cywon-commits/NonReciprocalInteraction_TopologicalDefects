python pair_escape.py '{"ell": 30, "cfac": 0.003}'
python pair_escape.py '{"ell": 30, "cfac": 0.01}'
for l in 10 100 300 1000 3000; do python pair_escape.py "{\"ell\": $l}"; done
python pair_escape.py '{"ell": 100, "alpha": 0.05, "beta": 0.1}'
python pair_escape.py '{"ell": 100, "alpha": 0.9, "beta": 0.3}'
for l in 30 100 1000; do python pair_escape.py "{\"ell\": $l, \"R0\": 8}"; done
