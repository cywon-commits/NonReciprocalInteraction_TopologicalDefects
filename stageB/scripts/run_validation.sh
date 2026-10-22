#!/bin/bash
# Runs V1 (two alphas), V2 and V3/V4 as four single-threaded processes.
cd "$(dirname "$0")/.."
export NUMBA_NUM_THREADS=1
mkdir -p results/validation
python3 -m skynuc.validate_v12 V1 '{}' > results/validation/V1_a0.1.json 2>results/validation/V1_a0.1.err &
python3 -m skynuc.validate_v12 V1 '{"alpha":0.5}' > results/validation/V1_a0.5.json 2>results/validation/V1_a0.5.err &
python3 -m skynuc.validate_v12 V2 '{}' > results/validation/V2.json 2>results/validation/V2.err &
python3 -m skynuc.single '{}' > results/validation/V34.log 2>&1 &
wait
