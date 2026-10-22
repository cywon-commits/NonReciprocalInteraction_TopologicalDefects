#!/bin/bash
# start the P5 diagnosis runs once fewer than 3 runs are active
cd "$(dirname "$0")/.."
while [ "$(pgrep -fc 'skynuc.nucleation')" -ge 3 ]; do sleep 30; done
exec scripts/run_batch.py -j 2 configs/dep/diag_P5_a0.3.json configs/dep/diag_P5_L128.json --resume
