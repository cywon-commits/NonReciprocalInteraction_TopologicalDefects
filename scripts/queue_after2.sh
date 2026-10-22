#!/bin/bash
# wait until batch 3 has finished, then run the 1C dependence set
cd "$(dirname "$0")/.."
while pgrep -f "queue_after.sh|run_batch.py" > /dev/null; do sleep 30; done
exec scripts/run_batch.py -j 4 configs/dep/dep_T0.45_P1.json configs/dep/dep_T0.55_P1.json configs/dep/dep_T0.55_P4.json configs/dep/dep_L128_P1.json --resume
