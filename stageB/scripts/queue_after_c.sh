#!/bin/bash
# start the extra stage C seeds once the main stage C batch has finished
cd "$(dirname "$0")/.."
while pgrep -f "run_batch.py.*configs/c/C_T0.60_rotp" > /dev/null; do sleep 30; done
exec scripts/run_batch.py -j 4 "$@" --resume
