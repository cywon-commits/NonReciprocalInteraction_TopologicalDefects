#!/bin/bash
# wait for any running batch, then run the given configs
cd "$(dirname "$0")/.."
while pgrep -f run_batch.py > /dev/null; do sleep 30; done
exec scripts/run_batch.py -j 4 "$@" --resume
