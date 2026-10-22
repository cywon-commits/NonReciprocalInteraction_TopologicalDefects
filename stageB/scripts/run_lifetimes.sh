#!/bin/bash
# Seeded-skyrmion lifetimes on a (B, T) grid, 4 single-threaded workers.
cd "$(dirname "$0")/.."
export NUMBA_NUM_THREADS=1
for B in 0.08 0.10 0.12; do for T in 0.30 0.40 0.50; do
  echo "$B $T"
done; done | xargs -P 4 -L 1 sh -c 'python3 -m skynuc.lifetime "{\"B\": $0, \"T\": $1}"' >> results/lifetime/lifetimes.jsonl
