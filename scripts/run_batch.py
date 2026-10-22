#!/usr/bin/env python3
"""Run several nucleation configs as single-threaded processes, N at a time.

usage: scripts/run_batch.py -j 4 configs/a.json configs/b.json ... [--resume] [key=value ...]
key=value overrides are applied to every config.
"""
import os, subprocess, sys, time

args = sys.argv[1:]
nj = 4
if args and args[0] == "-j":
    nj = int(args[1]); args = args[2:]
cfgs = [a for a in args if a.endswith(".json")]
extra = [a for a in args if not a.endswith(".json")]
env = dict(os.environ, NUMBA_NUM_THREADS="1")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
running = []
queue = list(cfgs)
while queue or running:
    while queue and len(running) < nj:
        c = queue.pop(0)
        log = os.path.splitext(os.path.basename(c))[0]
        os.makedirs(os.path.join(root, "results", "logs"), exist_ok=True)
        f = open(os.path.join(root, "results", "logs", log + ".log"), "a")
        p = subprocess.Popen([sys.executable, "-m", "skynuc.nucleation", c] + extra, cwd=root, env=env,
                             stdout=f, stderr=subprocess.STDOUT)
        running.append((p, c, time.time()))
        print("start", c, flush=True)
    time.sleep(2)
    for p, c, t0 in list(running):
        if p.poll() is not None:
            running.remove((p, c, t0))
            print("done", c, "rc", p.returncode, "%.0fs" % (time.time() - t0), flush=True)
