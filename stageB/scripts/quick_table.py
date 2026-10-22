#!/usr/bin/env python3
"""One-line summaries of run directories (object-based rates). usage: quick_table.py [--tau 5] DIR..."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skynuc.analysis import analyse_objects, fmt
args = sys.argv[1:]
taus = [5.0]
if args and args[0] == "--tau":
    taus = [float(x) for x in args[1].split(",")]; args = args[2:]
for d in args:
    for t in taus:
        print(fmt(analyse_objects(d, t)))
