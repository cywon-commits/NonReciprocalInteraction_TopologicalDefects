for u in 0.0710 0.0725 0.0750 0.0775; do python3 sweep.py helix hfix$u 400000 $u; done > hfix.log 2>&1
