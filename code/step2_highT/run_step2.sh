#!/usr/bin/env bash
# 2단계: 고온 계수 검증 (문서 6절 '검증 기록 (2단계)'). 인자: T lam dJ nsteps L R
for a in "2.5 0.5" "5 0.5" "2.5 1" "5 1" "2.5 2" "10 0.5"; do
  set -- $a; python hiT_check.py $1 $2 0.4 12000 8 256 > run_T$1_l$2.json
done
# 긴 계산 (cos 위상 성분 포함)
python hiT_check.py 2.5 0.5 0.4 36000 8 256 > ph_T2.5.json
python hiT_check.py 5   0.5 0.4 36000 8 256 > ph_T5.json
