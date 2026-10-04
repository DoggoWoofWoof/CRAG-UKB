#!/bin/bash
# QMAX_BALANCED_H2_PATCH1 chain (fourteenth ruling, item 1): the five DEV_A caches -> the pre-registered summary (from the repo root).
cd /c/Users/Swastik/Desktop/CRAG || exit 1
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
OUT=results/L1_COVPART
for c in metaqa metaqa_phg squad squad_phg musique; do
  if [ -f "$OUT/qmaxbal_A_$c.json" ]; then echo "skip $c (exists)"; continue; fi
  echo "=== $c $(date -u +%FT%TZ)"
  python -u scratchpad/_l1c_qmaxbal.py "$c" > "$OUT/qmaxbal_A_$c.log" 2>&1 || { echo "FAILED $c rc $?"; tail -5 "$OUT/qmaxbal_A_$c.log"; exit 1; }
  tail -4 "$OUT/qmaxbal_A_$c.log"
done
python -u scratchpad/_l1c_qmaxbal_summary.py > "$OUT/qmaxbal_SUMMARY.log" 2>&1 || { echo "FAILED summary"; tail -5 "$OUT/qmaxbal_SUMMARY.log"; exit 1; }
cat "$OUT/qmaxbal_SUMMARY.log"
echo CHAIN_DONE
