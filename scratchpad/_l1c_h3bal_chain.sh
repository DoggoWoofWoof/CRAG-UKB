#!/usr/bin/env bash
# BALANCED_H3_PATCH1 (fourteenth ruling, item 3): the five DEV_A caches in order, write-once, then the pre-registered summary.
cd "C:/Users/Swastik/Desktop/CRAG" || exit 1
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
for c in metaqa metaqa_phg squad squad_phg musique; do
  if [ -f "results/L1_COVPART/h3bal_A_$c.json" ]; then echo "=== $c exists, skipped"; continue; fi
  echo "=== $c $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python -u scratchpad/_l1c_h3bal.py "$c" > "results/L1_COVPART/h3bal_A_$c.log" 2>&1 || { echo "FAILED $c rc=$?"; tail -n 30 "results/L1_COVPART/h3bal_A_$c.log"; exit 2; }
  grep -E "BALANCED_H3_PATCH1 |labels|done" "results/L1_COVPART/h3bal_A_$c.log" | tail -n 3
done
echo "=== summary $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -u scratchpad/_l1c_h3bal_summary.py > results/L1_COVPART/h3bal_SUMMARY.log 2>&1 || { echo "FAILED summary rc=$?"; tail -n 30 results/L1_COVPART/h3bal_SUMMARY.log; exit 3; }
tail -n 8 results/L1_COVPART/h3bal_SUMMARY.log
echo CHAIN_DONE
