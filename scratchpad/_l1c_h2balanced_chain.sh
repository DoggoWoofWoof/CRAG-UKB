#!/bin/bash
# BALANCED_H2_PATCH1 chain: the five caches then the pre-registered summary (from the repo root).
cd /c/Users/Swastik/Desktop/CRAG || exit 1
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
OUT=results/L1_COVPART
for c in metaqa metaqa_phg squad squad_phg musique; do
  if [ -f "$OUT/h2balanced_A_$c.json" ]; then echo "skip $c (exists)"; continue; fi
  echo "=== $c $(date -u +%FT%TZ)"
  python -u scratchpad/_l1c_h2balanced.py "$c" > "$OUT/h2balanced_A_$c.log" 2>&1 || { echo "FAILED $c rc $?"; tail -5 "$OUT/h2balanced_A_$c.log"; exit 1; }
  tail -2 "$OUT/h2balanced_A_$c.log"
done
python -u scratchpad/_l1c_h2balanced.py --summary > "$OUT/h2balanced_A_SUMMARY.log" 2>&1 || { echo "FAILED summary"; exit 1; }
cat "$OUT/h2balanced_A_SUMMARY.log"
echo CHAIN_DONE
