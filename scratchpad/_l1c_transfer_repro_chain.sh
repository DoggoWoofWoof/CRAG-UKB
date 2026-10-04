#!/bin/bash
# Streamed-stage-2 identity test of the frozen composite (fifteenth ruling; PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json): the five DEV_A caches (from the repo root).
cd /c/Users/Swastik/Desktop/CRAG || exit 1
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
OUT=results/L1_COVPART
for c in metaqa metaqa_phg squad squad_phg musique; do
  if [ -f "$OUT/transfer_repro_$c.json" ]; then echo "skip $c (exists)"; continue; fi
  echo "=== $c $(date -u +%FT%TZ)"
  python -u scratchpad/_l1c_transfer_composite.py "$c" --reproduce > "$OUT/transfer_repro_$c.log" 2>&1 || { echo "FAILED $c rc $?"; tail -5 "$OUT/transfer_repro_$c.log"; exit 1; }
  grep -E "stage-2 divergence|stage-2 gate|reference reproduced|done" "$OUT/transfer_repro_$c.log" | tail -4
done
echo CHAIN_DONE
