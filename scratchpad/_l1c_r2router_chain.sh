#!/bin/bash
# R2_LEGACY_ROUTER chain (thirteenth ruling): metaqa -> squad -> musique -> the pre-registered summary (from the repo root).
cd /c/Users/Swastik/Desktop/CRAG || exit 1
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
OUT=results/L1_COVPART
for ds in metaqa squad musique; do
  if [ -f "$OUT/r2router_A_$ds.json" ]; then echo "skip $ds (exists)"; continue; fi
  echo "=== $ds $(date -u +%FT%TZ)"
  python -u scratchpad/_l1c_r2router.py "$ds" > "$OUT/r2router_A_$ds.log" 2>&1 || { echo "FAILED $ds rc $?"; tail -5 "$OUT/r2router_A_$ds.log"; exit 1; }
  tail -3 "$OUT/r2router_A_$ds.log"
done
python -u scratchpad/_l1c_r2router_summary.py > "$OUT/r2router_SUMMARY.log" 2>&1 || { echo "FAILED summary"; tail -5 "$OUT/r2router_SUMMARY.log"; exit 1; }
cat "$OUT/r2router_SUMMARY.log"
echo CHAIN_DONE
