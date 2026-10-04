#!/usr/bin/env bash
# The pre-registered flat-vs-partition measurement (results/L1_COVPART/PREREGISTRATION_FLAT_VS_PARTITION.json):
# five DEV_A caches, split A, sequential, stop at the first failure; then the cross-cache summary.  Write-once logs.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
for c in metaqa metaqa_phg squad squad_phg musique; do
  log="results/L1_COVPART/flat_A_${c}.log"
  if [ -e "$log" ] || [ -e "results/L1_COVPART/flat_A_${c}.json" ]; then echo "write-once: $c exists"; exit 1; fi
  python -u scratchpad/_l1c_flat.py "$c" > "$log" 2>&1
  rc=$?
  echo "$c exit $rc"
  if [ $rc -ne 0 ]; then tail -n 20 "$log"; exit $rc; fi
done
if [ -e results/L1_COVPART/flat_SUMMARY.log ] || [ -e results/L1_COVPART/flat_SUMMARY.json ]; then echo "write-once: summary exists"; exit 1; fi
python -u scratchpad/_l1c_flat_summary.py > results/L1_COVPART/flat_SUMMARY.log 2>&1
echo "summary exit $?"
tail -n 6 results/L1_COVPART/flat_SUMMARY.log
