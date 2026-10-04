#!/usr/bin/env bash
# The pre-registered FLAT_BALANCED_H2 ablation (results/L1_COVPART/PREREGISTRATION_FLAT_BALANCED_H2.json): the three fresh
# one-shot transfers (stage G gold-free, then stage E) and the five non-selecting DEV_A diagnostics, sequential, stop at the
# first failure; then the summary.  Write-once logs and records.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
run() {  # run <tag> <args...>
  tag="$1"; shift
  log="results/L1_COVPART/flath2_${tag}.log"
  if [ -e "$log" ] || [ -e "results/L1_COVPART/flath2_${tag}.json" ]; then echo "write-once: $tag exists"; exit 1; fi
  python -u scratchpad/_l1c_flath2.py "$@" > "$log" 2>&1
  rc=$?
  echo "$tag exit $rc"
  if [ $rc -ne 0 ]; then tail -n 20 "$log"; exit $rc; fi
}
run G_webqsp webqsp G
run E_webqsp webqsp E
run A_metaqa metaqa
run A_metaqa_phg metaqa_phg
run G_hotpotqa hotpotqa G
run E_hotpotqa hotpotqa E
run G_2wiki 2wiki G
run E_2wiki 2wiki E
run A_squad squad
run A_squad_phg squad_phg
run A_musique musique
if [ -e results/L1_COVPART/flath2_SUMMARY.log ] || [ -e results/L1_COVPART/flath2_SUMMARY.json ]; then echo "write-once: summary exists"; exit 1; fi
python -u scratchpad/_l1c_flath2_summary.py > results/L1_COVPART/flath2_SUMMARY.log 2>&1
echo "summary exit $?"
tail -n 20 results/L1_COVPART/flath2_SUMMARY.log
