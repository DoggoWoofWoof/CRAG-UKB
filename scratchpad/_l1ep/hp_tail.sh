set -e
cd /c/Users/Swastik/Desktop/CRAG
L=scratchpad/_l1ep/log_master_big.txt
# hotpot is the corpus that refuted the corpus-class reading, so it gets the 2x2 as well.
# Sequenced AFTER the main chain's hotpot Phase C so the two never write/read C_hotpot at once.
until grep -q "C_hotpotqa_clean_DONE" $L; do sleep 20; done
echo "MAIN_HOTPOT_C_SEEN $(date)"
for t in PM_CURRENT_EXACT P4_CE_LOCAL_ONLY; do
  python -u scratchpad/_l1ep_x.py hotpotqa_clean $t >> scratchpad/_l1ep/log_x_hotpotqa_clean.txt 2>&1
  echo "X22_hotpot_${t}_DONE $(date)"
done
echo "X22_hotpot_DONE $(date)"
for s in derive an returns report tables; do
  python -u scratchpad/_l1ep_$s.py > scratchpad/_l1ep/log_regen_$s.txt 2>&1
  echo "REGEN_${s}_DONE"
done
echo "ALL_DONE $(date)"
