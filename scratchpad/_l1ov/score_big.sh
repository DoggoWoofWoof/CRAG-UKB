cd /c/Users/Swastik/Desktop/CRAG
for d in webqsp hotpotqa_clean; do
  python -u scratchpad/_l1ov_hard.py score $d CURRENT
done
echo SCORE_BIG_DONE
