set -e
cd /c/Users/Swastik/Desktop/CRAG
# small corpora get every family; the two big ones get the decisive cells only
for d in musique_clean squad_clean 2wiki_clean metaqa; do
  python -u scratchpad/_l1ov_eval.py $d CURRENT O0_CORE O1_STRUCT O2_NERX O3_KNN O4_FULL_C b0.25 b0.5 b1.0 \
      >> scratchpad/_l1ov/log_sweep_$d.txt 2>&1
  echo "SWEEP_${d}_DONE $(date)"
done
for d in webqsp hotpotqa_clean; do
  python -u scratchpad/_l1ov_eval.py $d CURRENT O0_CORE O1_STRUCT O4_FULL_C b0.25 b0.5 b1.0 \
      >> scratchpad/_l1ov/log_sweep_$d.txt 2>&1
  echo "SWEEP_${d}_DONE $(date)"
done
echo SWEEP_ALL_DONE
