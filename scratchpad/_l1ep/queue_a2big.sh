set -e
cd /c/Users/Swastik/Desktop/CRAG
KND=results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_KNN
# the kNN-audit chain owns the big-corpus embeddings; do not run a second heavy traversal beside it
while [ ! -f $KND/diag/k8_hotpotqa_clean.json ] || [ ! -f $KND/diag/k8_webqsp.json ]; do sleep 120; done
echo "KNN_CHAIN_DONE $(date)"
for d in webqsp hotpotqa_clean; do
  python -u scratchpad/_l1ep_a2.py $d 1 > scratchpad/_l1ep/log_a2_$d.txt 2>&1
  echo "A2_${d}_DONE $(date)"
done
echo QUEUE_A2BIG_DONE
