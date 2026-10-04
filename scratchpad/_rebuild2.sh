set -e
for ds in webqsp hotpotqa_clean; do
  echo "=== BUILD $ds ==="
  python scratchpad/_l1bc_core.py $ds
  python -c "
import numpy as np,sys
z=np.load('results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_BACKWARD/data/bc_$ds.npz')
[z[k] for k in z.files]; print('INTEGRITY_OK $ds',len(z.files))
"
done
echo "REBUILD_DONE"
