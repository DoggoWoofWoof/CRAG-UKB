set -e
cd /c/Users/Swastik/Desktop/CRAG
TAGS="PM0_STRUCT PM1_STRUCT_NERX PM2_STRUCT_KNN PM3_TOPOLOGY_C P2_METIS_STRONG P3_FENNEL P4_HYPERGRAPH_CE P0_RANDOM_BALANCED_s0 P0_RANDOM_BALANCED_s1 P0_RANDOM_BALANCED_s2 P0_RANDOM_BALANCED_s3 P0_RANDOM_BALANCED_s4"
for d in "$@"; do
  python -u scratchpad/_l1ep_part.py $d >> scratchpad/_l1ep/log_part_$d.txt 2>&1
  echo "PART_${d}_DONE $(date)"
  : > scratchpad/_l1ep/log_puall_$d.txt
  for t in $TAGS; do
    python -u scratchpad/_l1ep_pu.py $d scratchpad/_l1ep/parts/${d}__${t}.npy $t >> scratchpad/_l1ep/log_puall_$d.txt 2>&1
  done
  echo "PUALL_${d}_DONE $(date)"
  python -u scratchpad/_l1ep_c.py $d PM_CURRENT_EXACT $TAGS >> scratchpad/_l1ep/log_c_$d.txt 2>&1
  echo "C_${d}_DONE $(date)"
done
echo MASTER_DONE
