set -e
cd /c/Users/Swastik/Desktop/CRAG
d=$1
for t in PM0_STRUCT PM1_STRUCT_NERX PM2_STRUCT_KNN PM3_TOPOLOGY_C P2_METIS_STRONG P3_FENNEL P4_HYPERGRAPH_CE P0_RANDOM_BALANCED_s0 P0_RANDOM_BALANCED_s1 P0_RANDOM_BALANCED_s2 P0_RANDOM_BALANCED_s3 P0_RANDOM_BALANCED_s4; do
  python -u scratchpad/_l1ep_pu.py $d scratchpad/_l1ep/parts/${d}__${t}.npy $t >> scratchpad/_l1ep/log_puall_$d.txt 2>&1
done
echo "PUALL_${d}_DONE"
