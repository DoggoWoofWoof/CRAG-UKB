set -e
cd /c/Users/Swastik/Desktop/CRAG
TAGS="PM0_STRUCT PM1_STRUCT_NERX PM2_STRUCT_KNN PM3_TOPOLOGY_C P2_METIS_STRONG P3_FENNEL P4_HYPERGRAPH_CE P4_CE_LOCAL_ONLY P4_CE_UNWEIGHTED P4_CE_NER_ONLY PM4_TOPOLOGY_C_NERW PM3_TOPOLOGY_C_seed1 PM3_TOPOLOGY_C_seed2 PM3_TOPOLOGY_C_seed3 PM3_TOPOLOGY_C_seed4 P0_RANDOM_BALANCED_s0 P0_RANDOM_BALANCED_s1 P0_RANDOM_BALANCED_s2 P0_RANDOM_BALANCED_s3 P0_RANDOM_BALANCED_s4"
# resumable: build exactly the tags that get scored, and skip any tag already recorded
todo () { python -c "
import json,os,sys
ds,which,tags=sys.argv[1],sys.argv[2],sys.argv[3:]
f={'B':'PARTITION_UTILITY/B_','C':'INTERACTION/C_'}[which]
p='results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_EDGE_AND_PARTITION_FINAL/'+f+ds+'.json'
have=set(json.load(open(p))) if os.path.exists(p) else set()
print(' '.join(t for t in tags if t not in have))" "$1" "$2" $TAGS; }

for d in "$@"; do
  python -u scratchpad/_l1ep_part.py $d $TAGS >> scratchpad/_l1ep/log_part_$d.txt 2>&1
  echo "PART_${d}_DONE $(date)"
  B=$(todo $d B)
  if ! python -c "
import json,os,sys
p='results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_EDGE_AND_PARTITION_FINAL/PARTITION_UTILITY/B_'+sys.argv[1]+'.json'
sys.exit(0 if os.path.exists(p) and 'P1_METIS_CURRENT' in json.load(open(p)) else 1)" $d; then
    python -u scratchpad/_l1ep_pu.py $d CURRENT P1_METIS_CURRENT >> scratchpad/_l1ep/log_puall_$d.txt 2>&1
  fi
  for t in $B; do
    python -u scratchpad/_l1ep_pu.py $d scratchpad/_l1ep/parts/${d}__${t}.npy $t >> scratchpad/_l1ep/log_puall_$d.txt 2>&1
  done
  echo "PUALL_${d}_DONE $(date)"
  C=$(todo $d C)
  if ! python -c "
import json,os,sys
p='results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_EDGE_AND_PARTITION_FINAL/INTERACTION/C_'+sys.argv[1]+'.json'
sys.exit(0 if os.path.exists(p) and 'PM_CURRENT_EXACT' in json.load(open(p)) else 1)" $d; then
    C="PM_CURRENT_EXACT $C"
  fi
  if [ -n "$C" ]; then
    python -u scratchpad/_l1ep_c.py $d $C >> scratchpad/_l1ep/log_c_$d.txt 2>&1
  fi
  echo "C_${d}_DONE $(date)"
done
echo MASTER3_DONE
