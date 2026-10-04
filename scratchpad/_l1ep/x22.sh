set -e
cd /c/Users/Swastik/Desktop/CRAG
# minimal 2x2:  {E0_STRUCT, E6_TOPOLOGY_C} x {shipped, best candidate partitioning}
# metaqa   biggest partition effect (KB)
# webqsp   the other KB, confirms the class
# 2wiki    the ONLY corpus where an edge substrate reached significance
run () {
  python -u scratchpad/_l1ep_x.py "$1" PM_CURRENT_EXACT E0_STRUCT E6_TOPOLOGY_C >> scratchpad/_l1ep/log_x_$1.txt 2>&1
  python -u scratchpad/_l1ep_x.py "$1" "$2"             E0_STRUCT E6_TOPOLOGY_C >> scratchpad/_l1ep/log_x_$1.txt 2>&1
  echo "X22_$1_DONE $(date)"
}
for spec in "$@"; do
  d="${spec%%:*}"; t="${spec##*:}"
  if python -c "
import json,os,sys
p='results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_EDGE_AND_PARTITION_FINAL/INTERACTION/X_'+sys.argv[1]+'.json'
r=json.load(open(p)) if os.path.exists(p) else {}
sys.exit(0 if ('PM_CURRENT_EXACT' in r and sys.argv[2] in r) else 1)" "$d" "$t"; then
    echo "X22_${d}_SKIP_ALREADY_DONE"
  else
    run "$d" "$t"
  fi
done
echo "X22_ALL_DONE $(date)"
