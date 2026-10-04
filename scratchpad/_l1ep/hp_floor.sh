set -e
cd /c/Users/Swastik/Desktop/CRAG
D=hotpotqa_clean
# hotpot's reseed noise floor: without the PM3 seed replicates scored through Phase C there is
# no yardstick to judge its P4_CE_LOCAL_ONLY delta against.  Waits for the front-run to build
# them, then scores only those, leaving the main chain's remaining tags alone.
until [ -f scratchpad/_l1ep/log_hp_front.txt ] && grep -q FRONT_ALL_DONE scratchpad/_l1ep/log_hp_front.txt; do sleep 15; done
until [ -f scratchpad/_l1ep/log_hp_decisive.txt ] && grep -q C_DECISIVE_DONE scratchpad/_l1ep/log_hp_decisive.txt; do sleep 15; done
T="PM3_TOPOLOGY_C PM3_TOPOLOGY_C_seed1 PM3_TOPOLOGY_C_seed2 PM3_TOPOLOGY_C_seed3 PM3_TOPOLOGY_C_seed4 P0_RANDOM_BALANCED_s0"
for t in $T; do
  [ -f scratchpad/_l1ep/parts/${D}__${t}.npy ] || { echo "MISSING $t"; continue; }
  python -u scratchpad/_l1ep_pu.py $D scratchpad/_l1ep/parts/${D}__${t}.npy $t >> scratchpad/_l1ep/log_puall_$D.txt 2>&1
done
echo "PU_FLOOR_DONE $(date)"
python -u scratchpad/_l1ep_c.py $D $T >> scratchpad/_l1ep/log_c_$D.txt 2>&1
echo "C_FLOOR_DONE $(date)"
