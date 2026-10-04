set -e
cd /c/Users/Swastik/Desktop/CRAG
# decisive tags only: is the P4 weighting effect a KB-class phenomenon or a MetaQA quirk?
T="PM3_TOPOLOGY_C PM3_TOPOLOGY_C_seed1 P4_CE_LOCAL_ONLY P4_HYPERGRAPH_CE P4_CE_UNWEIGHTED P0_RANDOM_BALANCED_s0"
python -u scratchpad/_l1ep_part.py webqsp $T >> scratchpad/_l1ep/log_part_webqsp.txt 2>&1
echo "WEBQSP_FOCUS_PART_DONE $(date)"
python -u scratchpad/_l1ep_c.py webqsp PM_CURRENT_EXACT $T >> scratchpad/_l1ep/log_c_webqsp.txt 2>&1
echo "WEBQSP_FOCUS_C_DONE $(date)"
