set -e
cd /c/Users/Swastik/Desktop/CRAG
# wait for the small-corpus Phase B/C chains so METIS on 4-8M edges does not race them for RAM
while ! grep -q MASTER2_DONE scratchpad/_l1ep/log_master_2wiki.txt 2>/dev/null; do sleep 60; done
echo "SMALL_BC_DONE $(date)"
bash scratchpad/_l1ep/master2.sh hotpotqa_clean webqsp
echo QUEUE_BIG_DONE
