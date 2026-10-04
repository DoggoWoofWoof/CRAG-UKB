set -e
cd /c/Users/Swastik/Desktop/CRAG
D=hotpotqa_clean
T=P4_CE_LOCAL_ONLY
# the shipped baseline first (needed as the comparison anchor + the C1 parity gate),
# then the one partitioning that decides whether the text class holds at 4/4.
python -u scratchpad/_l1ep_pu.py $D CURRENT P1_METIS_CURRENT >> scratchpad/_l1ep/log_puall_$D.txt 2>&1
python -u scratchpad/_l1ep_pu.py $D scratchpad/_l1ep/parts/${D}__${T}.npy $T >> scratchpad/_l1ep/log_puall_$D.txt 2>&1
echo "PU_DECISIVE_DONE $(date)"
python -u scratchpad/_l1ep_c.py $D PM_CURRENT_EXACT $T >> scratchpad/_l1ep/log_c_$D.txt 2>&1
echo "C_DECISIVE_DONE $(date)"
