cd /c/Users/Swastik/Desktop/CRAG
while [ "$(ps -ef | grep -c "[m]odal_partition")" -gt 0 ]; do sleep 60; done
echo MODAL_ALL_EXITED $(date)
python -u scratchpad/_l1ov_hard.py pull
python -u scratchpad/_l1ov_hard.py scoreall
python -u scratchpad/_l1ov_hard.py gate
echo HARD_ALL_DONE $(date)
