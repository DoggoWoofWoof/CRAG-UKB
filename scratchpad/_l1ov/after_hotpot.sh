cd /c/Users/Swastik/Desktop/CRAG
until grep -q "SWEEP_ALL_DONE" scratchpad/_l1ov/log_sweep.txt; do sleep 20; done
echo "sweep complete $(date)"
python -u scratchpad/_l1ov_matched.py hotpotqa_clean
python -u scratchpad/_l1ov_audit.py hotpotqa_clean
echo AFTER_HOTPOT_DONE
