#!/usr/bin/env bash
# Step 6 re-run (corrected display_text_empty rule + isolated-node characterisation), then an
# ATTEMPTED freeze, then the mirror.
#
# The freeze is expected to REFUSE: the pre-registered IDIR oracle gate returned STOP and
# v3_freeze_canonical.py now honours that gate rather than only recording it. The refusal is the
# correct outcome pending an explicit decision, so it must not abort the mirror -- the divergence
# records are exactly what needs to reach the tracked tree.
set -uo pipefail
cd /c/Users/Swastik/Desktop/CRAG
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8
D=scratchpad/final_canonical_build/webqsp_v1

echo "=== STEP 6 (re-run): canonical validation ==="
if ! python "$D/v3_validate_canonical.py"; then
  echo "VALIDATION FAILED TO RUN -- stopping, nothing downstream is meaningful"; exit 1
fi

echo
echo "=== STEP 7: freeze attempt (refusal expected while the oracle gate says STOP) ==="
python "$D/v3_freeze_canonical.py" && echo "FREEZE_RESULT=frozen" || echo "FREEZE_RESULT=refused"

echo
echo "=== MIRROR: copy V3 records into results/data_audit ==="
python scratchpad/final_canonical_build/mirror_audit.py || echo "MIRROR_FAILED"

echo
echo "ALL STEPS ATTEMPTED"
