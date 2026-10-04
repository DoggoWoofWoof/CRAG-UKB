#!/usr/bin/env bash
# STEP 6 -> STEP 5 -> STEP 7b, in that order and stopping on the first failure.
# Validation runs first because it is the gate the freeze reads; the oracle comparison runs second
# because it is evidence, not a gate on the tables; the freeze runs last and refuses if step 6
# failed. They run sequentially rather than in parallel: each one peaks at multiple GB and the
# machine has about 6 GB usable.
set -euo pipefail
cd /c/Users/Swastik/Desktop/CRAG
D=scratchpad/final_canonical_build/webqsp_v1
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8

echo "=== STEP 6: canonical validation ==="
python $D/v3_validate_canonical.py
echo "=== STEP 5: IDIR oracle comparison ==="
python $D/v3_idir_compare.py
echo "=== STEP 7b: freeze ==="
python $D/v3_freeze_canonical.py
echo "=== ALL STEPS COMPLETE ==="
