#!/bin/sh
# incremental: pull whatever has landed in the Modal volume and score any H* cell not already
# cached in REPLAY_<ds>.json.  Safe to run repeatedly; watch_hard.sh runs the same steps at the end.
cd "C:/Users/Swastik/Desktop/CRAG" || exit 1
export MODAL_PROFILE=swathihrao28 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
python -u scratchpad/_l1ov_hard.py pull     >> scratchpad/_l1ov/log_pullscore.txt 2>&1
python -u scratchpad/_l1ov_hard.py scoreall >> scratchpad/_l1ov/log_pullscore.txt 2>&1
echo "PULLSCORE_DONE rc=$?" >> scratchpad/_l1ov/log_pullscore.txt
