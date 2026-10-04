#!/usr/bin/env bash
# Unattended completion of 2wiki/splade.
#
# TWO PASSES, DELIBERATELY, NOT ONE
#
#   Shard 22 is built first and alone because it is the only one of the eleven missing shards
#   that reaches 2wiki's declared evaluation split (dev = query positions [167454, 180030)).
#   Shards 20 and 21 already exist, so shard 22 completes the dev pool -- the pool a candidate
#   ceiling is actually measured over -- roughly seven hours before the full artifact closes.
#   The cost of the split is one extra streamed corpus pass (~1 h of document loading); the
#   arithmetic is identical either way, because a score-matrix row depends on one query only.
#
#   The other ten shards (2,3,4,6,7,10,11,12,15,17) are train-only, and ride in one shared pass.
set -u
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0
cd /c/Users/Swastik/Desktop/CRAG
DST=scratchpad/_splade_parts/2wiki
TRAIN=2,3,4,6,7,10,11,12,15,17

echo "=== waiting for shard 22 (running in its own process)"
while [ ! -s "$DST/part_0022_of_0024.meta.json" ]; do
  if grep -qE "Traceback|MemoryError|Killed|SystemExit" scratchpad/_shard22.log; then
    echo "ABORT: shard 22 build reported a failure"; tail -20 scratchpad/_shard22.log; exit 1
  fi
  sleep 30
done
echo "SHARD22_READY"
python - <<'PY'
import numpy as np, json, io
z = np.load("scratchpad/_splade_parts/2wiki/part_0022_of_0024.npz")
m = json.load(io.open("scratchpad/_splade_parts/2wiki/part_0022_of_0024.meta.json",
                      encoding="utf-8"))
ids = z["ids"]
print("   shard 22: ids %s  rows %d..%d  kernel=%s  fallback_rows=%s"
      % (ids.shape, m["lo"], m["hi"], m.get("kernel"), m.get("prefilter_fallback_rows")))
assert ids.shape == (m["hi"] - m["lo"], m["K"]), "row/K shape disagrees with the meta"
assert int(ids.min()) >= 0, "negative canonical id -- a padded key survived"
print("   dev rows this shard supplies: %d" % (min(m["hi"], 180030) - max(m["lo"], 167454)))
PY
[ $? -eq 0 ] || { echo "ABORT: shard 22 failed its own shape/id checks"; exit 1; }

echo; echo "=== ten train-only shards in one shared pass: $TRAIN"
python -u src/dataset_canonical/build_splade_parts_local.py --ds 2wiki --qshards 24 --only "$TRAIN"
BE=$?
echo "TRAIN_BUILD_EXIT=$BE"
[ "$BE" -eq 0 ] || { echo "ABORT: train-shard build failed"; exit 1; }

echo; echo "=== all 24 parts should now exist; handing off to the record chain"
bash scratchpad/_finish_2wiki.sh
echo "FINISH_EXIT=$?"
