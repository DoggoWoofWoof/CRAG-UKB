#!/usr/bin/env bash
# Fires the whole downstream chain the moment 2wiki/splade's 24 parts exist.
#
# `modal volume get` exits NON-ZERO on Windows even when the transfer succeeded, so success is
# judged by counting local files, never by $?.  Each step below refuses on its own if its input
# is incomplete -- this script does not re-implement those checks, it just sequences them.
set -u
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0
export MODAL_PROFILE=extra_ip9HxU
cd /c/Users/Swastik/Desktop/CRAG

# Atomic single-run lock.  mkdir either creates the directory or fails; there is no window in
# which two callers both think they created it.  A stale lock is a deliberate manual decision to
# clear, not something this script guesses at.
if ! mkdir scratchpad/_finish_2wiki.lock 2>/dev/null; then
  echo "LOCKED: scratchpad/_finish_2wiki.lock exists -- another run holds the chain. Exiting."
  exit 3
fi
trap 'rmdir scratchpad/_finish_2wiki.lock 2>/dev/null' EXIT

N=24
DST=scratchpad/_splade_parts/2wiki
mkdir -p "$DST"

# This script no longer pulls.  The 24 parts were produced in THREE workspaces -- 13 in
# extra_ip9HxU before it was disabled at 13/24, then 6 in extra_wNzonK and 5 in crm -- and
# `crag-retrieval-out` is a different volume in each, so there is no single profile to pull
# from.  The pulls are done per-workspace outside this script; here the parts are only counted.
echo "=== checking $N local parts in $DST"
HAVE=$(ls -1 "$DST"/part_*_of_00$N.npz 2>/dev/null | wc -l)
META=$(ls -1 "$DST"/part_*_of_00$N.meta.json 2>/dev/null | wc -l)
echo "local parts: $HAVE npz / $META meta  (need $N of each)"
if [ "$HAVE" -ne "$N" ] || [ "$META" -ne "$N" ]; then
  MISSING=""
  for i in $(seq 0 $((N-1))); do
    f=$(printf 'part_%04d_of_%04d' "$i" "$N")
    [ -s "$DST/$f.npz" ] && [ -s "$DST/$f.meta.json" ] || MISSING="$MISSING $i"
  done
  echo "ABORT: incomplete --  missing:$MISSING"
  exit 1
fi

echo; echo "=== merge"
python scratchpad/merge_splade_parts.py 2wiki:$N || exit 1

echo; echo "=== verify (local recomputation over the full 5,989,847-doc corpus)"
python src/dataset_canonical/verify_heavy_caches.py 2wiki splade || exit 1

echo; echo "=== 1/6 retrieval cache record (additive-drift adjudication + update)"
python src/dataset_canonical/retrieval_cache_record_update.py --apply || exit 1

echo; echo "=== 2/6 frozen caveat supersession (expect COMPLETE=true now)"
python src/dataset_canonical/frozen_caveat_supersession.py || exit 1

echo; echo "=== 3/6 manifest cache slots"
python src/dataset_canonical/manifest_cache_slots.py --apply || exit 1

echo; echo "=== 4/6 prose docs"
python src/dataset_canonical/doc_cache_status_patch.py --apply || exit 1

echo; echo "=== 4b/6 K-semantics record (its per-dataset cache-presence block was measured
            before the caches landed and reads present=false for the three"
python src/dataset_canonical/k_semantics_and_splits.py || exit 1

echo; echo "=== 5/6 manifest verification WITH --full (the record it replaces ran full: sha256 of all 206 declared artifacts, ~29 GB; dropping it would swap a stronger record for a weaker one)"
python src/dataset_canonical/verify_manifest.py --records --sizes --graphs --builders --full \
  --out=data/final_canonical/UKB_COMMON_MANIFEST_VERIFICATION.json
echo "verify_manifest exit=$?"

echo; echo "=== 6/6 package bytes ledger (re-measured: +1.156 GB for 2wiki splade)"
python src/dataset_canonical/package_bytes_ledger.py || exit 1

echo; echo "CHAIN_DONE"
