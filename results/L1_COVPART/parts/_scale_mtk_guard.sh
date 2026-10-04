#!/bin/bash
# run the frozen worker under an RSS cap; poll every second; kill -9 on breach
SRC="$1"; OUT="$2"; STATS="$3"; THREADS="$4"; CAP_KB="$5"; LOG="$6"
cd "$7"
python3 scratchpad/_l1hu_local_worker.py "$SRC" "$OUT" "$STATS" "$THREADS" > "$LOG" 2>&1 &
PID=$!
PEAK=0
KILLED=0
while kill -0 $PID 2>/dev/null; do
  RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
  if [ -n "$RSS" ]; then
    if [ "$RSS" -gt "$PEAK" ]; then PEAK=$RSS; fi
    if [ "$RSS" -gt "$CAP_KB" ]; then kill -9 $PID; KILLED=1; fi
  fi
  sleep 1
done
wait $PID
RC=$?
echo "GUARD peak_rss_kb=$PEAK killed=$KILLED rc=$RC"
