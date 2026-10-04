cd /c/Users/Swastik/Desktop/CRAG
for i in $(seq 1 60); do
  ok=$(grep -ac "\[c5\] OK" scratchpad/c5_full3.log 2>/dev/null)
  pf=$(grep -ac "PERMANENTLY FAILED" scratchpad/c5_full3.log 2>/dev/null)
  if grep -qa "\[c5\] DONE" scratchpad/c5_full3.log; then echo "EVENT=C5_DONE"; break; fi
  if [ "${ok:-0}" -ge 6 ]; then echo "EVENT=GOT_COMPLETIONS ok=$ok pf=$pf"; break; fi
  if [ "${pf:-0}" -ge 10 ]; then echo "EVENT=PERMFAIL_CLIMBING pf=$pf"; break; fi
  if ! kill -0 9417 2>/dev/null; then echo "EVENT=PROC_GONE"; break; fi
  sleep 20
done
echo "=== snapshot ==="; grep -a healthy scratchpad/c5_full3.log | tail -1
echo "OK=$(grep -ac '\[c5\] OK' scratchpad/c5_full3.log) PERMFAIL=$(grep -ac 'PERMANENTLY FAILED' scratchpad/c5_full3.log)"
echo "=== rc=0 done=[] recurrence? (the corruption signature) ==="; grep -a "rc=0 cls=TEMP_FAILED done=\[\]" scratchpad/c5_full3.log | tail -3
