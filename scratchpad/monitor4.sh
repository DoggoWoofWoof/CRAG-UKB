cd /c/Users/Swastik/Desktop/CRAG
for i in $(seq 1 300); do   # ~300*30s = 2.5h
  if grep -qa "\[c5\] DONE" scratchpad/c5_full3.log; then echo "EVENT=C5_DONE"; break; fi
  if [ "$(grep -ac 'PERMANENTLY FAILED' scratchpad/c5_full3.log)" -ge 8 ]; then echo "EVENT=PERMFAIL"; break; fi
  h=$(grep -a healthy scratchpad/c5_full3.log | tail -1 | grep -oE "healthy [0-9]+/" | grep -oE "[0-9]+")
  if [ -n "$h" ] && [ "$h" -le 2 ]; then echo "EVENT=HEALTHY_CRITICAL=$h"; break; fi
  if ! kill -0 9417 2>/dev/null; then echo "EVENT=PROC_GONE"; break; fi
  sleep 30
done
echo "=== snapshot ==="; grep -a healthy scratchpad/c5_full3.log | tail -1
echo "OK=$(grep -ac '\[c5\] OK' scratchpad/c5_full3.log) PERMFAIL=$(grep -ac 'PERMANENTLY FAILED' scratchpad/c5_full3.log)"
