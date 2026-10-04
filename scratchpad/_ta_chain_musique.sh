#!/bin/sh
for spec in "BASE 0" "A1b 0" "A2 32" "A2dOnly 32" "A1a 32"; do
  set -- $spec
  echo "=== musique_clean $1 M=$2 ==="
  python scratchpad/_ta_down.py musique_clean "$1" "$2" 2>&1 | grep -E "GATE|scope delta|label consistency|AGG \{|Error|Traceback"
done
echo "=== MUSIQUE DOWNSTREAM CHAIN DONE ==="
