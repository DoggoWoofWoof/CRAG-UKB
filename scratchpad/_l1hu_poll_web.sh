#!/bin/bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MODAL_PROFILE=spanishorgay
for i in $(seq 1 300); do
  if modal volume ls crag-partition hmeta 2>&1 | grep -q "webqsp__H4_SPLIT_PRESERVE__SK.json"; then
    echo "DONE at iter $i"
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT after 300 iters"
exit 1
