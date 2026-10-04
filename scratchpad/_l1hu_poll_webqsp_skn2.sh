#!/bin/bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MODAL_PROFILE=spanishorgay
for i in $(seq 1 180); do
  if modal volume ls crag-partition hmeta 2>&1 | grep -q "webqsp__H4_SPLIT_PRESERVE__SKN.json"; then
    echo "DONE at iter $i"
    exit 0
  fi
  if ! modal app list 2>&1 | grep -q "ap-kZUq71EcSclzdDf8hYQ41d.*ephemeral"; then
    echo "APP_NO_LONGER_RUNNING at iter $i"
    exit 2
  fi
  sleep 60
done
echo "TIMEOUT after 180 iters"
exit 1
