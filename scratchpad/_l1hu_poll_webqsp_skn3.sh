#!/bin/bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MODAL_PROFILE=spanishorgay
miss=0
for i in $(seq 1 180); do
  if modal volume ls crag-partition hmeta 2>&1 | grep -q "webqsp__H4_SPLIT_PRESERVE__SKN.json"; then
    echo "DONE at iter $i"
    exit 0
  fi
  out=$(modal app list 2>&1)
  if echo "$out" | grep -qi "connect call failed\|connection\|timed out\|getaddrinfo\|gaierror"; then
    echo "TRANSIENT_NETWORK_ERROR at iter $i, ignoring and retrying"
    sleep 60
    continue
  fi
  if echo "$out" | grep -q "ap-kZUq71EcSclzdDf8hYQ41d.*ephemeral"; then
    miss=0
  else
    miss=$((miss+1))
    echo "miss $miss at iter $i"
    if [ $miss -ge 3 ]; then
      echo "APP_NO_LONGER_RUNNING at iter $i (confirmed over 3 consecutive checks)"
      exit 2
    fi
  fi
  sleep 60
done
echo "TIMEOUT after 180 iters"
exit 1
