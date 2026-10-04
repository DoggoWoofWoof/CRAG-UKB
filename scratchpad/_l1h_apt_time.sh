#!/usr/bin/env bash
# L1 host lane: the PHG rank wrapper (src/l1_lowmem/phg_driver/rank_wrap.sh) and phg.mpirun exec /usr/bin/time -v, as on the laptop.
# Installs the Ubuntu 'time' package in WSL Ubuntu-24.04 only if /usr/bin/time is missing (same authorization as scratchpad/_l1h_apt.sh:
# the user's "bro i give you authorization please start utilizing the gpu and cpu dont waste time", 2026-09-29). Records before / after.
set -u
OUT="results/L1_HOST/apt"
mkdir -p "$OUT"
LOG="$OUT/apt_time_$(date -u +%Y%m%dT%H%M%SZ).log"
{
  echo "== utc $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname) user $(id -un)"
  echo "== before"; ls -l /usr/bin/time 2>&1; dpkg-query -W time 2>&1
  if [ -x /usr/bin/time ]; then
    echo "TIME_PRESENT (nothing installed)"
  else
    echo "== sudo -n true"; sudo -n true && echo "sudo ok" || { echo "SUDO_NEEDS_PASSWORD"; exit 3; }
    echo "== simulate"; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -s -y time 2>&1 | grep -E "^(Inst|Conf|Remv)|newly installed|upgraded"
    echo "== install"; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y time 2>&1 | tail -n 15
    echo "rc_install=${PIPESTATUS[0]}"
  fi
  echo "== after"; ls -l /usr/bin/time 2>&1; dpkg-query -W time 2>&1; /usr/bin/time --version 2>&1 | head -n 1
  echo "== done utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$LOG" 2>&1
tail -n 12 "$LOG"
