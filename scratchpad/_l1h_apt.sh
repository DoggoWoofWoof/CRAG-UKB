#!/usr/bin/env bash
# L1 host lane: install the PHG toolchain in the lab host's WSL Ubuntu-24.04 (authorized by the user in chat, 2026-09-29:
# "bro i give you authorization please start utilizing the gpu and cpu dont waste time", answering the apt-install question).
# Mirrors the laptop's authorized command (results/L1_LOWMEM/PHG_PACKAGES.json: apt update; apt install -y
# libtrilinos-zoltan-dev libopenmpi-dev openmpi-bin) plus gcc, which this distro lacks. Records before / simulate / after.
set -u
OUT="results/L1_HOST/apt"
mkdir -p "$OUT"
LOG="$OUT/apt_$(date -u +%Y%m%dT%H%M%SZ).log"
PKGS="libtrilinos-zoltan-dev libopenmpi-dev openmpi-bin gcc"
{
  echo "== utc $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname) user $(id -un) distro:"; cat /etc/os-release | head -n 4
  echo "== before: dpkg-query"; dpkg-query -W $PKGS libc6-dev 2>&1
  echo "== sudo -n true"; sudo -n true && echo "sudo ok" || { echo "SUDO_NEEDS_PASSWORD"; exit 3; }
  echo "== apt-get update"; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update 2>&1 | tail -n 25
  echo "== simulate"; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -s -y $PKGS 2>&1 | grep -E "^(Inst|Conf|Remv)|newly installed|upgraded|Need to get|After this" | head -n 400
  echo "== download size"; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install --print-uris -y -qq $PKGS 2>/dev/null | awk '{s+=$3} END {printf "%d bytes in %d files\n", s, NR}'
  echo "== install"; sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y $PKGS 2>&1 | tail -n 40
  echo "rc_install=${PIPESTATUS[0]}"
  echo "== after: dpkg-query"; dpkg-query -W $PKGS libtrilinos-zoltan-13.2 libc6-dev 2>&1
  echo "== versions"; gcc --version | head -n 1; mpicc --version | head -n 1; mpirun --version | head -n 1
  echo "== zoltan header"; grep -h "ZOLTAN_VERSION_NUMBER" /usr/include/trilinos/Zoltan_config.h /usr/include/trilinos/zoltan*.h 2>/dev/null | head -n 3
  echo "== apt-cache policy"; apt-cache policy $PKGS 2>&1 | head -n 40
  echo "== nproc $(nproc) mem"; free -g | head -n 2
  echo "== done utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$LOG" 2>&1
tail -n 30 "$LOG"
