#!/usr/bin/env bash
# L1 host lane diagnostic (read-only): what the PHG mpirun processes are doing in WSL Ubuntu-24.04.
echo "== utc $(date -u +%FT%TZ) $(hostname)"
echo "== procs"; ps -eo pid,ppid,stat,etime,pcpu,rss,wchan:24,args --sort=start_time | grep -E "mpirun|phg_driver|orted|prte|rank_wrap|/usr/bin/time" | grep -v grep | cut -c1-260 | head -n 60
echo "== ip"; ip -br addr 2>&1 | head -n 20
echo "== routes"; ip route 2>&1 | head -n 10
echo "== ptrace_scope $(cat /proc/sys/kernel/yama/ptrace_scope 2>&1)"
echo "== lscpu"; lscpu | grep -E "^(Socket|Core|Thread|CPU\(s\)|NUMA node\(s\))"
echo "== mpirun wchan / fds"
for p in $(pgrep -x mpirun | head -n 3); do
  echo "-- mpirun $p $(grep -E '^State' /proc/$p/status) wchan=$(cat /proc/$p/wchan 2>/dev/null)"
  ls -l /proc/$p/fd 2>/dev/null | awk '{print $NF}' | sort | uniq -c | sort -rn | head -n 8
done
for p in $(pgrep -f "phg_build_crag/phg_driver" | head -n 4); do
  echo "-- driver $p $(grep -E '^State|^VmRSS' /proc/$p/status | tr '\n' ' ') wchan=$(cat /proc/$p/wchan 2>/dev/null)"
done
echo "== done"
