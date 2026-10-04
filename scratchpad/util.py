"""One command for utilization across the board: laptop CPU / RAM / disk, host CPU / RAM / GPU, and the busiest processes on each.

Usage:  python scratchpad/util.py            -> one reading (about 10 s: three 2-second counter samples per machine, run in parallel)
        python scratchpad/util.py --watch 60 -> repeat every 60 s until Ctrl-C

The host is reached with rx exec (MSYS_NO_PATHCONV set for Git Bash); a host that cannot be reached prints the error and the laptop
section still shows.  Read-only: it queries performance counters and nvidia-smi, starts nothing on either machine.
"""
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RX = "C:/Users/Swastik/Desktop/message-passing-retrieval/tools/rx/rx.py"

# One PowerShell body for both machines: single quotes only (it travels as one argv item through rx to the host).
BODY = "; ".join([
    "$n=[Environment]::ProcessorCount",
    "$s=Get-Counter '\\Processor(_Total)\\% Processor Time','\\Process(*)\\% Processor Time','\\Process(*)\\ID Process' -SampleInterval 2 -MaxSamples 3 -ErrorAction SilentlyContinue",
    "$tot=@(); $pp=@{}; $nm=@{}",
    "foreach($x in $s){$im=@{}; foreach($c in $x.CounterSamples){if($c.Path -like '*\\id process'){$im[$c.InstanceName]=[int]$c.CookedValue}}; "
    "foreach($c in $x.CounterSamples){$i=$c.InstanceName; "
    "if($c.Path -like '*\\processor(_total)\\*'){$tot+=$c.CookedValue} "
    "elseif($c.Path -like '*\\% processor time' -and $i -ne '_total' -and $i -ne 'idle' -and $im.ContainsKey($i)){$k=$im[$i]; $pp[$k]+=$c.CookedValue/3/$n; $nm[$k]=$i}}}",
    "$cl=@{}; Get-CimInstance Win32_Process|%{$cl[[int]$_.ProcessId]=$_.CommandLine}",
    "$m=($tot|Measure-Object -Average).Average",
    "'CPU   {0,5:N1}%  ({1} logical cores; samples {2})' -f $m,$n,(($tot|%{'{0:N0}' -f $_}) -join '/')",
    "$o=Get-CimInstance Win32_OperatingSystem; $tk=$o.TotalVisibleMemorySize/1MB; $u=$tk-$o.FreePhysicalMemory/1MB",
    "'RAM   {0:N1} of {1:N1} GB used ({2:N0}%)' -f $u,$tk,(100*$u/$tk)",
    "$d=Get-PSDrive C; 'DISK  C: {0:N1} GB free' -f ($d.Free/1GB)",
    "if(Get-Command nvidia-smi -ErrorAction SilentlyContinue){"
    "nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader|%{$f=$_ -split ', '; "
    "'GPU   {0}: {1} busy, memory-bus {2}, VRAM {3} / {4}, {5} C, {6}' -f $f[0],$f[1],$f[2],$f[3],$f[4],$f[5],$f[6]}; "
    "$g=@(nvidia-smi --query-compute-apps=pid --format=csv,noheader|?{$_}); 'GPU   {0} compute process(es) attached' -f $g.Count"
    "} else {'GPU   none (no NVIDIA driver)'}",
    "'busiest processes (share of the whole machine):'",
    "$pp.GetEnumerator()|sort Value -Descending|select -First 5|%{$c=[string]$cl[$_.Key]; if($c.Length -gt 90){$c=$c.Substring(0,90)}; '  {0,5:N1}%  {1} (pid {2})  {3}' -f $_.Value,$nm[$_.Key],$_.Key,$c}",
])


def run_laptop(out):
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", BODY], capture_output=True, text=True, timeout=120)
        out["LAPTOP"] = (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        out["LAPTOP"] = "laptop query failed: %s" % e


def run_host(out):
    try:
        env = dict(os.environ, MSYS_NO_PATHCONV="1")
        r = subprocess.run([sys.executable, RX, "exec", "--shell", "powershell", "--no-ws", "--timeout", "90", "--", BODY],
                           capture_output=True, text=True, timeout=150, cwd=REPO, env=env)
        out["HOST (rx)"] = (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        out["HOST (rx)"] = "host query failed: %s" % e


def once():
    out = {}
    ts = [threading.Thread(target=run_laptop, args=(out,)), threading.Thread(target=run_host, args=(out,))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("== %s ==" % time.strftime("%Y-%m-%d %H:%M:%S"))
    for k in ("LAPTOP", "HOST (rx)"):
        print("\n--- %s ---" % k)
        print(out.get(k, ""))
    print(flush=True)


def main():
    if "--watch" in sys.argv:
        every = float(sys.argv[sys.argv.index("--watch") + 1])
        while True:
            once()
            time.sleep(every)
    once()


if __name__ == "__main__":
    main()
