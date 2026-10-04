"""Capacity-gated submitter (generic): polls the rx scheduler and submits ONE job only when the host admits its reservation, so nothing ever sits in the rx queue (user rule: no waiting jobs); then waits for it and fetches the outputs.
Laptop-side, no host load.
  python -u scratchpad/_rx_gate.py --name NAME --mem GB --cpus N [--hard GB] [--poll 90] [--outputs GLOB] -- <command run under scratchpad/_host_yield.py on the host>
Idempotent per NAME: does nothing if a job whose name contains NAME is already running / queued."""
import os
import re
import subprocess
import sys
import time

RX = [sys.executable, "C:/Users/Swastik/Desktop/message-passing-retrieval/tools/rx/rx.py"]


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def sh(cmd, t=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=t, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    return (r.stdout or "") + (r.stderr or "")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def main():
    assert "--" in sys.argv, __doc__
    cut = sys.argv.index("--")
    name, mem, cpus, poll = arg("--name", ""), arg("--mem", 0.0), arg("--cpus", 0.0), arg("--poll", 240)
    hard, outputs = arg("--hard", mem + 8), arg("--outputs", "")
    cmd = sys.argv[cut + 1:]
    assert name and mem and cpus and cmd, __doc__
    log("gate %s: waiting for %.1f GB / %.1f cpus" % (name, mem, cpus))
    while True:
        try:
            if any(name in ln and (" running " in ln or " queued " in ln) for ln in sh(RX + ["ls"], 60).splitlines()):
                log("a %s job is already running / queued: nothing to do" % name)
                return
            m = re.search(r"scheduler:\s*([\d.]+)/([\d.]+) cpus,\s*([\d.]+)/([\d.]+) GB", sh(RX + ["monitor", "--once"], 90))
            if m:
                cu, cc, mu, mc = map(float, m.groups())
                log("scheduler: free %.1f cpus, %.1f GB" % (cc - cu, mc - mu))
                if mc - mu >= mem + 0.3 and cc - cu >= cpus:
                    run = RX + ["run", "-n", name, "--cpus", str(cpus), "--mem", str(mem), "--mem-hard", str(hard)] + (["--outputs", outputs] if outputs else []) + ["--", "python", "-u", "scratchpad/_host_yield.py", "run", "--"] + cmd
                    out = sh(run, 300)
                    log("SUBMITTED:", out.strip().replace("\n", " | ")[:400])
                    job = next((t for t in reversed(out.split()) if re.fullmatch(r"\d{6}-\d{6}-.+", t)), None)
                    if job:
                        log("waiting for", job)
                        log(sh(RX + ["wait", job, "--fetch"], 86400 * 2).strip().replace("\n", " | ")[-600:])
                    return
        except Exception as e:  # transient ssh / rx errors: keep polling
            log("poll error:", repr(e)[:200])
        time.sleep(poll)


if __name__ == "__main__":
    main()
