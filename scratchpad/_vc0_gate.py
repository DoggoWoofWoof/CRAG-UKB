"""Capacity-gated submitter for the level-0 V-cycle (FBX_SCALE addendum 18): polls the rx scheduler and submits the job ONLY when the host admits the reservation, so no job ever sits in the rx queue (user rule: no waiting jobs).
Laptop-side, no host load.   python -u scratchpad/_vc0_gate.py [--mem 47] [--cpus 4] [--anon 46] [--poll 90]
Idempotent: does nothing if an ooc-fb-vc0 job is already running / queued."""
import re
import subprocess
import sys
import time

RX = [sys.executable, "C:/Users/Swastik/Desktop/message-passing-retrieval/tools/rx/rx.py"]
ENV = {"MSYS_NO_PATHCONV": "1"}


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


MEM, CPUS, ANON, POLL = arg("--mem", 47.0), arg("--cpus", 4.0), arg("--anon", 46), arg("--poll", 240)
CAP, RESERVE_MEM = 119.7, 0.0


def sh(cmd, t=120):
    import os
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=t, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env={**os.environ, **ENV})
    return (r.stdout or "") + (r.stderr or "")


def sched():
    out = sh(RX + ["monitor", "--once"], 90)
    m = re.search(r"scheduler:\s*([\d.]+)/([\d.]+) cpus,\s*([\d.]+)/([\d.]+) GB", out)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))) if m else None


def already():
    out = sh(RX + ["ls"], 60)
    return any("ooc-fb-vc0" in ln and (" running " in ln or " queued " in ln) for ln in out.splitlines())


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def main():
    log("gate: waiting for %.0f GB / %.0f cpus (anon cap %d)" % (MEM, CPUS, ANON))
    while True:
        try:
            if already():
                log("an ooc-fb-vc0 job is already running / queued: nothing to do")
                return
            s = sched()
            if s:
                cu, cc, mu, mc = s
                free_c, free_m = cc - cu, mc - mu
                log("scheduler: cpus %.1f/%.0f, mem %.0f/%.1f GB reserved -> free %.1f cpus, %.1f GB" % (cu, cc, mu, mc, free_c, free_m))
                if free_m >= MEM + 0.5 and free_c >= CPUS:
                    cmd = RX + ["run", "-n", "ooc-fb-vc0", "--cpus", str(CPUS), "--mem", str(MEM), "--mem-hard", str(MEM + 12), "--",
                                "python", "-u", "scratchpad/_host_yield.py", "run", "--", "python", "-u", "scratchpad/_ooc_vc.py", "VCYCLE",
                                "/home/student2/crag_ooc/fb", "/home/student2/crag_ooc/fb_l1", "/home/student2/crag_ooc/fb_work", "4", "levels=0-0", "anon_gb=%d" % ANON]
                    out = sh(cmd, 300)
                    log("SUBMITTED:", out.strip().replace("\n", " | ")[:400])
                    return
        except Exception as e:  # keep polling through transient ssh / rx errors
            log("poll error:", repr(e)[:200])
        time.sleep(POLL)


if __name__ == "__main__":
    main()
