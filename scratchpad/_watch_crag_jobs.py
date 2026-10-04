"""Emit one line whenever a crag rx job changes (state / real-running vs yield-wait / rc).  Polls `rx ls` every 60 s, exits after MAXMIN minutes or when no crag job is pending."""
import re
import subprocess
import sys
import time

RX = ["python", "C:/Users/Swastik/Desktop/message-passing-retrieval/tools/rx/rx.py"]
MAXMIN = int(sys.argv[1]) if len(sys.argv) > 1 else 55
NAMES = ("pq-", "enc", "ner", "h2l", "fbx", "wsl", "ivf")


def snap():
    out = subprocess.run(RX + ["ls", "-n", "30"], capture_output=True, text=True, timeout=120).stdout
    s = {}
    for ln in out.splitlines():
        if ln.startswith(" ") or not ln.strip() or ln.startswith("JOB"):
            continue
        f = ln.split()
        if len(f) < 6 or not any(n in f[0] for n in NAMES):
            continue
        yw = "yield wait" in ln
        rc = f[-1] if False else ""
        m = re.search(r"\s(\d+)\s+python -u", ln)
        if m:
            rc = m.group(1)
        s[f[0]] = "%s%s%s" % (f[1], " (yield-wait)" if yw else "", (" rc=" + rc) if rc else "")
    return s


prev = {}
t0 = time.time()
while time.time() - t0 < MAXMIN * 60:
    try:
        cur = snap()
    except Exception as e:  # transient rx failure
        print("poll error: %r" % (e,), flush=True)
        time.sleep(60)
        continue
    for k, v in cur.items():
        if prev.get(k) != v:
            print("%s -> %s" % (k, v), flush=True)
    prev = cur
    time.sleep(60)
print("watch window ended", flush=True)
