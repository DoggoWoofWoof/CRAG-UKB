"""COLLOC host job (step 16 / 17): ONE job = a 3-row smoke of scratchpad/_l1x_colloc<step>.py (runs every arm, the identity checks against the stored verdicts and the step-12 record, writes nothing)
and ONLY IF it passes the full run, which is write-once.  A smoke failure therefore costs ~3 minutes (WebQSP, measured), not the hours of the full run.
  python -u scratchpad/_colloc_chain.py <metaqa|webqsp> <tag> [step=16|17]"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print(time.strftime("[chain %H:%M:%S]"), " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        print(time.strftime("[chain %H:%M:%S]"), "FAILED rc=%d: %s" % (r.returncode, " ".join(cmd[2:])), flush=True)
        sys.exit(r.returncode)


def main():
    ds, tag = sys.argv[1], sys.argv[2]
    step = next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("step=")), "16")
    base = [sys.executable, "-u", "scratchpad/_l1x_colloc%s.py" % step, "RUN", ds, tag]
    run(base + ["--rows=3"])
    run(base)
    print(time.strftime("[chain %H:%M:%S]"), "COLLOC_CHAIN PASS", flush=True)


if __name__ == "__main__":
    main()
