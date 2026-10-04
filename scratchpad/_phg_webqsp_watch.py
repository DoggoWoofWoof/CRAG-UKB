"""Preflight-gated wait loop for the WebQSP PHG lane (tenth ruling, 2026-09-15).  Drives the frozen program's own subcommands only:

  preflight  (one sample; rc 0 = READY after 3 consecutive clean samples, rc 2 = keep waiting)
  inputs -> prereg -> run -> report   when READY   (inputs / run refuse with rc 3 while contended -> back to waiting)
  not_run -> report                   when the armed wait budget is exhausted (deadline_passed rc 0)

Never signals any process.  Log: results/L1_LOWMEM/logs/phg_webqsp_watch.log."""
import io
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROG = os.path.join(REPO, "src", "l1_lowmem", "phg_webqsp.py")
LOG = os.path.join(REPO, "results", "L1_LOWMEM", "logs", "phg_webqsp_watch.log")
ENV = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
SPACING = 60          # the program's own wait-loop spacing (sample_spacing_seconds_wait_loop)


def log(s):
    line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), s)
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def sub(cmd, timeout=None):
    """run one program subcommand; stdout/stderr tail into the watch log; returns rc."""
    log("> %s" % cmd)
    r = subprocess.run([sys.executable, "-u", PROG, cmd], cwd=REPO, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    out = (r.stdout or "").strip().splitlines()
    err = (r.stderr or "").strip().splitlines()
    for ln in out[-40:]:
        log("  | " + ln)
    for ln in err[-40:]:
        log("  ! " + ln)
    log("< %s rc %d" % (cmd, r.returncode))
    return r.returncode


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log("watch start pid %d" % os.getpid())
    while True:
        rc = sub("preflight", timeout=600)
        if rc == 0:
            rc_i = sub("inputs", timeout=6 * 3600)
            if rc_i == 3:
                log("inputs refused: contended -> keep waiting")
                time.sleep(SPACING)
                continue
            if rc_i != 0:
                log("inputs failed rc %d -> STOP (human decision required)" % rc_i)
                return 10
            rc_p = sub("prereg", timeout=3600)
            if rc_p != 0:
                log("prereg failed rc %d -> STOP (human decision required)" % rc_p)
                return 11
            rc_r = sub("run", timeout=None)
            if rc_r == 3:
                log("run refused at its first preflight: contended -> keep waiting")
                time.sleep(SPACING)
                continue
            rc_rep = sub("report", timeout=3600)
            log("chain finished: run rc %d report rc %d -> STOP" % (rc_r, rc_rep))
            return 0 if (rc_r == 0 and rc_rep == 0) else 12
        if sub("deadline_passed", timeout=120) == 0:
            rc_n = sub("not_run", timeout=1200)
            rc_rep = sub("report", timeout=3600)
            log("budget exhausted: not_run rc %d report rc %d -> STOP" % (rc_n, rc_rep))
            return 0 if (rc_n == 0 and rc_rep == 0) else 13
        time.sleep(SPACING)


if __name__ == "__main__":
    sys.exit(main())
