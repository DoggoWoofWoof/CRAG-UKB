"""Zoltan-PHG lane -- package/version evidence for the WSL environment, recorded BEFORE and AFTER the authorized apt install.

    python -u src/l1_lowmem/phg_env.py before      -> results/L1_LOWMEM/PHG_PACKAGES.json  {"before": {...}}
    python -u src/l1_lowmem/phg_env.py install     -> runs exactly the two authorized commands as the distro root user (sudo needs a
                                                      password here, which the assistant never types); appends {"install": {...}}
    python -u src/l1_lowmem/phg_env.py after       -> appends {"after": {...}} (+ header / library discovery)
The record is what the PHG preregistration pins.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, DISTRO, log, rj, wj  # noqa: E402

REC = os.path.join(OUT, "PHG_PACKAGES.json")
EVIDENCE_COMMANDS = ["apt-cache policy libtrilinos-zoltan-dev libopenmpi-dev openmpi-bin", "mpirun --version", "mpicxx --version",
                     "dpkg-query -W libtrilinos-zoltan-dev libtrilinos-zoltan-13.2 libopenmpi-dev openmpi-bin"]
AUTHORIZED_INSTALL = ["apt update", "apt install -y libtrilinos-zoltan-dev libopenmpi-dev openmpi-bin"]


def wslrun(cmd, user=None, timeout=1800):
    args = ["wsl", "-d", DISTRO] + (["-u", user] if user else []) + ["-e", "bash", "-lc", cmd]
    t = time.time()
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return {"command": cmd, "user": user or "default", "rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-4000:],
            "seconds": round(time.time() - t, 1)}


def evidence():
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "distro": DISTRO, "commands": [wslrun(c) for c in EVIDENCE_COMMANDS]}
    out["os_release"] = wslrun("grep -E '^(NAME|VERSION|VERSION_ID)=' /etc/os-release; uname -r; nproc; free -m | head -2")["stdout"]
    dq = out["commands"][3]
    out["dpkg_versions"] = {ln.split("\t")[0]: (ln.split("\t")[1] if "\t" in ln else "") for ln in dq["stdout"].split("\n") if ln.strip()}
    return out


def before():
    rec = rj(REC) or {"RECORD": "PHG_PACKAGES", "authorized_by_user": AUTHORIZED_INSTALL,
                      "authorization_note": "the user authorized exactly these Ubuntu downloads (2026-09-13); nothing else is fetched"}
    if "before" in rec:
        raise RuntimeError("'before' evidence already recorded")
    rec["before"] = evidence()
    wj(REC, rec)
    for c in rec["before"]["commands"]:
        log("BEFORE rc=%d  %s -> %s" % (c["rc"], c["command"], (c["stdout"] or c["stderr"]).split("\n")[0][:120]))


def install():
    rec = rj(REC)
    if rec is None or "before" not in rec:
        raise RuntimeError("record the 'before' evidence first")
    if "install" in rec and all(c["rc"] == 0 for c in rec["install"]["commands"]):
        raise RuntimeError("install already recorded as successful")
    rec["install"] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "how": "wsl -d %s -u root -e bash -lc '<command>' (the WSL distro's root user; no password is typed by the assistant)" % DISTRO,
                      "commands": []}
    for c in AUTHORIZED_INSTALL:
        r = wslrun("DEBIAN_FRONTEND=noninteractive " + c, user="root", timeout=3600)
        r["stdout_tail"] = r.pop("stdout").split("\n")[-40:]
        rec["install"]["commands"].append(r)
        log("INSTALL rc=%d  %s  (%.0fs)" % (r["rc"], c, r["seconds"]))
        wj(REC, rec)
        if r["rc"] != 0:
            raise RuntimeError("install step failed: %s\n%s" % (c, r["stderr"]))


def after():
    rec = rj(REC)
    if rec is None or "install" not in rec:
        raise RuntimeError("run install first")
    rec["after"] = evidence()
    disc = wslrun("ls /usr/include/trilinos/zoltan.h /usr/include/trilinos/zoltan_types.h 2>&1; ls /usr/lib/x86_64-linux-gnu/libtrilinos_zoltan* 2>&1; "
                  "dpkg -L libtrilinos-zoltan-dev | grep -E 'zoltan(_types|_align|_comm)?\\.h$|\\.so$' | head -20; "
                  "grep -n 'ZOLTAN_ID_TYPE\\b' /usr/include/trilinos/zoltan_types.h | head -5; "
                  "grep -n 'ZOLTAN_VERSION_NUMBER' /usr/include/trilinos/zoltan.h | head -2")
    rec["after"]["discovery"] = disc["stdout"].split("\n")
    wj(REC, rec)
    for c in rec["after"]["commands"]:
        log("AFTER  rc=%d  %s -> %s" % (c["rc"], c["command"], (c["stdout"] or c["stderr"]).split("\n")[0][:120]))
    log("dpkg versions:", rec["after"]["dpkg_versions"])
    log("wrote", os.path.relpath(REC, REPO))


if __name__ == "__main__":
    a = sys.argv[1:]
    {"before": before, "install": install, "after": after}.get(a[0] if a else "", lambda: print(__doc__))()
