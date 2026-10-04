"""Host housekeeping -- READ-ONLY inventory of what the crag project occupies on the lab host: the rx workspace (Windows side), the rx job directories, the WSL engine workspace and the engine binaries.
Prints sizes and mtimes only; deletes nothing.   python -u scratchpad/_host_inventory.py"""
import os
import subprocess
import sys
import time

DISTRO = os.environ.get("FBX_WSL_DISTRO", "Ubuntu-24.04")
TOP = int(os.environ.get("INV_TOP", "45"))


def gb(b):
    return "%8.2f GB" % (b / 1e9)


def tree_size(root):
    tot, n = 0, 0
    for dp, dn, fn in os.walk(root, onerror=lambda e: None):
        for f in fn:
            try:
                tot += os.lstat(os.path.join(dp, f)).st_size
                n += 1
            except OSError:
                pass
    return tot, n


def level(root, depth, label):
    print("== %s  (%s)" % (label, root), flush=True)
    rows = []
    try:
        ents = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError as e:
        print("   cannot list: %s" % e)
        return
    for e in ents:
        if e.is_dir(follow_symlinks=False):
            s, n = tree_size(e.path)
            rows.append((s, n, e.name + "/"))
            if depth > 1 and s > 0.3e9:
                try:
                    for e2 in sorted(os.scandir(e.path), key=lambda x: x.name):
                        if e2.is_dir(follow_symlinks=False):
                            s2, n2 = tree_size(e2.path)
                            if s2 > 0.2e9:
                                rows.append((s2, n2, "   " + e.name + "/" + e2.name + "/"))
                except OSError:
                    pass
        else:
            try:
                rows.append((e.stat().st_size, 1, e.name))
            except OSError:
                pass
    for s, n, nm in sorted(rows, key=lambda r: -r[0])[:TOP]:
        if s > 5e7:
            print("   %s  %7d files  %s" % (gb(s), n, nm), flush=True)


def wsl(cmd):
    r = subprocess.run(["wsl.exe", "-d", DISTRO, "--exec", "bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("$", cmd, "\n", r.stdout, flush=True)


ws = os.getcwd()
print("cwd (rx workspace):", ws, "| python", sys.version.split()[0], "| time", time.strftime("%F %T"))
level(ws, 2, "rx workspace top")
for sub in ("data", "results", "models", "ckpt", "checkpoints"):
    p = os.path.join(ws, sub)
    if os.path.isdir(p):
        level(p, 2, "workspace/" + sub)
home = os.path.dirname(ws)
print("workspace parent:", home)
level(home, 1, "rx home / parent")
jobs = None
for cand in (os.path.join(home, "jobs"), os.path.join(os.path.dirname(home), "jobs"), os.path.join(home, ".rx", "jobs")):
    if os.path.isdir(cand):
        jobs = cand
        break
print("jobs dir:", jobs)
if jobs:
    rows = []
    for e in os.scandir(jobs):
        if e.is_dir():
            s, n = tree_size(e.path)
            rows.append((s, n, e.name, time.strftime("%F", time.localtime(e.stat().st_mtime))))
    print("   jobs total", gb(sum(r[0] for r in rows)), len(rows), "dirs")
    for s, n, nm, mt in sorted(rows, key=lambda r: -r[0])[:25]:
        if s > 2e7:
            print("   %s %7d files  %s  %s" % (gb(s), n, mt, nm))
print("== drives", flush=True)
for dr in "CDEFG":
    if os.path.exists(dr + ":\\"):
        import shutil
        t, u, f = shutil.disk_usage(dr + ":\\")
        print("   %s: total %.0f used %.0f free %.1f GB" % (dr, t / 1e9, u / 1e9, f / 1e9))
wsl("df -h /home/student2 | cat")
wsl("du -sh /home/student2/crag_ooc/*/ 2>/dev/null | cat; ls -la /home/student2/crag_ooc | cat")
wsl("for d in fb fb_l1 fb_work; do echo == $d; ls -la --time-style=+%F_%T /home/student2/crag_ooc/$d | sort -k5 -n -r | head -60 | cat; done")
wsl("ls -la /home/student2/crag_ooc/*/ 2>/dev/null | grep -v '^total' | awk '{print $5, $9}' | sort -n -r | head -5 | cat")
wsl("ls /dev/shm | head -20 | cat; du -sh /tmp /var/tmp 2>/dev/null | cat")
print("== engine binaries and scratch on the workspace drive", flush=True)
for sub in ("data/ooc", "data/fbx_zoltan"):
    p = os.path.join(ws, sub)
    if os.path.isdir(p):
        level(p, 2, sub)
