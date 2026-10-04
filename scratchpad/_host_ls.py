"""Host housekeeping -- read-only listing: per-entry size/file count of directories on the lab host (rx workspace-relative, or `wsl:<sub>` = the host WSL crag_ooc ext4 tree, listed by a wsl.exe call because the
Windows job cannot see the distro's ext4 through a share) + drive free space + the WSL python.
   python -u scratchpad/_host_ls.py <dir> [<dir> ...]"""
import os
import shutil
import subprocess
import sys
import time

DISTRO = os.environ.get("FBX_WSL_DISTRO", "Ubuntu-24.04")
WSL_ROOT = "/home/student2/crag_ooc"
WS = os.getcwd()


def wsl(cmd):
    r = subprocess.run(["wsl.exe", "-d", DISTRO, "--exec"] + cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return (r.stdout + r.stderr).strip()


def size_of(p):
    n = b = 0
    newest = 0.0
    for dp, dn, fn in os.walk(p):
        for f in fn:
            try:
                st = os.lstat(os.path.join(dp, f))
            except OSError:
                continue
            n += 1
            b += st.st_size
            newest = max(newest, st.st_mtime)
    return n, b, newest


print("drive free %.1f GB of %.1f GB at %s" % (shutil.disk_usage(WS).free / 1e9, shutil.disk_usage(WS).total / 1e9, time.strftime("%F %T")), flush=True)
for d in sys.argv[1:]:
    if d.startswith("wsl:"):
        p = WSL_ROOT + ("/" + d[4:] if d[4:] else "")
        print("== %s" % p, flush=True)
        print(wsl(["ls", "-la", "--time-style=long-iso", p]), flush=True)
        print(wsl(["du", "-s", "--block-size=1M", p]), "(MB)", flush=True)
        if d == "wsl:":
            print(wsl(["df", "-h", "/home"]), flush=True)
            print(wsl(["python3", "--version"]), flush=True)
            print(wsl(["which", "python3", "tar", "sha256sum"]), flush=True)
        continue
    root = os.path.join(WS, d.replace("/", os.sep))
    print("== %s (%s)" % (d, root), flush=True)
    if not os.path.isdir(root):
        print("   absent", flush=True)
        continue
    for e in sorted(os.listdir(root))[:60]:
        p = os.path.join(root, e)
        if os.path.isdir(p):
            n, b, nw = size_of(p)
            print("   D %-46s %7d files %9.3f GB  newest %s" % (e, n, b / 1e9, time.strftime("%m-%d %H:%M", time.localtime(nw)) if nw else "-"), flush=True)
        else:
            st = os.stat(p)
            print("   F %-46s %7s       %9.3f GB  mtime  %s" % (e, "", st.st_size / 1e9, time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime))), flush=True)
