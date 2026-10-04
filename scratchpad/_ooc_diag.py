"""FBX_SCALE addendum 17 -- read-only diagnostics of the host engine workspace (disk, file sizes, the WSL virtual disks).  python -u scratchpad/_ooc_diag.py [work_dir]"""
import os
import subprocess
import sys

DISTRO = os.environ.get("FBX_WSL_DISTRO", "Ubuntu-24.04")
WORK = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "RM" else "/home/student2/crag_ooc/fb_work"
ROOT = "/home/student2/crag_ooc/"


def w(cmd):
    r = subprocess.run(["wsl.exe", "-d", DISTRO, "--exec", "bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("$", cmd, "\n", r.stdout, flush=True)


def win(args):
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("$", " ".join(args), "\n", r.stdout[-6000:], flush=True)


if sys.argv[1:2] == ["RM"]:                                    # delete our own intermediate files (paths must be under the crag_ooc workspace, no wildcards)
    for pth in sys.argv[2:]:
        assert pth.startswith(ROOT) and ".." not in pth and "*" not in pth, pth
        w("ls -la %s && rm -f %s && echo removed" % (pth, pth))
    w("df -h /home/student2 | cat")
    raise SystemExit(0)
w("df -h /home/student2 | cat")
w("du -xsh /home/student2/* /home/student2/.[a-z]* /tmp /var/tmp /root /opt /usr /var 2>/dev/null | sort -h | tail -14 | cat")
w("du -sh /home/student2/crag_ooc/*/ 2>/dev/null | cat; ls -la /home/student2/crag_ooc/fb_l1 | head -40")
w("ls /dev/shm | head; df -h /dev/shm | cat; free -g | cat")
win(["powershell.exe", "-NoProfile", "-Command",
     "Get-PSDrive C | Format-List Used,Free; Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss | ForEach-Object { $p = Get-ItemProperty $_.PSPath; $p.DistributionName + ' ' + $p.BasePath }"])
win(["powershell.exe", "-NoProfile", "-Command",
     "Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss | ForEach-Object { $p = Get-ItemProperty $_.PSPath; $b = $p.BasePath -replace '^\\\\\\\\\\?\\\\',''; Get-ChildItem -Path $b -Filter *.vhdx -ErrorAction SilentlyContinue | ForEach-Object { $p.DistributionName + ' ' + $_.FullName + ' ' + [math]::Round($_.Length/1GB,1) + ' GB' } }"])
